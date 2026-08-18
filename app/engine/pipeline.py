"""Pipeline steps 2-7 as (nearly) pure functions over session state.

Mirrors ``research/formula_analysis_examples/entry.ipynb`` (cells 2-23):

2. deisotoping           -> per-point ion labels + per-ion charges
3. element classification-> per-ion probabilities over all 119 elements
4. knee detection        -> threshold for any element
5. highlight             -> per-point probability array for the threshold layer
6. formula determination -> ranked candidate formulas for a selected ion
7. compare               -> Plotly figure: experimental window vs theoretical pattern

Every per-ion call is wrapped in try/except (the core has a known latent bug:
``RealIsotopicDistribution.get_representation`` raises ``ValueError: max() arg is
an empty sequence`` for ions with empty ``peak_indices``); a failing ion is
skipped with a logged warning and the session continues.

Results are cached in the session state keyed by a hash of the parameters:
re-running a step with unchanged parameters returns the cached result instantly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from joblib import Parallel, delayed
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from mass_automation.deisotoping.process import MlDeisotoper
from mass_automation.formula import ELECTRON_MASS, Formula, RealIsotopicDistribution
from mass_automation.formula.check_formula import check_presence, del_isotopologues
from mass_automation.formula.determination import formula_generator_parallel
from mass_automation.utils import ELEMENT_DICT

from ..errors import PipelineError
from ..state import Session
from .downsample import DEFAULT_MAX_PTS, downsample_window

logger = logging.getLogger("medusa_web.pipeline")

# The transformer checkpoint was trained with output_dim = 119 (= Element.n_elements).
# Note: ELEMENT_DICT contains 118 element names; the model has one extra class.
# Column c of ion_probs corresponds to atomic number c + 1.
N_ELEMENTS = 119


# ---------------------------------------------------------------------- #
# Parameters (defaults = entry.ipynb values)
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeisotopeParams:
    algorithm: str = "adaptive"
    z_max: int = 3
    min_distance: float = 0.01
    threshold: float = 0.15
    delta: float = 0.007
    n1: float = 2.0
    n2: float = 6.0


@dataclass(frozen=True)
class ElementsParams:
    element: str = "Ir"


@dataclass(frozen=True)
class ThresholdParams:
    element: str = "Ir"
    source: str = "auto"  # auto (knee) | manual
    manual_value: Optional[float] = None


@dataclass(frozen=True)
class FormulaParams:
    ion_id: int
    elements: Dict[str, Tuple[int, int]]  # e.g. {"C": (0, 90), "Ir": (1, 1)}
    mass_threshold_ppm: float = 4.0
    num_workers: int = 1
    max_chunk_size: int = 10_000


FORMULA_PRESETS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "ir_system": {  # entry.ipynb preset
        "C": (0, 90), "H": (0, 90), "N": (0, 10), "O": (0, 10), "I": (0, 2), "Ir": (1, 1),
    },
    "pubchem10": {  # data/pubchem_formulas filtered_by_limits_10000
        "C": (0, 67), "H": (0, 97), "N": (0, 10), "O": (0, 17), "Cl": (0, 3),
        "I": (0, 1), "Br": (0, 2), "F": (0, 6), "S": (0, 3), "P": (0, 1),
    },
    "empty": {  # light CHNO default
        "C": (0, 60), "H": (0, 120), "N": (0, 10), "O": (0, 10),
    },
}


def params_hash(params: Any) -> str:
    payload = json.dumps(asdict(params), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()


def element_atomic_number(symbol: str) -> int:
    """1-based atomic number for a symbol like 'Ir'; raises PipelineError(400)."""
    for number, name in ELEMENT_DICT.items():
        if name == symbol:
            return number
    raise PipelineError(f"unknown element symbol {symbol!r}", 400)


def element_symbols() -> List[str]:
    return [name for _, name in sorted(ELEMENT_DICT.items())]


# ---------------------------------------------------------------------- #
# Local copies of notebook helpers (no import-path coupling to research/)
# ---------------------------------------------------------------------- #
def find_knee_threshold_numpy(probs: np.ndarray) -> Tuple[float, int]:
    """Knee point = point farthest from the chord (0, max) -> (N, min)."""
    probs = np.asarray(probs).flatten()
    n = len(probs)
    if n < 3:
        return float(probs.min()), n - 1

    y = np.sort(probs)[::-1]
    x = np.arange(n)

    x_norm = x / (n - 1)
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)

    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]
    m = dy / (dx + 1e-8)
    c = y_norm[0] - m * x_norm[0]

    distances = np.abs(y_norm - (m * x_norm + c)) / np.sqrt(m**2 + 1)
    knee_idx = int(np.argmax(distances))
    return float(y[knee_idx]), knee_idx


def center_representation(representation: np.ndarray, representation_mass: float, charge_mean: float) -> np.ndarray:
    """Center an isotopic distribution representation around its maximum."""
    center = int(np.argmax(representation[40:60])) - 50 + 40
    centered = representation.copy()
    med = float(np.median(centered))

    centered[:20] = med
    centered[-20:] = med

    if center < 0:
        centered = np.concatenate([np.full(-center, med), centered[:center] if center != 0 else centered])
    else:
        centered = np.concatenate([centered[center:], np.full(center, med)])

    centered = centered - med
    centered[-1] = representation_mass / 1000 * charge_mean
    return centered


# ---------------------------------------------------------------------- #
# Step 2: deisotoping
# ---------------------------------------------------------------------- #
def deisotope(store, session_id: str, params: DeisotopeParams, deisotoper: MlDeisotoper) -> Dict[str, Any]:
    session = store.get(session_id)
    if not session.is_loaded():
        raise PipelineError("spectrum not loaded", 409)
    key = params_hash(params)

    with session.lock:
        cached = session.deisotope_cache.get(key)
        if cached is not None:
            out = dict(cached)
            out["reused"] = True
            return out

        t0 = time.time()
        try:
            labels, charge_states = deisotoper(
                spectrum=session.spectrum,
                algorithm=params.algorithm,
                z_max=params.z_max,
                min_distance=params.min_distance,
                threshold=params.threshold,
                delta=params.delta,
                n1=params.n1,
                n2=params.n2,
            )
        except Exception as exc:  # noqa: BLE001 - core peak-finding chokes on some spectra
            session.log("warning", f"deisotoping failed: {exc}")
            raise PipelineError(f"deisotoping failed: {exc}", 400)
        elapsed = time.time() - t0

        n_ions = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
        masses = session.masses
        ints = session.ints
        ions: List[Dict[str, Any]] = []
        ion_charge = np.zeros(n_ions, dtype=np.float64)
        for i in range(n_ions):
            idx = np.where(labels == i)[0]
            base = idx[int(np.argmax(ints[idx]))]
            ion_charge[i] = float(np.mean(charge_states[idx]))
            ions.append({
                "ion_id": i,
                "mz": float(masses[base]),
                "charge": float(ion_charge[i]),
                "n_peaks": int(idx.size),
                "mz_min": float(masses[idx[0]]),
                "mz_max": float(masses[idx[-1]]),
            })

        result = {
            "params_hash": key,
            "n_ions": n_ions,
            "elapsed_s": round(elapsed, 3),
            "ions": ions,
        }
        session.deisotope_cache[key] = result
        session.ion_id = labels.astype(np.int32)
        session.ion_info = ions
        session.ion_charge = ion_charge
        session.last_deisotope_hash = key
        session.log("info", f"deisotoping: {n_ions} ions in {elapsed:.2f}s "
                            f"(algorithm={params.algorithm}, z_max={params.z_max})")
        out = dict(result)
        out["reused"] = False
        return out


# ---------------------------------------------------------------------- #
# Step 3: element classification
# ---------------------------------------------------------------------- #
def classify_elements(store, session_id: str, params: ElementsParams, transformer) -> Dict[str, Any]:
    session = store.get(session_id)
    if session.ion_id is None:
        raise PipelineError("run deisotoping (step 2) first", 400)
    atomic = element_atomic_number(params.element)
    key = params_hash(params)

    with session.lock:
        cached = session.elements_cache.get(key)
        if cached is not None:
            out = dict(cached)
            out["reused"] = True
            return out

        # Full classification is element-independent: reuse it if the current
        # ion_probs was computed for the current deisotoping result.
        if session.ion_probs is None or session.ion_probs_for_deiso != session.last_deisotope_hash:
            t0 = time.time()
            session.ion_probs, skipped = _classify_all_ions(session, transformer)
            session.ion_probs_for_deiso = session.last_deisotope_hash
            session.log("info", f"element classification: {int(np.isfinite(session.ion_probs).sum())} ions "
                                f"in {time.time() - t0:.2f}s"
                                + (f", {len(skipped)} skipped" if skipped else ""))
        else:
            skipped = []

        prob_col = session.ion_probs[:, atomic - 1]
        rows = []
        for ion in session.ion_info:
            i = ion["ion_id"]
            p = float(prob_col[i]) if np.isfinite(prob_col[i]) else None
            rows.append({
                "ion_id": i,
                "mz": ion["mz"],
                "charge": ion["charge"],
                "n_peaks": ion["n_peaks"],
                "prob": p,
            })
        rows.sort(key=lambda r: (r["prob"] is None, -(r["prob"] or 0.0)))

        result = {
            "params_hash": key,
            "element": params.element,
            "rows": rows,
            "skipped": skipped,
        }
        session.elements_cache[key] = result
        session.elements_element = params.element
        out = dict(result)
        out["reused"] = False
        return out


def _classify_all_ions(session: Session, transformer) -> Tuple[np.ndarray, List[int]]:
    """Per-ion transformer forward pass; every ion is guarded by try/except."""
    n_ions = len(session.ion_info)
    ion_probs = np.full((n_ions, N_ELEMENTS), np.nan, dtype=np.float64)
    skipped: List[int] = []
    if n_ions == 0:
        return ion_probs, skipped

    spectrum = session.spectrum
    seqs: List[Tuple[int, np.ndarray]] = []
    for i, ion in enumerate(session.ion_info):
        idx = np.where(session.ion_id == i)[0]
        if idx.size == 0:
            skipped.append(i)
            session.log("warning", f"ion {i}: no peaks, skipped")
            continue
        try:
            ri = RealIsotopicDistribution(spectrum, idx.tolist())
            representations, representation_masses = zip(*ri.get_representation(
                f=np.mean, mode="middle", length=101
            ))
            charge_mean = float(session.ion_charge[i])
            centered = [
                center_representation(repr_, mass, charge_mean)
                for repr_, mass in zip(representations, representation_masses)
            ]
            if not centered:
                skipped.append(i)
                session.log("warning", f"ion {i}: empty representation, skipped")
                continue
            max_val = max(item.max() for item in centered)
            if not np.isfinite(max_val) or max_val <= 0:
                skipped.append(i)
                session.log("warning", f"ion {i}: degenerate representation, skipped")
                continue
            seq = np.stack([item / max_val for item in centered]).astype(np.float32)
            seqs.append((i, seq))
        except Exception as exc:  # noqa: BLE001 - mandatory per-ion guard (empty peak_indices bug)
            logger.warning("ion %s skipped during element classification: %s", i, exc)
            session.log("warning", f"ion {i} skipped (element classification): {exc}")
            skipped.append(i)

    if not seqs:
        return ion_probs, skipped

    max_len = max(seq.shape[0] for _, seq in seqs)
    batch = np.zeros((len(seqs), max_len, 100), dtype=np.float32)
    padding_mask = np.zeros((len(seqs), max_len), dtype=bool)
    for row, (_, seq) in enumerate(seqs):
        length = seq.shape[0]
        batch[row, :length] = seq
        padding_mask[row, length:] = True

    src = torch.from_numpy(batch)
    mask = torch.from_numpy(padding_mask)
    try:
        with torch.no_grad():
            logits = transformer(src, src_key_padding_mask=mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        for row, (i, _) in enumerate(seqs):
            ion_probs[i] = probs[row]
    except Exception:
        # fallback: per-ion loop (also guards against batch shape surprises)
        logger.exception("batched transformer forward failed; falling back to per-ion loop")
        for i, seq in seqs:
            try:
                src_i = torch.from_numpy(seq[None, ...])
                mask_i = torch.zeros(1, seq.shape[0], dtype=torch.bool)
                with torch.no_grad():
                    p = torch.sigmoid(transformer(src_i, src_key_padding_mask=mask_i)).cpu().numpy()[0]
                ion_probs[i] = p
            except Exception as exc:  # noqa: BLE001
                logger.warning("ion %s failed in per-ion transformer pass: %s", i, exc)
                session.log("warning", f"ion {i} skipped (transformer forward): {exc}")
                skipped.append(i)
    return ion_probs, skipped


# ---------------------------------------------------------------------- #
# Step 4: knee detection
# ---------------------------------------------------------------------- #
def knee(store, session_id: str, element_symbol: str) -> Dict[str, Any]:
    session = store.get(session_id)
    if session.ion_probs is None:
        raise PipelineError("run element classification (step 3) first", 400)
    atomic = element_atomic_number(element_symbol)

    with session.lock:
        cached = session.knee_cache.get(element_symbol)
        if cached is not None and cached.get("for_deiso") == session.last_deisotope_hash:
            return dict(cached)

        col = session.ion_probs[:, atomic - 1]
        values = col[np.isfinite(col)]
        if values.size == 0:
            raise PipelineError("no classified ions available for knee detection", 400)
        threshold, knee_idx = find_knee_threshold_numpy(values)
        result = {
            "element": element_symbol,
            "probs": [float(v) for v in values],
            "threshold": float(threshold),
            "knee_idx": int(knee_idx),
            "n_ions": int(values.size),
            "for_deiso": session.last_deisotope_hash,
        }
        session.knee_cache[element_symbol] = result
        return dict(result)


# ---------------------------------------------------------------------- #
# Step 5: threshold + per-point probabilities (highlight layer)
# ---------------------------------------------------------------------- #
def apply_threshold(store, session_id: str, params: ThresholdParams) -> Dict[str, Any]:
    session = store.get(session_id)
    if session.ion_probs is None or session.ion_id is None:
        raise PipelineError("run deisotoping (step 2) and elements (step 3) first", 400)
    atomic = element_atomic_number(params.element)

    if params.source == "manual":
        if params.manual_value is None:
            raise PipelineError("manual_value is required when source='manual'", 400)
        if not (0.0 < params.manual_value < 1.0):
            raise PipelineError("manual_value must be in (0, 1)", 400)
        threshold = float(params.manual_value)
    elif params.source == "auto":
        threshold = knee(store, session_id, params.element)["threshold"]
    else:
        raise PipelineError("source must be 'auto' or 'manual'", 400)

    point_probs = np.zeros(session.masses.size, dtype=np.float32)
    is_ion = session.ion_id >= 0
    if is_ion.any():
        col = session.ion_probs[:, atomic - 1]
        point_probs[is_ion] = col[session.ion_id[is_ion]].astype(np.float32)

    with session.lock:
        session.point_probs = point_probs
        session.point_probs_element = params.element
        session.threshold_state[params.element] = {
            "element": params.element,
            "source": params.source,
            "threshold": threshold,
            "set_at": time.time(),
        }

    above = np.where(is_ion & (point_probs > threshold))[0]
    mz_range = None
    if above.size:
        mz_range = [float(session.masses[above.min()]), float(session.masses[above.max()])]
    session.log("info", f"highlight: threshold={threshold:.3g} ({params.source}, {params.element}), "
                        f"{above.size} points above")
    return {
        "element": params.element,
        "source": params.source,
        "threshold": threshold,
        "n_points_above": int(above.size),
        "n_ions_above": int(np.unique(session.ion_id[above]).size) if above.size else 0,
        "mz_range": mz_range,
    }


# ---------------------------------------------------------------------- #
# Step 6: molecular formula determination
# ---------------------------------------------------------------------- #
def _formula_str(elements: List[str], counts) -> str:
    return "".join(f"{el}{int(c) if int(c) > 1 else ''}" for el, c in zip(elements, counts) if int(c) > 0)


def formulas(store, session_id: str, params: FormulaParams) -> Dict[str, Any]:
    session = store.get(session_id)
    if session.ion_id is None or not session.ion_info:
        raise PipelineError("run deisotoping (step 2) first", 400)
    if not (0 <= params.ion_id < len(session.ion_info)):
        raise PipelineError(f"ion_id {params.ion_id} out of range (0..{len(session.ion_info) - 1})", 400)

    elements = list(params.elements.keys())
    if not elements:
        raise PipelineError("elements table is empty", 400)
    low_limits: List[int] = []
    high_limits: List[int] = []
    for el in elements:
        if not isinstance(el, str) or len(el) > 3:
            raise PipelineError(f"invalid element symbol {el!r}", 400)
        element_atomic_number(el)  # raises on unknown symbol
        lo, hi = params.elements[el]
        lo, hi = int(lo), int(hi)
        if lo < 0 or hi < lo:
            raise PipelineError(f"invalid limits for {el}: min={lo}, max={hi}", 400)
        low_limits.append(lo)
        high_limits.append(hi)

    key = params_hash(params)
    with session.lock:
        cached = session.formulas_cache.get(key)
        if cached is not None:
            out = dict(cached)
            out["reused"] = True
            return out

    ion = session.ion_info[params.ion_id]
    target_mz = ion["mz"]
    target_charge = ion["charge"]
    target_mass = (target_mz + ELECTRON_MASS) * target_charge

    t0 = time.time()
    try:
        candidates, skipped_pct = formula_generator_parallel(
            ELEMENTS=elements,
            low_limits=low_limits,
            high_limits=high_limits,
            target_mass=target_mass,
            threshold=params.mass_threshold_ppm,
            num_workers=params.num_workers,
            max_chunk_size=params.max_chunk_size,
            mode="max",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("formula generation failed")
        raise PipelineError(f"formula generation failed: {exc}", 500)
    if not candidates:
        raise PipelineError(
            f"no candidate formulas within +/-{params.mass_threshold_ppm} ppm of "
            f"m {target_mass:.4f}; increase the mass threshold or check the element limits",
            400,
        )

    spectrum = session.spectrum

    def _validate(candidate):
        mass, counts = candidate
        try:
            formula_obj = Formula(_formula_str(elements, counts), charge=target_charge)
            cosine_dist, _, _, delta = check_presence(spectrum, formula_obj)
            return {
                "formula": _formula_str(elements, counts),
                "mass": float(mass),
                "delta_ppm": float(delta),
                "cosine": float(1.0 - cosine_dist),
            }
        except Exception:  # noqa: BLE001 - check_presence can fail on odd formulas
            return None

    validated = Parallel(n_jobs=-1, backend="loky", verbose=0, batch_size="auto")(
        delayed(_validate)(candidate) for candidate in candidates
    )
    ranked = [row for row in validated if row is not None]
    n_failed = len(validated) - len(ranked)
    ranked.sort(key=lambda row: (-row["cosine"], row["delta_ppm"]))
    ranked = [{"rank": i + 1, **row} for i, row in enumerate(ranked)]

    result = {
        "params_hash": key,
        "ion_id": params.ion_id,
        "target_mz": target_mz,
        "target_charge": target_charge,
        "target_mass": target_mass,
        "n_candidates": len(candidates),
        "n_valid": len(ranked),
        "n_failed": n_failed,
        "skipped_pct": float(skipped_pct),
        "elapsed_s": round(time.time() - t0, 3),
        "ranked": ranked,
    }
    with session.lock:
        session.formulas_cache[key] = result
    session.log(
        "info",
        f"formulas ion {params.ion_id}: {len(ranked)}/{len(candidates)} validated "
        f"in {result['elapsed_s']:.2f}s (skipped {skipped_pct:.1f}% of search space)"
        + (f", {n_failed} check failures" if n_failed else ""),
    )
    out = dict(result)
    out["reused"] = False
    return out


# ---------------------------------------------------------------------- #
# Step 7: compare chosen formula with the spectrum (Plotly figure)
# ---------------------------------------------------------------------- #
def compare(store, session_id: str, ion_id: int, formula_str: str, max_pts: int = DEFAULT_MAX_PTS) -> Dict[str, Any]:
    session = store.get(session_id)
    if session.ion_id is None or not session.ion_info:
        raise PipelineError("run deisotoping (step 2) first", 400)
    if not (0 <= ion_id < len(session.ion_info)):
        raise PipelineError(f"ion_id {ion_id} out of range", 400)
    if not formula_str or not formula_str.strip():
        raise PipelineError("formula is empty", 400)

    ion = session.ion_info[ion_id]
    target_charge = ion["charge"]
    try:
        formula = Formula(formula_str.strip(), charge=target_charge)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"invalid formula {formula_str!r}: {exc}", 400)

    try:
        theo_masses, theo_ints = formula.isodistribution()
        theo_masses, theo_ints = del_isotopologues(theo_masses, theo_ints)
        cosine_dist, real_coords, matched_pct, mass_delta = check_presence(session.spectrum, formula)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"formula comparison failed: {exc}", 500)
    real_masses, real_ints = real_coords

    x_left = int(round(float(theo_masses[0]) - 1))
    x_right = int(round(float(theo_masses[-1]) + 1))
    window = downsample_window(session.masses, session.ints, x_left, x_right, max_pts)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"Experimental  (ion {ion_id}, m/z {ion['mz']:.4f}, z={target_charge:g})",
                        f"Calculated  {formula.str_formula}"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(x=window["masses"].tolist(), y=window["ints"].tolist(),
                   mode="lines", name="Experimental", line=dict(color="#222222", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=real_masses.tolist(), y=real_ints.tolist(), mode="markers",
                   name="Matched peaks", marker=dict(color="orange", size=7, line=dict(color="black", width=0.5))),
        row=1, col=1,
    )
    # vertical lines (plotly 5.x has no go.Vline): open line-ns markers spanning the row
    fig.add_trace(
        go.Scatter(x=theo_masses.tolist(), y=[1.0] * len(theo_masses), mode="markers",
                   marker=dict(symbol="line-ns-open", size=16, color="#0055aa",
                               line=dict(width=2)),
                   name="Isotopic pattern", hoverinfo="skip", showlegend=True),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=theo_masses.tolist(), y=theo_ints.tolist(), mode="lines",
                   name="rel. intensity", line=dict(color="#0055aa", width=1, dash="dot"),
                   hoverinfo="skip"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=theo_masses.tolist(), y=theo_ints.tolist(), mode="markers",
                   name="Isotope m/z", marker=dict(color="#0055aa", size=6),
                   text=[f"{m:.4f}" for m in theo_masses.tolist()],
                   hovertemplate="<b>m/z %{x:.4f}</b><br>rel. intensity %{y:.3f}<extra></extra>"),
        row=2, col=1,
    )

    y_max = float(np.max(window["ints"])) if len(window["ints"]) else 1.0
    annotations = [
        dict(text=f"<b>&Delta; = {mass_delta:.2f} ppm</b>", x=x_right, y=0.5 * y_max,
             xref="x1", yref="y1", showarrow=False, xanchor="right"),
        dict(text=f"<b>Cos. dist. = {cosine_dist:.1e}</b>", x=x_right, y=0.62 * y_max,
             xref="x1", yref="y1", showarrow=False, xanchor="right"),
        dict(text=f"<b>Matched peaks = {matched_pct * 100:.1f} %</b>", x=x_right, y=0.74 * y_max,
             xref="x1", yref="y1", showarrow=False, xanchor="right"),
    ]
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(range=[x_left, x_right], row=1, col=1)
    fig.update_xaxes(range=[x_left, x_right], row=2, col=1)
    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_yaxes(title_text="Relative intensity", range=[0, 1.1], row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=680,
        title=dict(text=f"{formula.str_formula} vs spectrum  (charge {target_charge:g})", x=0.05),
        showlegend=True,
        legend=dict(orientation="h", y=1.12, x=0.4),
        margin=dict(l=60, r=30, t=80, b=40),
    )
    return fig.to_plotly_json()
