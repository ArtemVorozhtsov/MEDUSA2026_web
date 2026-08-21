"""Pipeline smoke test on a small real spectrum (first sample of test.h5).

Deisotope -> element classification -> knee -> formulas for the best ion;
asserts the ranked list is non-empty and the ground-truth formula (test.pkl)
is present. Also checks parameter caching and the empty-peak_indices guard.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.pipeline import (
    DeisotopeParams,
    ElementsParams,
    FormulaParams,
    _classify_all_ions,
    classify_elements,
    deisotope,
    compare,
    formulas,
    knee,
)
from app.state import SessionStore
from mass_automation.deisotoping.process import MlDeisotoper
from mass_automation.experiment import Spectrum
from mass_automation.formula import Formula


PUBCHEM10 = {
    "C": (0, 67), "H": (0, 97), "N": (0, 10), "O": (0, 17), "Cl": (0, 3),
    "I": (0, 1), "Br": (0, 2), "F": (0, 6), "S": (0, 3), "P": (0, 1),
}


@pytest.fixture(scope="module")
def models():
    import os
    deisotoper = MlDeisotoper().load(os.environ["CGB_MODEL"])
    from mass_automation.formula.Transformer import TransformerModel
    model = TransformerModel.load_from_checkpoint(os.environ["TRANSFORMER_CKPT"], map_location="cpu")
    model.eval()
    return deisotoper, model


@pytest.fixture(scope="module")
def session(small_spectrum):
    from conftest import load_test_sample

    masses, ints = load_test_sample(0)
    store = SessionStore(max_active_sessions=4)
    sess = store.create("test_sample0", "test", "test_sample0")
    sess.mark_loaded({
        "masses": masses,
        "ints": ints,
        "spectrum": Spectrum(masses, ints),
        "n_points": int(masses.size),
        "mz_min": float(masses.min()),
        "mz_max": float(masses.max()),
        "n_scans": 1,
        "load_time_s": 0.0,
    })
    return store, sess


@pytest.fixture(scope="module")
def full_run(models, session):
    store, sess = session
    deisotoper, transformer = models

    r_deiso = deisotope(store, sess.id, DeisotopeParams(), deisotoper)
    r_elements = classify_elements(store, sess.id, ElementsParams(element="Ir"), transformer)
    r_knee = knee(store, sess.id, "Ir")
    best_ion = max(r_elements["rows"], key=lambda r: r["prob"] or 0.0)["ion_id"]
    r_formulas = formulas(
        store, sess.id,
        FormulaParams(ion_id=best_ion, elements=dict(PUBCHEM10), mass_threshold_ppm=4.0,
                     num_workers=1, max_chunk_size=10_000),
    )
    return {
        "store": store, "sess": sess,
        "deiso": r_deiso, "elements": r_elements, "knee": r_knee,
        "formulas": r_formulas, "best_ion": best_ion,
    }


def test_deisotope(full_run, small_spectrum):
    r = full_run["deiso"]
    assert r["n_ions"] >= 1
    assert "reused" not in r  # step 2 is uncached by design
    sess = full_run["sess"]
    assert sess.ion_id.shape == (small_spectrum[0].size,)
    assert sess.ion_id.min() >= -1
    assert len(r["ions"]) == r["n_ions"]
    for ion in r["ions"]:
        assert ion["n_peaks"] >= 2
        assert ion["charge"] >= 1


def test_elements(full_run):
    r = full_run["elements"]
    assert r["element"] == "Ir"
    assert len(r["rows"]) == full_run["deiso"]["n_ions"]
    for row in r["rows"]:
        if row["prob"] is not None:
            assert 0.0 <= row["prob"] <= 1.0


def test_knee(full_run):
    r = full_run["knee"]
    assert r["element"] == "Ir"
    assert r["n_ions"] >= 1
    assert 0.0 < r["threshold"] <= 1.0
    assert 0 <= r["knee_idx"] < r["n_ions"]
    assert r["probs"] == sorted(r["probs"], reverse=True)


def test_formulas_ranked_and_gt(full_run, gt_formulas):
    r = full_run["formulas"]
    assert r["n_candidates"] > 0
    assert r["n_valid"] > 0, "ranked list must be non-empty"
    ranked = r["ranked"]
    assert [row["rank"] for row in ranked] == list(range(1, len(ranked) + 1))
    # sorted by cosine desc, then delta asc
    for a, b in zip(ranked, ranked[1:]):
        assert (a["cosine"] > b["cosine"]) or \
               (abs(a["cosine"] - b["cosine"]) < 1e-12 and a["delta_ppm"] <= b["delta_ppm"])
    # ground truth formula for sample 0 must be present
    gt_formula = gt_formulas[0]["formula"]
    gt_dict = {k: int(v) for k, v in Formula(gt_formula).dict_formula.items() if int(v) > 0}
    found = [
        row for row in ranked
        if {k: int(v) for k, v in Formula(row["formula"]).dict_formula.items() if int(v) > 0} == gt_dict
    ]
    assert found, f"GT formula {gt_formula} not in the ranked list"
    assert found[0]["delta_ppm"] < 4.0


def test_cache_reuse_is_instant(full_run, models):
    deisotoper, _ = models
    store, sess = full_run["store"], full_run["sess"]
    # step 2 is uncached: a re-run recomputes, but the ion set is unchanged
    again = deisotope(store, sess.id, DeisotopeParams(), deisotoper)
    assert again["n_ions"] == full_run["deiso"]["n_ions"]
    again_el = classify_elements(store, sess.id, ElementsParams(element="Ir"), None)
    assert again_el["reused"] is True


def test_empty_peak_indices_guard(models):
    """The known latent core bug (empty peak_indices ->
    ``ValueError: max() arg is an empty sequence``) must not crash the
    session: the engine skips the ion and logs a visible warning."""
    import app.engine.pipeline as pipe
    from mass_automation.experiment import Spectrum

    deisotoper, transformer = models

    # Synthetic two-ion spectrum: two small isotopic clusters.
    masses = []
    ints = []
    for cluster_mz, charge in ((500.0, 2.0), (510.0, 2.0)):
        for offset, rel in ((0.0, 1.0), (0.5, 0.5), (1.0, 0.2), (1.5, 0.05)):
            masses.append(cluster_mz + offset)
            ints.append(rel * 1e6)
    masses = np.array(masses)
    ints = np.array(ints)

    store = SessionStore(max_active_sessions=2)
    sess = store.create("guard", "test", "guard")
    sess.mark_loaded({
        "masses": masses, "ints": ints,
        "spectrum": Spectrum(masses, ints),
        "n_points": int(masses.size),
        "mz_min": float(masses.min()),
        "mz_max": float(masses.max()),
        "n_scans": 1, "load_time_s": 0.0,
    })
    # emulate deisotoping output: two ions, each with 4 peaks
    sess.ion_id = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    sess.ion_info = [
        {"ion_id": 0, "mz": 500.0, "charge": 2.0, "n_peaks": 4, "mz_min": 500.0, "mz_max": 501.5},
        {"ion_id": 1, "mz": 510.0, "charge": 2.0, "n_peaks": 4, "mz_min": 510.0, "mz_max": 511.5},
    ]
    sess.ion_charge = np.array([2.0, 2.0])
    sess.last_deisotope_hash = "test-hash"

    original = pipe.RealIsotopicDistribution
    calls = {"n": 0}

    class RaisingRI:
        """Emulates the core bug: raises on the 2nd constructed ion."""

        def __init__(self, spectrum, peak_indices):
            self.spectrum = spectrum
            self.peak_indices = sorted(peak_indices)

        def get_representation(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("max() arg is an empty sequence")
            return original(self.spectrum, self.peak_indices).get_representation(*args, **kwargs)

    pipe.RealIsotopicDistribution = RaisingRI
    try:
        result = classify_elements(store, sess.id, ElementsParams(element="Ir"), transformer)
    finally:
        pipe.RealIsotopicDistribution = original

    assert result["skipped"] == [1], result["skipped"]
    row = next(r for r in result["rows"] if r["ion_id"] == 1)
    assert row["prob"] is None
    row0 = next(r for r in result["rows"] if r["ion_id"] == 0)
    assert row0["prob"] is not None
    warnings = [l for l in sess.recent_logs() if l["level"] == "warning"]
    assert any("ion 1" in l["message"] for l in warnings), "skipped ion must be logged"
    # session still usable: knee works on the remaining ions
    r = knee(store, sess.id, "Ir")
    assert r["n_ions"] == 1


def test_compare_figure(full_run, small_spectrum):
    """Compare figure: experimental window keeps true relative intensities,
    point count is bounded, and the metrics (incl. delta) are annotated."""
    store, sess = full_run["store"], full_run["sess"]
    best_ion = full_run["best_ion"]
    formula_str = full_run["formulas"]["ranked"][0]["formula"]

    fig = compare(store, sess.id, best_ion, formula_str, max_pts=2500)
    data = fig["data"]
    assert len(data) >= 3

    exp = data[0]  # experimental trace, row 1
    matched = data[1]
    # soft bound: uniform grid <= max_pts + merged matched-peak neighborhoods
    assert len(exp["x"]) <= 2500 + len(matched["x"]) * 300
    # every matched peak (full resolution) lies exactly on the experimental line
    exp_set = set(exp["x"])
    for mx in matched["x"]:
        assert mx in exp_set
    masses, ints = small_spectrum
    x0, x1 = min(exp["x"]), max(exp["x"])
    mask = (masses >= x0 - 1e-9) & (masses <= x1 + 1e-9)
    assert max(exp["y"]) == pytest.approx(float(ints[mask].max())), \
        "experimental max intensity must match the true window (no distortion)"
    # the returned max-intensity point is the true window argmax
    assert exp["x"][int(np.argmax(exp["y"]))] == pytest.approx(
        float(masses[mask][np.argmax(ints[mask])]))

    # metrics block with delta (item: delta on the plot)
    annotations = fig["layout"].get("annotations") or []
    joined = " ".join(a.get("text", "") for a in annotations)
    assert "ppm" in joined and "Cos. dist." in joined and "Matched" in joined


def test_knee_probs_sorted_desc(full_run):
    """knee() must return probabilities sorted descending (chart contract);
    knee_idx is the position in the sorted array."""
    r = full_run["knee"]
    assert r["probs"] == sorted(r["probs"], reverse=True)
    assert r["threshold"] == pytest.approx(r["probs"][r["knee_idx"]])
