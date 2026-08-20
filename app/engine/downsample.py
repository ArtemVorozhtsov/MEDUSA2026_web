"""Peak-preserving windowed decimation for very large spectra in the browser.

Guarantees (asserted in tests):

* returned m/z values are strictly increasing;
* the result has at most ``max_pts`` points;
* the global maximum-intensity point of the window is always kept;
* the endpoints of the window are represented within one bin width;
* when the window already fits into ``max_pts`` points, the exact (undecimated)
  points are returned with ``decimated=False``.

The function is pure: it only reads the input arrays and returns new ones.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

HARD_MAX_PTS = 5000
DEFAULT_MAX_PTS = 2500
# Static compare figure: the cluster window is small and server-fixed (not the
# pan/zoom loop), so it may carry more points than the interactive viewer.
COMPARE_HARD_MAX_PTS = 40000


def clamp_max_pts(max_pts: int, hard_cap: int = HARD_MAX_PTS) -> int:
    return int(max(1, min(int(max_pts), hard_cap)))


def window_slice(masses: np.ndarray, x0: float, x1: float):
    """Index range [lo, hi) of points with x0 <= m/z <= x1 (masses sorted)."""
    lo = int(np.searchsorted(masses, x0, side="left"))
    hi = int(np.searchsorted(masses, x1, side="right"))
    return lo, hi


def downsample_window(
    masses: np.ndarray,
    ints: np.ndarray,
    x0: float,
    x1: float,
    max_pts: int = DEFAULT_MAX_PTS,
    aux: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """Return the (possibly decimated) window ``[x0, x1]`` plus parallel aux arrays.

    ``aux`` maps name -> full-length array (aligned with ``masses``); each value is
    sliced to the window and decimated in parallel with the spectrum.
    """
    max_pts = clamp_max_pts(max_pts)
    if x1 <= x0:
        raise ValueError("x1 must be greater than x0")

    lo, hi = window_slice(masses, x0, x1)
    window_masses = masses[lo:hi]
    window_ints = ints[lo:hi]

    result: Dict[str, object] = {
        "n_in_window": int(hi - lo),
        "x0": float(x0),
        "x1": float(x1),
        "full_range": [float(masses[0]), float(masses[-1])],
    }
    if window_masses.size == 0:
        result.update(masses=[], ints=[], decimated=False, **_empty_aux(aux))
        return result

    aux_slices = {name: array[lo:hi] for name, array in (aux or {}).items()}

    if window_masses.size <= max_pts:
        result.update(
            masses=window_masses,
            ints=window_ints,
            decimated=False,
            **{name: array for name, array in aux_slices.items()},
        )
        return result

    # --- peak-preserving decimation ------------------------------------- #
    k = max(1, max_pts // 2)
    n_bins = max(1, (max_pts - k) // 2)

    # (a) top-K points by intensity (always kept; includes the global max)
    top_k = set(np.argpartition(window_ints, -k)[-k:].tolist())

    # (b) contiguous bins over the window mass range; keep max + min of each bin
    bin_edges = np.linspace(window_masses[0], window_masses[-1], n_bins + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, window_masses, side="right") - 1, 0, n_bins - 1)

    bin_max_val = np.full(n_bins, -np.inf)
    np.maximum.at(bin_max_val, bin_idx, window_ints)
    bin_min_val = np.full(n_bins, np.inf)
    np.minimum.at(bin_min_val, bin_idx, window_ints)

    first_max = _first_extremum(window_ints, bin_idx, bin_max_val, is_max=True, n_bins=n_bins)
    first_min = _first_extremum(window_ints, bin_idx, bin_min_val, is_max=False, n_bins=n_bins)

    keep = set(top_k)
    for b in range(n_bins):
        if first_max[b] >= 0:
            keep.add(int(first_max[b]))
        if first_min[b] >= 0:
            keep.add(int(first_min[b]))

    order = np.fromiter(keep, dtype=np.int64)
    order.sort()
    out_masses = window_masses[order]
    out_ints = window_ints[order]

    result.update(
        masses=out_masses,
        ints=out_ints,
        decimated=True,
        **{name: array[order] for name, array in aux_slices.items()},
    )
    return result


def _first_extremum(ints: np.ndarray, bin_idx: np.ndarray, bin_val: np.ndarray, is_max: bool, n_bins: int) -> np.ndarray:
    """Index of the first point in each bin whose intensity equals the bin extremum."""
    match = (ints == bin_val[bin_idx])
    candidates = np.where(match)[0]
    sentinel = np.iinfo(np.int64).max
    out = np.full(n_bins, sentinel, dtype=np.int64)
    if candidates.size:
        np.minimum.at(out, bin_idx[candidates], candidates)
    out[out == sentinel] = -1
    return out


def _empty_aux(aux: Optional[Dict[str, np.ndarray]]) -> Dict[str, list]:
    return {name: [] for name in (aux or {})}


def uniform_window(
    masses: np.ndarray,
    ints: np.ndarray,
    x0: float,
    x1: float,
    max_pts: int = DEFAULT_MAX_PTS,
    hard_cap: int = HARD_MAX_PTS,
    extra: Optional[np.ndarray] = None,
):
    """Shape-preserving window for the compare figure.

    ``extra``: optional window-relative indices that must be included in the
    output (e.g. raw neighborhoods around matched peaks, so the compare line
    passes through every matched peak apex). The result is bounded by
    ``max_pts + len(extra)``; ``extra`` is a no-op when the window is exact.

    Unlike :func:`downsample_window` (peak-preserving, biased to high-intensity
    points), this keeps every N-th point so the *relative* intensities of the
    isotopic cluster match the real spectrum. The window maximum is always
    kept. Returns ``{"masses", "ints", "n_in_window", "decimated"}``.

    ``hard_cap`` bounds ``max_pts``: the interactive viewer uses
    ``HARD_MAX_PTS`` (5000, task spec section 7), the compare figure uses
    ``COMPARE_HARD_MAX_PTS`` (40000) so the window can be sent exact.
    """
    max_pts = clamp_max_pts(max_pts, hard_cap)
    if x1 <= x0:
        raise ValueError("x1 must be greater than x0")

    lo, hi = window_slice(masses, x0, x1)
    window_masses = masses[lo:hi]
    window_ints = ints[lo:hi]

    if window_masses.size <= max_pts:
        return {
            "masses": window_masses, "ints": window_ints,
            "n_in_window": int(window_masses.size), "decimated": False,
        }
    step = int(np.ceil(window_masses.size / max_pts))
    sel = np.arange(0, window_masses.size, step)
    argmax = int(np.argmax(window_ints))
    sel = np.unique(np.concatenate([sel, [argmax]]))
    if extra is not None and extra.size:
        sel = np.unique(np.concatenate([sel,
                                        np.clip(extra, 0, window_masses.size - 1)]))
    elif sel.size > max_pts:
        # keep the bound exact: drop the sampled point nearest to the argmax
        keep = [i for i in sel if i != argmax]
        nearest = min(keep, key=lambda i: abs(int(i) - argmax))
        sel = sel[sel != nearest]
    sel = np.sort(sel)
    return {
        "masses": window_masses[sel], "ints": window_ints[sel],
        "n_in_window": int(window_masses.size), "decimated": True,
    }
