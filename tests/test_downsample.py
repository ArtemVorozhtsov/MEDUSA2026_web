"""Properties of peak-preserving windowed decimation (task spec section 7)."""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.downsample import COMPARE_HARD_MAX_PTS, HARD_MAX_PTS, clamp_max_pts, downsample_window, uniform_window


def _check_properties(out, window_masses, window_ints, max_pts, full_masses):
    m, i = out["masses"], out["ints"]
    assert len(m) <= max_pts, f"{len(m)} > {max_pts}"
    if len(m) > 1:
        assert np.all(np.diff(np.asarray(m)) > 0), "m/z not strictly increasing"
    # global maximum of the window is always present
    global_max_pos = int(np.argmax(window_ints))
    assert m[np.argmax(i)] == window_masses[global_max_pos] or \
        max(i) == max(window_ints), "global window maximum lost"
    # endpoints within one bin width
    if len(m) and window_masses.size > max_pts:
        bin_width = (window_masses[-1] - window_masses[0]) / max(1, (max_pts - max(1, max_pts // 2)) // 2)
        assert m[0] - window_masses[0] <= bin_width + 1e-9
        assert window_masses[-1] - m[-1] <= bin_width + 1e-9
    # within the original range
    assert min(m) >= min(full_masses) - 1e-9 and max(m) <= max(full_masses) + 1e-9


def test_exact_when_window_fits():
    masses = np.linspace(100, 200, 1200)
    ints = np.random.default_rng(0).integers(1, 100, size=1200).astype(float)
    out = downsample_window(masses, ints, 100, 200, max_pts=2500)
    assert out["decimated"] is False
    assert len(out["masses"]) == 1200
    assert out["masses"].tolist() == masses.tolist()
    assert out["n_in_window"] == 1200


def test_synthetic_full_range():
    rng = np.random.default_rng(42)
    masses = np.sort(rng.uniform(150, 3000, 100_000))
    ints = rng.lognormal(mean=4, sigma=2, size=100_000)
    # inject sharp peaks
    peak_idx = rng.choice(100_000, size=500, replace=False)
    ints[peak_idx] *= 100
    max_pts = 2500
    out = downsample_window(masses, ints, 150, 3000, max_pts=max_pts)
    assert out["decimated"] is True
    assert out["n_in_window"] == 100_000
    _check_properties(out, masses, ints, max_pts, masses)
    assert out["full_range"] == [float(masses[0]), float(masses[-1])]


@pytest.mark.parametrize("max_pts", [2, 3, 4, 5, 10, 50, 2500, 5000])
def test_result_bounded_for_various_max_pts(max_pts):
    rng = np.random.default_rng(7)
    masses = np.linspace(500, 510, 20_000)
    ints = rng.lognormal(mean=2, sigma=1, size=20_000)
    out = downsample_window(masses, ints, 500, 510, max_pts=max_pts)
    assert out["decimated"] is True
    assert len(out["masses"]) <= max_pts
    assert np.all(np.diff(out["masses"]) > 0)
    assert max(out["ints"]) == max(ints)


def test_real_window_data():
    from conftest import load_test_sample

    masses, ints = load_test_sample(0)  # 100k points, 674-687 Da
    # full range
    out = downsample_window(masses, ints, masses[0], masses[-1], max_pts=2500)
    assert out["decimated"] is True
    _check_properties(out, masses, ints, 2500, masses)
    # a small sub-window (isotopic cluster region)
    sub = (masses >= 679.0) & (masses <= 681.0)
    wm, wi = masses[sub], ints[sub]
    out2 = downsample_window(masses, ints, 679.0, 681.0, max_pts=2500)
    assert out2["n_in_window"] == int(sub.sum())
    if wm.size > 2500:
        _check_properties(out2, wm, wi, 2500, masses)
    else:
        assert out2["decimated"] is False
        assert len(out2["masses"]) == wm.size


def test_aux_arrays_stay_aligned():
    masses = np.linspace(10, 20, 30_000)
    ints = np.random.default_rng(1).integers(1, 100, 30_000).astype(float)
    out = downsample_window(masses, ints, 10, 20, max_pts=2000, aux={"ion_id": np.arange(30_000, dtype=np.int32)})
    assert len(out["ion_id"]) == len(out["masses"])
    # the kept points must correspond to the same positions in the full array
    kept_positions = np.searchsorted(masses, out["masses"])
    np.testing.assert_array_equal(out["ion_id"], np.arange(30_000, dtype=np.int32)[kept_positions])


def test_empty_window():
    masses = np.linspace(100, 200, 1000)
    ints = np.ones(1000)
    out = downsample_window(masses, ints, 250, 300, max_pts=1000)
    assert out["masses"] == [] and out["ints"] == []
    assert out["n_in_window"] == 0


def test_clamp_max_pts():
    assert clamp_max_pts(5001) == HARD_MAX_PTS
    assert clamp_max_pts(0) == 1
    assert clamp_max_pts(2500) == 2500
    assert clamp_max_pts(40_000, hard_cap=COMPARE_HARD_MAX_PTS) == COMPARE_HARD_MAX_PTS
    assert clamp_max_pts(40_001, hard_cap=COMPARE_HARD_MAX_PTS) == COMPARE_HARD_MAX_PTS


def test_invalid_range():
    masses = np.linspace(100, 200, 100)
    ints = np.ones(100)
    with pytest.raises(ValueError):
        downsample_window(masses, ints, 150, 150, max_pts=10)


# ---------------------------------------------------------------------- #
# uniform_window (shape-preserving, used by the compare figure)
# ---------------------------------------------------------------------- #
def _iso_like(masses):
    """A few gaussian 'isotope' peaks on a fine grid."""
    rng = np.random.default_rng(11)
    ints = np.full(masses.size, 1e3)
    for center, h, rel in ((405.0, 0.08, 1.0), (406.1, 0.09, 0.55), (407.2, 0.1, 0.21), (408.3, 0.1, 0.06)):
        ints += rel * h * 1e6 * np.exp(-0.5 * ((masses - center) / 0.03) ** 2)
    return ints + rng.normal(0, 2e3, masses.size).clip(min=0)


def test_uniform_window_exact_when_fits():
    masses = np.linspace(100, 200, 1200)
    ints = _iso_like(masses)
    out = uniform_window(masses, ints, 100, 200, max_pts=2500)
    assert out["decimated"] is False
    assert len(out["masses"]) == 1200
    assert out["masses"].tolist() == masses.tolist()


@pytest.mark.parametrize("max_pts", [2500, 500, 100])
def test_uniform_window_shape_preserving(max_pts):
    masses = np.linspace(400, 412, 30_000)
    ints = _iso_like(masses)
    out = uniform_window(masses, ints, 400, 412, max_pts=max_pts)
    m, i = out["masses"], out["ints"]
    assert out["decimated"] is True
    assert len(m) <= max_pts
    assert np.all(np.diff(m) > 0), "not strictly increasing"
    # window maximum is always kept (relative intensities anchored)
    assert max(i) == ints.max()
    assert m[np.argmax(i)] == masses[np.argmax(ints)]
    # returned points are exact original points (no interpolation)
    for v in m[:20]:
        assert v in set(masses.tolist())
    # no missing chunks: the argmax-insertion may create at most two locally
    # enlarged gaps (~2x base); everything else sits on the uniform stride
    gaps = np.diff(m)
    base = np.median(gaps)
    assert gaps.max() <= base * 2.05 + 1e-9, "a chunk of the window was dropped"
    assert (gaps > base * 1.05).sum() <= 2
    assert (np.abs(gaps - base) <= 0.05 * base).mean() >= 0.9


def test_uniform_window_extra_indices_merged():
    masses = np.linspace(400, 412, 30_000)
    ints = _iso_like(masses)
    # apexes of the four synthetic isotope peaks (none lies on the stride-31 grid)
    apexes = np.array(
        [int(np.argmax(np.exp(-0.5 * ((masses - c) / 0.03) ** 2))) for c in (405.0, 406.1, 407.2, 408.3)]
    )
    assert all(int(a) % 31 != 0 for a in apexes)
    base = uniform_window(masses, ints, 400, 412, max_pts=997, hard_cap=COMPARE_HARD_MAX_PTS)
    out = uniform_window(masses, ints, 400, 412, max_pts=997, hard_cap=COMPARE_HARD_MAX_PTS, extra=apexes)
    expected = set(base["masses"].tolist()) | {float(masses[a]) for a in apexes}
    assert set(out["masses"].tolist()) == expected
    assert np.all(np.diff(out["masses"]) > 0)


def test_uniform_window_compare_resolution():
    # compare policy: a window bigger than the interactive viewer cap (5000)
    # but within COMPARE_HARD_MAX_PTS comes back exact (undecimated)
    masses = np.linspace(400, 412, 20_000)
    ints = _iso_like(masses)
    out = uniform_window(masses, ints, 400, 412, max_pts=20_000, hard_cap=COMPARE_HARD_MAX_PTS)
    assert out["decimated"] is False
    assert len(out["masses"]) == 20_000
    assert out["masses"].tolist() == masses.tolist()
    # above the cap the result stays bounded by the cap
    out2 = uniform_window(masses, ints, 400, 412, max_pts=20_000, hard_cap=5_000)
    assert out2["decimated"] is True
    assert len(out2["masses"]) <= 5_000
    assert np.all(np.diff(out2["masses"]) > 0)
