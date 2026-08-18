"""Knee detection on known-shape synthetic probability vectors."""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.pipeline import find_knee_threshold_numpy


def test_elbow_vector_classic():
    # five "significant" ions, then a steep drop
    probs = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.05, 0.02, 0.01, 0.005, 0.001])
    threshold, knee_idx = find_knee_threshold_numpy(probs)
    assert knee_idx == 5
    assert threshold == pytest.approx(0.05)


def test_elbow_vector_second():
    probs = np.array([0.9, 0.85, 0.8, 0.02, 0.01, 0.004, 0.001])
    threshold, knee_idx = find_knee_threshold_numpy(probs)
    assert knee_idx == 3
    assert threshold == pytest.approx(0.02)


def test_exponential_decay_knee_in_upper_region():
    probs = np.array(10 ** (-np.linspace(0, 4, 50)))
    threshold, knee_idx = find_knee_threshold_numpy(probs)
    assert 0 <= knee_idx < 50
    assert threshold == pytest.approx(float(np.sort(probs)[::-1][knee_idx]))
    assert knee_idx < 20  # knee must be in the steep part, not the tail


def test_threshold_matches_sorted_value():
    rng = np.random.default_rng(3)
    for _ in range(25):
        probs = rng.random(rng.integers(3, 40))
        threshold, knee_idx = find_knee_threshold_numpy(probs)
        assert threshold == pytest.approx(float(np.sort(probs)[::-1][knee_idx]))


def test_short_vectors():
    assert find_knee_threshold_numpy(np.array([0.1])) == (0.1, 0)
    thr, idx = find_knee_threshold_numpy(np.array([0.2, 0.1]))
    assert idx == 1 and thr == pytest.approx(0.1)
    thr, idx = find_knee_threshold_numpy(np.array([0.5, 0.4, 0.3]))
    assert idx == 2 and thr == pytest.approx(0.3)


def test_constant_vector():
    thr, idx = find_knee_threshold_numpy(np.full(10, 0.3))
    assert thr == pytest.approx(0.3)
    assert 0 <= idx < 10
