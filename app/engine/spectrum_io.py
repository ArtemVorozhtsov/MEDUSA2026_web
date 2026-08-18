"""Spectrum loading (one scan) and server-folder file listing."""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

import numpy as np

from mass_automation.experiment import Experiment, Spectrum

logger = logging.getLogger("medusa_web.spectrum_io")

SPECTRA_EXTENSIONS = (".mzxml",)
MAX_LIST_DEPTH = 3


def list_spectra_files(root: str) -> List[Dict[str, object]]:
    """List ``*.mzXML`` files under ``root`` (depth <= 3), sorted by path.

    Returns relative POSIX-style paths and sizes in bytes.
    """
    root = os.path.abspath(root)
    found: List[Dict[str, object]] = []
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if os.path.relpath(dirpath, root) == ".":
            depth = 0
        if depth >= MAX_LIST_DEPTH:
            dirnames[:] = []
            continue
        for name in filenames:
            if name.lower().endswith(SPECTRA_EXTENSIONS):
                full = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                found.append({
                    "path": os.path.relpath(full, root).replace(os.sep, "/"),
                    "name": name,
                    "size": size,
                })
    found.sort(key=lambda item: str(item["path"]))
    return found


def load_spectrum(path: str) -> Dict[str, object]:
    """Load the FIRST scan of an mzXML file.

    Returns a dict with float64 ``masses``/``ints`` arrays, a ``Spectrum``
    object (used by the pipeline), and metadata.
    """
    t0 = time.time()
    experiment = Experiment(path, verbose=False)
    if experiment.len == 0:
        raise ValueError(f"no scans found in {path!r}")

    spectrum: Spectrum = experiment[0]
    masses = np.asarray(spectrum.masses, dtype=np.float64)
    ints = np.asarray(spectrum.ints, dtype=np.float64)
    if masses.size == 0:
        raise ValueError(f"scan 0 of {path!r} has no data points")
    if not (np.all(np.isfinite(masses)) and np.all(np.isfinite(ints))):
        raise ValueError(f"scan 0 of {path!r} contains non-finite values")

    data = {
        "masses": masses,
        "ints": ints,
        "spectrum": Spectrum(masses, ints, experiment.n_scans, experiment.n_points, path),
        "n_points": int(masses.size),
        "mz_min": float(masses.min()),
        "mz_max": float(masses.max()),
        "n_scans": int(experiment.len),
        "load_time_s": round(time.time() - t0, 3),
    }
    logger.info(
        "loaded %s: %d points, m/z %.2f..%.2f, %d scan(s), %.2fs",
        path, data["n_points"], data["mz_min"], data["mz_max"], data["n_scans"], data["load_time_s"],
    )
    return data
