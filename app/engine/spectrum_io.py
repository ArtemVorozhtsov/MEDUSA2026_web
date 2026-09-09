"""Spectrum loading (one scan) and server-folder file listing."""
from __future__ import annotations

import gzip
import logging
import os
import re
import shutil
import time
import zipfile
from typing import Dict, List

import numpy as np

from mass_automation.experiment import Experiment, Spectrum

logger = logging.getLogger("medusa_web.spectrum_io")

SPECTRA_EXTENSIONS = (".mzxml",)
MAX_LIST_DEPTH = 3

# The mzXML <polarity> marker always lives near the start of the file
# (instrumentConfiguration or the first scan), so a bounded prefix scan suffices.
POLARITY_SCAN_BYTES = 2 * 1024 * 1024
_POLARITY_RE = re.compile(
    rb"<polarity>\s*(positive|negative|unknown|\+|-|1|-1)\s*</polarity>"
    rb"|polarity\s*=\s*[\x22\x27](positive|negative|unknown|\+|-|1|-1)[\x22\x27]",
    re.IGNORECASE,
)


def _read_file_prefix(path: str, limit: int) -> bytes:
    """First ``limit`` bytes of XML, transparently following gzip (magic bytes)."""
    try:
        with open(path, "rb") as f:
            if f.read(2) == b"\x1f\x8b":
                f.seek(0)
                buf = gzip.GzipFile(fileobj=f)
                out = bytearray()
                while len(out) < limit:
                    chunk = buf.read(1024 * 1024)
                    if not chunk:
                        break
                    out.extend(chunk)
                return bytes(out[:limit])
            f.seek(0)
            return f.read(limit)
    except (OSError, EOFError):
        return b""


def detect_polarity(path: str) -> str:
    """Detect the ion polarity from the first ``<polarity>`` marker in the file.

    Handles the standard child-element form (``<scan><polarity>negative</polarity>``
    or ``<instrumentConfiguration><polarity>…``) and the attribute form
    (``<scan polarity="+" …>``) used by some exporters; gzip-compressed files
    (``.mzXML.gz``) are followed transparently. Returns ``"positive"``,
    ``"negative"`` or ``"unknown"``.
    """
    chunk = _read_file_prefix(path, POLARITY_SCAN_BYTES)
    if not chunk:
        return "unknown"
    m = _POLARITY_RE.search(chunk)
    if m is None:
        return "unknown"
    value = (m.group(1) or m.group(2)).decode("ascii").lower()
    if value in ("positive", "+", "1"):
        return "positive"
    if value in ("negative", "-", "-1"):
        return "negative"
    return "unknown"


def extract_d_zip(zip_path: str, dest_dir: str, max_uncompressed_bytes: int) -> int:
    """Extract an Agilent ``.d`` directory archive (uploaded as zip).

    Guards against zip-slip paths and unbounded expansion. If the archive
    wraps the ``.d`` folder as a single top-level directory (typical when a
    user zips the folder itself), that level is stripped so the raw files
    land directly under ``dest_dir``. Returns the number of files written.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("not a valid zip archive")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        file_parts: List[List[str]] = []
        total = 0
        for info in members:
            parts = info.filename.replace("\\", "/").split("/")
            if info.filename.startswith(("/", "\\")) or ".." in parts:
                raise ValueError(f"unsafe path in archive: {info.filename!r}")
            if not info.is_dir():
                total += info.file_size
                file_parts.append(parts)
        if total > max_uncompressed_bytes:
            raise ValueError(
                f"archive expands to {total} bytes (limit {max_uncompressed_bytes})")
        strip_top = None
        if file_parts and all(len(parts) > 1 for parts in file_parts):
            tops = {parts[0] for parts in file_parts}
            if len(tops) == 1:
                strip_top = tops.pop()
        os.makedirs(dest_dir, exist_ok=True)
        count = 0
        for info in members:
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            rel = parts[1:] if strip_top is not None else parts
            target = os.path.join(dest_dir, *rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src_file, open(target, "wb") as dst:
                shutil.copyfileobj(src_file, dst)
            count += 1
        return count


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
        "polarity": detect_polarity(path),
    }
    logger.info(
        "loaded %s: %d points, m/z %.2f..%.2f, %d scan(s), %.2fs",
        path, data["n_points"], data["mz_min"], data["mz_max"], data["n_scans"], data["load_time_s"],
    )
    return data
