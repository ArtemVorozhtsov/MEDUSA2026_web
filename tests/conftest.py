"""Test bootstrap: sys.path setup, model paths, small-spectrum fixtures.

The tests live in ``medusa_web/tests`` and run from the repo root
(``pytest medusa_web/tests -q``) both in the dev venv and inside the Docker
image (``docker compose run --rm web pytest medusa_web/tests -q``).

Test data: ``tests/data/test.h5`` + ``tests/data/test.pkl`` are a slice (first
8 samples) of ``MEDUSA2026/data/formula_determination_test/`` — synthetic
single-molecule FT-ICR-like spectra (100k points each) with ground-truth
formulas.
"""
from __future__ import annotations

import base64
import gzip
import os
import pickle
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
MEDUSA_WEB_DIR = TESTS_DIR.parent
DATA_DIR = TESTS_DIR / "data"


def _find_mass_automation_root() -> Path:
    candidates = []
    env = os.environ.get("MASS_AUTOMATION_PATH")
    if env:
        candidates.append(Path(env))
    candidates += [
        MEDUSA_WEB_DIR.parent / "MEDUSA2026",  # dev checkout (sibling)
        Path("/app"),                            # docker image
    ]
    for candidate in candidates:
        if (candidate / "mass_automation" / "__init__.py").exists():
            return candidate
    raise RuntimeError("mass_automation not found; set MASS_AUTOMATION_PATH")


MA_ROOT = _find_mass_automation_root()
for _p in (str(MEDUSA_WEB_DIR), str(MA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _default_model(env_name: str, repo_rel: str) -> str:
    if os.environ.get(env_name):
        return os.environ[env_name]
    baked = Path("/app/models") / Path(repo_rel).name
    if baked.exists():
        return str(baked)
    return str(MA_ROOT / repo_rel)


os.environ.setdefault("CGB_MODEL", _default_model("CGB_MODEL", "data/models/charge1_big_optuna150.pkl"))
os.environ.setdefault("TRANSFORMER_CKPT", _default_model("TRANSFORMER_CKPT", "nn_models/transfomer_classifier.ckpt"))
os.environ.setdefault("SPECTRA_DIR", str(DATA_DIR))
os.environ.setdefault("UPLOAD_DIR", str(DATA_DIR))


# ---------------------------------------------------------------------- #
# small spectrum helpers
# ---------------------------------------------------------------------- #
def load_test_sample(index: int = 0):
    """(masses, ints) float64 arrays of test.h5 sample 0..7."""
    with h5py.File(DATA_DIR / "test.h5", "r") as f:
        data = f["Dataset"][index]
    return np.array(data[0], dtype=np.float64), np.array(data[1], dtype=np.float64)


def load_gt_formulas() -> list:
    """Ground-truth dicts (with 'formula' key) from test.pkl (gzip of pickles)."""
    gt = []
    with gzip.open(DATA_DIR / "test.pkl", "rb") as f:
        while True:
            try:
                gt.append(pickle.load(f))
            except EOFError:
                break
    return gt


def write_mzxml(path, masses: np.ndarray, ints: np.ndarray, polarity: str = "+") -> Path:
    """Write a minimal mzXML (sashimi 3.2 namespace) readable by pyopenms 3.2.

    Structure mirrors real instrument exports: zlib-compressed interleaved
    m/z-intensity binary, big-endian doubles, base64-encoded.
    """
    n = len(masses)
    interleaved = np.empty(2 * n, dtype=np.float64)
    interleaved[0::2] = masses
    interleaved[1::2] = ints
    b64 = base64.b64encode(zlib.compress(interleaved.astype(">f8").tobytes(), 1)).decode()
    xml = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        '<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"\n'
        '       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '       xsi:schemaLocation="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2'
        ' http://sashimi.sourceforge.net/schema_revision/mzXML_3.2/mzXML_idx_3.2.xsd">\n'
        f'  <msRun scanCount="1">\n'
        f'    <scan num="1" scanType="Full" centroided="1" msLevel="1" peaksCount="{n}" polarity="{polarity}"\n'
        f'          startMz="{float(masses.min()):.6f}" endMz="{float(masses.max()):.6f}">\n'
        f'      <peaks compressionType="zlib" compressedLen="{len(b64)}" precision="64"'
        f' byteOrder="network" contentType="m/z-int">{b64}</peaks>\n'
        "    </scan>\n"
        "  </msRun>\n"
        "</mzXML>\n"
    )
    if isinstance(path, (str, os.PathLike)):
        target = Path(path)
        target.write_text(xml, encoding="ascii")
        return target
    path.write(xml.encode("ascii"))
    return None


# ---------------------------------------------------------------------- #
# fixtures
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def small_spectrum():
    return load_test_sample(0)


@pytest.fixture(scope="session")
def gt_formulas():
    return load_gt_formulas()


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """FastAPI TestClient with isolated SPECTRA_DIR/UPLOAD_DIR and loaded models."""
    spectra = tmp_path / "spectra"
    uploads = tmp_path / "uploads"
    spectra.mkdir()
    uploads.mkdir()
    monkeypatch.setenv("SPECTRA_DIR", str(spectra))
    monkeypatch.setenv("UPLOAD_DIR", str(uploads))
    monkeypatch.setenv("MAX_ACTIVE_SESSIONS", "4")
    monkeypatch.setenv("MAX_CONCURRENT_CPU_JOBS", "2")

    from app.config import reset_settings_cache
    reset_settings_cache()
    from app import main
    from fastapi.testclient import TestClient

    app = main.create_app()
    with TestClient(app) as client:
        yield client
