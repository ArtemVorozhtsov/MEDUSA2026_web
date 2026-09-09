"""End-to-end .d (Agilent) zip upload flow: extract, (simulated) msconvert, session ready."""
from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

from conftest import write_mzxml


def _d_zip_bytes() -> bytes:
    """A zip mimicking an Agilent .d folder (member layout, fake contents)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Bromhexine.d/analysis.baf", b"fake agilent binary")
        zf.writestr("Bromhexine.d/analysis.baf_idx", b"idx-bytes")
    return buf.getvalue()


def _simulate_converter(upload_dir: str, file_id: str, masses, ints, polarity: str,
                        exit_code: int = 0, stderr_tail: str = "") -> None:
    """Write what the converter sidecar would produce (mzXML.gz output + .done)."""
    mz = Path(upload_dir) / "sim_tmp.mzXML"
    write_mzxml(mz, masses, ints, polarity=polarity)
    raw = mz.read_bytes()
    mz.unlink()
    with gzip.open(Path(upload_dir) / f"{file_id}.d.mzXML.gz", "wb") as f:
        f.write(raw)
    jobs = Path(upload_dir) / "jobs"
    jobs.mkdir(exist_ok=True)
    (jobs / f"{file_id}.d.done").write_text(f"exit={exit_code}\nlog_tail: {stderr_tail}\n")


def test_d_upload_convert_and_ready(api_client, small_spectrum):
    from app.config import get_settings

    masses, ints = small_spectrum
    r = api_client.post("/api/upload", files={
        "file": ("Bromhexine.d.zip", _d_zip_bytes(), "application/zip"),
    })
    assert r.status_code == 200, r.text
    file_id = r.json()["file_id"]
    upload_dir = Path(get_settings().upload_dir)

    r = api_client.post("/api/sessions", json={"source": "upload", "file_id": file_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "converting"
    assert body["n_points"] is None
    sid = body["session_id"]

    # archive extracted, job request waiting for the sidecar
    assert (upload_dir / f"{file_id}.d" / "analysis.baf").is_file()
    assert (upload_dir / "jobs" / f"{file_id}.d.req").is_file()
    req_text = (upload_dir / "jobs" / f"{file_id}.d.req").read_text()
    assert f"input={upload_dir}/{file_id}.d" in req_text

    # still converting until the converter reports done
    assert api_client.get(f"/api/sessions/{sid}").json()["status"] == "converting"

    # sidecar finished: gzip output, negative polarity (also exercises the
    # gzip-aware polarity detection)
    _simulate_converter(str(upload_dir), file_id, masses, ints, polarity="negative")
    meta = api_client.get(f"/api/sessions/{sid}").json()
    assert meta["status"] == "ready"
    assert meta["n_points"] == len(masses)
    assert meta["polarity"] == "negative"
    assert meta["polarity_source"] == "file"

    # the loaded session works end-to-end
    r = api_client.post(f"/api/sessions/{sid}/deisotope", json={})
    assert r.status_code == 200, r.text

    # a second session reuses the conversion output (ready immediately)
    r = api_client.post("/api/sessions", json={"source": "upload", "file_id": file_id})
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_d_conversion_error(api_client, small_spectrum):
    from app.config import get_settings

    masses, ints = small_spectrum
    r = api_client.post("/api/upload", files={
        "file": ("Broken.d.zip", _d_zip_bytes(), "application/zip"),
    })
    file_id = r.json()["file_id"]
    r = api_client.post("/api/sessions", json={"source": "upload", "file_id": file_id})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["status"] == "converting"

    _simulate_converter(str(Path(get_settings().upload_dir)), file_id, masses, ints,
                        polarity="positive", exit_code=1,
                        stderr_tail="msconvert: could not read the input file")
    meta = api_client.get(f"/api/sessions/{sid}").json()
    assert meta["status"] == "error"
    assert "could not read the input file" in meta["convert_error"]

    # pipeline steps are rejected while the session is not ready
    assert api_client.post(f"/api/sessions/{sid}/deisotope", json={}).status_code == 400
    assert api_client.delete(f"/api/sessions/{sid}").status_code == 200


def test_d_conversion_timeout(api_client, small_spectrum):
    r = api_client.post("/api/upload", files={
        "file": ("Slow.d.zip", _d_zip_bytes(), "application/zip"),
    })
    file_id = r.json()["file_id"]
    r = api_client.post("/api/sessions", json={"source": "upload", "file_id": file_id})
    sid = r.json()["session_id"]
    assert r.json()["status"] == "converting"

    store = api_client.app.state.store
    store.get(sid).convert_started_at -= 61 * 60  # beyond the 60 min default
    meta = api_client.get(f"/api/sessions/{sid}").json()
    assert meta["status"] == "error"
    assert "timed out" in meta["convert_error"]


def test_d_archive_not_allowed_from_folder(api_client):
    from app.config import get_settings

    (Path(get_settings().spectra_dir) / "x.d.zip").write_bytes(_d_zip_bytes())
    r = api_client.post("/api/sessions", json={"source": "folder", "path": "x.d.zip"})
    assert r.status_code == 400
    assert "server folder" in r.json()["detail"]


def test_upload_unknown_extension_rejected(api_client):
    r = api_client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 200
    file_id = r.json()["file_id"]
    r = api_client.post("/api/sessions", json={"source": "upload", "file_id": file_id})
    assert r.status_code == 400
    assert "uploads:" in r.json()["detail"]


def test_invalid_d_zip_rejected(api_client):
    r = api_client.post("/api/upload", files={
        "file": ("broken.d.zip", b"not a zip at all", "application/zip"),
    })
    assert r.status_code == 200
    file_id = r.json()["file_id"]
    r = api_client.post("/api/sessions", json={"source": "upload", "file_id": file_id})
    assert r.status_code == 400
    assert "extract" in r.json()["detail"]


def test_upload_mzxml_gz(api_client, small_spectrum):
    from app.config import get_settings

    masses, ints = small_spectrum
    raw_path = Path(get_settings().spectra_dir) / "raw_tmp.mzXML"
    write_mzxml(raw_path, masses, ints, polarity="negative")
    raw = raw_path.read_bytes()
    raw_path.unlink()

    r = api_client.post("/api/upload", files={
        "file": ("raw.mzXML.gz", gzip.compress(raw), "application/gzip"),
    })
    assert r.status_code == 200, r.text
    file_id = r.json()["file_id"]

    r = api_client.post("/api/sessions", json={"source": "upload", "file_id": file_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready"
    assert body["n_points"] == len(masses)
    assert body["polarity"] == "negative"
    assert body["polarity_source"] == "file"
