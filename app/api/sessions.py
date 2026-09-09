"""Session lifecycle: create (from folder file or upload), inspect, delete."""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Literal, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..engine.pipeline import element_symbols
from ..engine.spectrum_io import extract_d_zip, load_spectrum
from ..errors import ApiError, BadRequest, NotFound
from ..jobs import run_cpu_job

logger = logging.getLogger("medusa_web.sessions")

router = APIRouter(prefix="/api", tags=["sessions"])


#: which file kinds each source accepts (.d is a directory format -> zip uploads only)
FOLDER_EXTENSIONS = (".mzxml",)
UPLOAD_EXTENSIONS = (".mzxml", ".mzxml.gz", ".d.zip", ".zip")
D_ZIP_EXTENSIONS = (".d.zip", ".zip")


class SessionCreate(BaseModel):
    source: str = Field(..., description="'folder' or 'upload'")
    path: Optional[str] = None
    file_id: Optional[str] = None


def _resolve_path(settings, source: str, path: Optional[str], file_id: Optional[str]) -> str:
    if source == "folder":
        if not path:
            raise BadRequest("'path' is required for source=folder")
        root = os.path.realpath(settings.spectra_dir)
        full = os.path.realpath(os.path.join(root, path))
        if not full.startswith(root + os.sep) and full != root:
            raise BadRequest("path escapes the spectra folder")
        return full
    if source == "upload":
        if not file_id:
            raise BadRequest("'file_id' is required for source=upload")
        root = os.path.realpath(settings.upload_dir)
        full = os.path.realpath(os.path.join(root, file_id))
        if not full.startswith(root + os.sep):
            raise BadRequest("file_id escapes the uploads folder")
        return full
    raise BadRequest("source must be 'folder' or 'upload'")


@router.post("/sessions")
async def create_session(body: SessionCreate, request: Request) -> dict:
    settings = request.app.state.settings
    store = request.app.state.store
    full_path = _resolve_path(settings, body.source, body.path, body.file_id)
    if not os.path.isfile(full_path):
        raise NotFound(f"file not found: {body.path or body.file_id}")

    name_lower = full_path.lower()
    if body.source == "folder":
        if not name_lower.endswith(FOLDER_EXTENSIONS):
            raise BadRequest("only .mzXML files are supported from the server folder")
    elif not name_lower.endswith(UPLOAD_EXTENSIONS):
        raise BadRequest(
            "uploads: only .mzXML, .mzXML.gz or a zipped .d folder (.d.zip) are supported")

    session = store.create(os.path.basename(full_path), body.source, full_path)
    load_path = full_path
    try:
        if name_lower.endswith(D_ZIP_EXTENSIONS):
            # Agilent .d archive: extract, then convert with the sidecar
            # (msconvert under wine) or reuse a previous conversion output.
            converted = _prepare_d_conversion(session, full_path, settings)
            if converted is None:
                return _session_payload(session, None)
            load_path = converted
        data = await run_cpu_job(request.app, load_spectrum, load_path)
    except ApiError as exc:
        store.delete(session.id)
        raise
    except Exception as exc:  # noqa: BLE001 - loading errors must not leak a broken session
        store.delete(session.id)
        raise BadRequest(f"could not load spectrum: {exc}")
    session.mark_loaded(data)
    session.log("info", f"loaded {session.file_name}: {data['n_points']} points, "
                        f"m/z {data['mz_min']:.2f}..{data['mz_max']:.2f} in {data['load_time_s']:.2f}s")
    return _session_payload(session, data)


def _session_payload(session, data) -> dict:
    """Creation / status response; data is None while a .d conversion is pending."""
    return {
        "session_id": session.id,
        "file_name": session.file_name,
        "source": session.source,
        "n_points": data["n_points"] if data else None,
        "mz_range": [data["mz_min"], data["mz_max"]] if data else None,
        "n_scans": data["n_scans"] if data else None,
        "load_time_s": data["load_time_s"] if data else None,
        "polarity": session.polarity,
        "polarity_source": session.polarity_source,
        "status": session.status,
        "convert_error": session.convert_error,
        "logs": session.recent_logs(),
    }


def _prepare_d_conversion(session, zip_path: str, settings) -> Optional[str]:
    """Extract the .d archive; return an existing conversion output path, or
    register a job for the converter sidecar and leave the session "converting".
    """
    file_id = os.path.basename(zip_path)
    upload_dir = settings.upload_dir
    d_dir = os.path.join(upload_dir, f"{file_id}.d")
    if not os.path.isdir(d_dir):
        try:
            extract_d_zip(zip_path, d_dir, settings.max_upload_mb * 1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            raise BadRequest(f"could not extract .d archive: {exc}")
    for candidate in (f"{file_id}.d.mzXML.gz", f"{file_id}.d.mzxml.gz"):
        out = os.path.join(upload_dir, candidate)
        if os.path.isfile(out):
            return out
    _write_conversion_job(upload_dir, file_id)
    session.mark_converting(os.path.join(upload_dir, f"{file_id}.d.mzXML.gz"), file_id)
    session.log("info", f"conversion started for {file_id} (msconvert/wine, may take minutes)")
    return None


def _write_conversion_job(upload_dir: str, file_id: str) -> None:
    """File-based job for the converter sidecar (atomic write)."""
    jobs_dir = os.path.join(upload_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    req = os.path.join(jobs_dir, f"{file_id}.d.req")
    tmp = req + ".tmp"
    content = (
        f"input={upload_dir}/{file_id}.d\n"
        f"outfile={file_id}.d.mzXML\n"
        f"outdir={upload_dir}\n"
    )
    with open(tmp, "w", encoding="ascii") as f:
        f.write(content)
    os.replace(tmp, req)


@router.get("/sessions/{session_id}")
def get_session(request: Request, session_id: str) -> dict:
    session = request.app.state.store.get(session_id)
    if session.status == "converting":
        _advance_conversion(request, session)
    return {**session.summary(), "logs": session.recent_logs()}


def _advance_conversion(request: Request, session) -> None:
    """Poll the converter's .done marker; load the result when it appears."""
    settings = request.app.state.settings
    file_id = session.convert_file_id
    done_path = os.path.join(settings.upload_dir, "jobs", f"{file_id}.d.done")
    if os.path.isfile(done_path):
        try:
            with open(done_path, "r", encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError:
            body = ""
        if body.startswith("exit=0") and session.convert_output and os.path.isfile(session.convert_output):
            gate = request.app.state.cpu_gate
            if not gate.try_acquire():
                return  # CPU pool busy; the next poll retries
            try:
                data = load_spectrum(session.convert_output)
            except Exception as exc:  # noqa: BLE001
                session.mark_convert_error(f"conversion finished but loading failed: {exc}")
                return
            finally:
                gate.release()
            session.mark_loaded(data)
            session.log("info", f"loaded {os.path.basename(session.convert_output)}: "
                                f"{data['n_points']} points in {data['load_time_s']:.2f}s")
        else:
            tail = " ".join(body.replace("\n", " ").split())
            session.mark_convert_error(f"conversion failed: {tail or 'unknown error'}")
        return
    if session.convert_started_at and (
        time.time() - session.convert_started_at > settings.convert_timeout_min * 60
    ):
        session.mark_convert_error(f"conversion timed out after {settings.convert_timeout_min} min")


class PolarityIn(BaseModel):
    polarity: Literal["positive", "negative"]


@router.post("/sessions/{session_id}/polarity")
def set_session_polarity(session_id: str, body: PolarityIn, request: Request) -> dict:
    """Override the ion polarity (charge sign) used by steps 6-7."""
    session = request.app.state.store.get(session_id)
    session.set_polarity(body.polarity)
    session.log("info", f"polarity set to {body.polarity} (manual override)")
    return {"polarity": session.polarity, "polarity_source": session.polarity_source}


def _purge_upload_if_configured(settings, session) -> None:
    """Drop the uploaded file on session reset if DELETE_UPLOADS_ON_SESSION_RESET is on.

    Folder-source files are never touched (they live on a shared volume).
    For .d archives the derived artifacts (extracted dir, converted spectrum,
    job files) are purged as well.
    """
    if not settings.delete_uploads_on_session_reset or session.source != "upload":
        return
    try:
        if os.path.isfile(session.path):
            os.unlink(session.path)
            logger.info("purged uploaded file %s", session.path)
        if session.path.lower().endswith(D_ZIP_EXTENSIONS):
            _purge_d_artifacts(settings, os.path.basename(session.path))
    except OSError:
        logger.warning("could not purge uploaded file %s", session.path, exc_info=True)


def _purge_d_artifacts(settings, file_id: str) -> None:
    upload_dir = settings.upload_dir
    d_dir = os.path.join(upload_dir, f"{file_id}.d")
    if os.path.isdir(d_dir):
        shutil.rmtree(d_dir, ignore_errors=True)
        logger.info("purged extracted .d dir for %s", file_id)
    for name in (f"{file_id}.d.mzXML.gz", f"{file_id}.d.mzxml.gz"):
        out = os.path.join(upload_dir, name)
        if os.path.isfile(out):
            os.unlink(out)
    for kind in (".d.req", ".d.done", ".d.log"):
        job = os.path.join(upload_dir, "jobs", file_id + kind)
        if os.path.isfile(job):
            os.unlink(job)


@router.delete("/sessions/{session_id}")
def delete_session(request: Request, session_id: str) -> dict:
    session = request.app.state.store.delete(session_id)
    _purge_upload_if_configured(request.app.state.settings, session)
    return {"deleted": session.id}


@router.get("/elements")
def elements() -> dict:
    """All 119 element symbols (for UI validation of the element selector)."""
    return {"elements": element_symbols()}
