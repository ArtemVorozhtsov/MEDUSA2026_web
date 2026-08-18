"""Session lifecycle: create (from folder file or upload), inspect, delete."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..engine.pipeline import element_symbols
from ..engine.spectrum_io import load_spectrum
from ..errors import ApiError, BadRequest, NotFound
from ..jobs import run_cpu_job

router = APIRouter(prefix="/api", tags=["sessions"])


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
    if not full_path.lower().endswith(".mzxml"):
        raise BadRequest("only .mzXML files are supported")

    session = store.create(os.path.basename(full_path), body.source, full_path)
    try:
        data = await run_cpu_job(request.app, load_spectrum, full_path)
    except ApiError as exc:
        store.delete(session.id)
        raise
    except Exception as exc:  # noqa: BLE001 - loading errors must not leak a broken session
        store.delete(session.id)
        raise BadRequest(f"could not load spectrum: {exc}")
    session.mark_loaded(data)
    session.log("info", f"loaded {session.file_name}: {data['n_points']} points, "
                        f"m/z {data['mz_min']:.2f}..{data['mz_max']:.2f} in {data['load_time_s']:.2f}s")
    return {
        "session_id": session.id,
        "file_name": session.file_name,
        "source": session.source,
        "n_points": data["n_points"],
        "mz_range": [data["mz_min"], data["mz_max"]],
        "n_scans": data["n_scans"],
        "load_time_s": data["load_time_s"],
        "logs": session.recent_logs(),
    }


@router.get("/sessions/{session_id}")
def get_session(request: Request, session_id: str) -> dict:
    session = request.app.state.store.get(session_id)
    return {**session.summary(), "logs": session.recent_logs()}


@router.delete("/sessions/{session_id}")
def delete_session(request: Request, session_id: str) -> dict:
    session = request.app.state.store.delete(session_id)
    return {"deleted": session.id}


@router.get("/elements")
def elements() -> dict:
    """All 119 element symbols (for UI validation of the element selector)."""
    return {"elements": element_symbols()}
