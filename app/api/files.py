"""File access: server/NFS folder listing and browser uploads."""
from __future__ import annotations

import os
import re
import uuid

from fastapi import APIRouter, File, Request, UploadFile

from ..config import Settings
from ..errors import ApiError
from ..engine.spectrum_io import list_spectra_files

router = APIRouter(prefix="/api", tags=["files"])

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@router.get("/files")
def files_list(request: Request) -> dict:
    """List *.mzXML under SPECTRA_DIR (depth <= 3)."""
    settings: Settings = request.app.state.settings
    files = list_spectra_files(settings.spectra_dir)
    return {"root": settings.spectra_dir, "files": files}


@router.post("/upload")
async def upload(file: UploadFile = File(...), request: Request = None) -> dict:
    """Store an uploaded spectrum in UPLOAD_DIR; returns a file_id (relative name)."""
    settings: Settings = request.app.state.settings
    max_bytes = settings.max_upload_mb * 1024 * 1024
    os.makedirs(settings.upload_dir, exist_ok=True)

    base_name = os.path.basename(file.filename or "upload.mzXML")
    base_name = _SAFE_NAME_RE.sub("_", base_name)[-120:] or "upload.mzXML"
    file_id = f"{uuid.uuid4().hex[:10]}_{base_name}"
    dest = os.path.join(settings.upload_dir, file_id)

    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise ApiError(
                        f"upload too large: limit is {settings.max_upload_mb} MB", 413
                    )
                out.write(chunk)
    except ApiError:
        os.unlink(dest)
        raise
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(dest):
            os.unlink(dest)
        raise ApiError(f"upload failed: {exc}", 500)

    if size == 0:
        os.unlink(dest)
        raise ApiError("uploaded file is empty", 400)

    return {"file_id": file_id, "size": size, "name": file_id}
