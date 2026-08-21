"""Figure endpoints: windowed (downsampled) spectrum and the compare figure."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request

from ..engine.downsample import COMPARE_HARD_MAX_PTS, DEFAULT_MAX_PTS, downsample_window
from ..engine.pipeline import compare
from ..errors import BadRequest
from ..jobs import run_cpu_job

router = APIRouter(prefix="/api", tags=["figures"])


@router.get("/sessions/{session_id}/spectrum")
def spectrum_window(
    request: Request,
    session_id: str,
    x0: Optional[float] = Query(None),
    x1: Optional[float] = Query(None),
    max_pts: int = Query(DEFAULT_MAX_PTS, ge=1, le=5000),
    layers: str = Query("raw"),
) -> dict:
    """Windowed, peak-preserving decimated spectrum (never more than max_pts points)."""
    session = request.app.state.store.get(session_id)
    if not session.is_loaded():
        raise BadRequest("spectrum not loaded for this session")

    x0 = x0 if x0 is not None else session.mz_min
    x1 = x1 if x1 is not None else session.mz_max
    if x1 <= x0:
        raise BadRequest("x1 must be greater than x0")

    requested = {part.strip() for part in layers.split(",") if part.strip()}
    unknown = requested - {"raw", "ions", "probs"}
    if unknown:
        raise BadRequest(f"unknown layer(s): {', '.join(sorted(unknown))}")

    aux = {}
    if "ions" in requested:
        if session.ion_id is None:
            raise BadRequest("run deisotoping (step 2) first to get the ions layer")
        aux["ion_id"] = session.ion_id
    if "probs" in requested:
        if session.point_probs is None:
            raise BadRequest("apply a threshold (step 5) first to get the probs layer")
        aux["prob"] = session.point_probs

    window = downsample_window(session.masses, session.ints, x0, x1, max_pts, aux=aux or None)

    out = {
        "masses": [float(v) for v in window["masses"]],
        "ints": [float(v) for v in window["ints"]],
        "decimated": bool(window["decimated"]),
        "n_in_window": window["n_in_window"],
        "max_pts": max_pts,
        "full_range": window["full_range"],
        "window_range": [float(x0), float(x1)],
    }
    if "ion_id" in window:
        out["ion_id"] = [int(v) for v in window["ion_id"]]
    if "prob" in window:
        out["prob"] = [float(v) for v in window["prob"]]
    return out


@router.get("/sessions/{session_id}/compare")
async def compare_endpoint(
    request: Request,
    session_id: str,
    ion_id: int = Query(...),
    formula: str = Query(..., min_length=2),
    max_pts: Optional[int] = Query(None, ge=1, le=COMPARE_HARD_MAX_PTS),
    charge: Optional[float] = Query(None, ge=-10, le=10),
) -> dict:
    """Plotly figure: experimental window around the ion vs theoretical isotope pattern.

    ``max_pts`` caps the experimental trace. Default: the exact (undecimated)
    window whenever it fits ``COMPARE_HARD_MAX_PTS`` points.
    """
    return await run_cpu_job(request.app, compare,
                             request.app.state.store, session_id, ion_id, formula, max_pts, charge)
