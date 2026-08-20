"""Pipeline steps 2-6 as API endpoints (synchronous; CPU-bound via thread pool)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..engine.pipeline import (
    FORMULA_PRESETS,
    MAX_FORMULA_WORKERS,
    DeisotopeParams,
    ElementsParams,
    FormulaParams,
    ThresholdParams,
    apply_threshold,
    classify_elements,
    deisotope,
    formulas,
    knee,
)
from ..errors import ModelUnavailable
from ..jobs import run_cpu_job

router = APIRouter(prefix="/api", tags=["analysis"])


def _models(app) -> None:
    if app.state.deisotoper is None or app.state.transformer is None:
        raise ModelUnavailable(
            "models are not loaded (check CGB_MODEL / TRANSFORMER_CKPT in the logs)"
        )


# --- step 2: deisotoping ------------------------------------------------- #
@router.post("/sessions/{session_id}/deisotope")
async def post_deisotope(request: Request, session_id: str, body: DeisotopeParams) -> dict:
    _models(request.app)
    result = await run_cpu_job(request.app, deisotope,
                               request.app.state.store, session_id, body,
                               request.app.state.deisotoper)
    result["logs"] = request.app.state.store.get(session_id).recent_logs()
    return result


# --- step 3: element classification --------------------------------------- #
class ElementsBody(BaseModel):
    element: str = "Ir"


@router.post("/sessions/{session_id}/elements")
async def post_elements(request: Request, session_id: str, body: ElementsBody) -> dict:
    _models(request.app)
    result = await run_cpu_job(request.app, classify_elements,
                               request.app.state.store, session_id,
                               ElementsParams(element=body.element),
                               request.app.state.transformer)
    result["logs"] = request.app.state.store.get(session_id).recent_logs()
    return result


# --- step 4: knee ---------------------------------------------------------- #
@router.get("/sessions/{session_id}/knee")
def get_knee(request: Request, session_id: str, element: str = "Ir") -> dict:
    result = knee(request.app.state.store, session_id, element)
    return result


# --- step 5: threshold / highlight ------------------------------------------ #
@router.post("/sessions/{session_id}/threshold")
def post_threshold(request: Request, session_id: str, body: ThresholdParams) -> dict:
    result = apply_threshold(request.app.state.store, session_id, body)
    result["logs"] = request.app.state.store.get(session_id).recent_logs()
    return result


# --- step 6: formulas -------------------------------------------------------- #
class FormulaBody(BaseModel):
    ion_id: int
    elements: Dict[str, Tuple[int, int]]
    mass_threshold_ppm: float = 4.0
    num_workers: int = Field(default=1, ge=1, le=MAX_FORMULA_WORKERS)
    max_chunk_size: int = 10_000


@router.post("/sessions/{session_id}/formulas")
async def post_formulas(request: Request, session_id: str, body: FormulaBody) -> dict:
    params = FormulaParams(
        ion_id=body.ion_id,
        elements=dict(body.elements),
        mass_threshold_ppm=body.mass_threshold_ppm,
        num_workers=body.num_workers,
        max_chunk_size=body.max_chunk_size,
    )
    result = await run_cpu_job(request.app, formulas,
                               request.app.state.store, session_id, params)
    result["logs"] = request.app.state.store.get(session_id).recent_logs()
    return result


@router.get("/formula_presets")
def formula_presets() -> dict:
    return {name: {el: list(limits) for el, limits in table.items()}
            for name, table in FORMULA_PRESETS.items()}
