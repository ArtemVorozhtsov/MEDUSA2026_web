"""FastAPI application entry point.

Run (Docker):  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
A single worker is mandatory — session state is in process memory.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__
from .api import analysis, figures, files, sessions
from .config import get_settings
from .errors import register_error_handlers
from .jobs import CpuGate
from .state import SessionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("medusa_web")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    app.state.store = SessionStore(settings.max_active_sessions)
    app.state.cpu_gate = CpuGate(settings.max_concurrent_cpu_jobs)
    app.state.models = {"cgb": False, "transformer": False}

    # Load both models once at startup (no side effects on engine import).
    try:
        from mass_automation.deisotoping.process import MlDeisotoper
        app.state.deisotoper = MlDeisotoper().load(settings.cgb_model)
        app.state.models["cgb"] = True
        logger.info("loaded deisotoping model: %s", settings.cgb_model)
    except Exception:
        logger.exception("failed to load deisotoping model from %s", settings.cgb_model)
        app.state.deisotoper = None

    try:
        import torch
        from mass_automation.formula.Transformer import TransformerModel
        model = TransformerModel.load_from_checkpoint(settings.transformer_ckpt, map_location="cpu")
        model.eval()
        app.state.transformer = model
        app.state.models["transformer"] = True
        logger.info("loaded transformer model: %s", settings.transformer_ckpt)
    except Exception:
        logger.exception("failed to load transformer model from %s", settings.transformer_ckpt)
        app.state.transformer = None

    logger.info(
        "MEDUSA 2026 web service ready: spectra_dir=%s upload_dir=%s max_sessions=%d cpu_jobs=%d",
        settings.spectra_dir, settings.upload_dir,
        settings.max_active_sessions, settings.max_concurrent_cpu_jobs,
    )
    yield
    app.state.store.clear()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="MEDUSA 2026 Web Service",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # university LAN, no auth in v1
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)

    app.include_router(files.router)
    app.include_router(sessions.router)
    app.include_router(analysis.router)
    app.include_router(figures.router)

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "models": app.state.models,
            "active_sessions": len(app.state.store),
            "max_sessions": settings.max_active_sessions,
        }

    class _NoCacheStatic(BaseHTTPMiddleware):
        """Revalidate UI assets on every load (no stale JS/CSS after rebuilds)."""

        def _target(self, path: str) -> bool:
            return path in ("/", "/index.html") or path.rsplit(".", 1)[-1].lower() in {"js", "css"}

        async def dispatch(self, request, call_next):
            response = await call_next(request)
            if self._target(request.url.path):
                response.headers["Cache-Control"] = "no-cache"
            return response

    app.add_middleware(_NoCacheStatic)

    # Static UI (single page). Mounted last so /api/* and /healthz win.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
