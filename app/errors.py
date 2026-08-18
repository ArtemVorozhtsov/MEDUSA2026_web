"""Application error types and the JSON error handler."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("medusa_web.errors")


class ApiError(Exception):
    """Error rendered as {"detail": "<message>"} with an HTTP status code."""

    status_code = 500

    def __init__(self, detail: str, status_code: int | None = None):
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class BadRequest(ApiError):
    status_code = 400


class NotFound(ApiError):
    status_code = 404


class SessionLimit(ApiError):
    status_code = 409


class JobBusy(ApiError):
    status_code = 409

    def __init__(self, detail: str = "job slot busy, retry"):
        super().__init__(detail)


class PipelineError(ApiError):
    """Raised by engine steps; maps to a 4xx/5xx with a human-readable message."""

    def __init__(self, detail: str, status_code: int = 500):
        super().__init__(detail, status_code)


class ModelUnavailable(ApiError):
    status_code = 503


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Last-resort guard so every error is JSON {"detail": ...} (API contract),
        # even for unexpected exceptions raised deep in the core library.
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": f"internal error: {exc}"}, status_code=500)
