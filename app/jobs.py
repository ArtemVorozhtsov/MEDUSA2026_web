"""Bounded CPU-job execution: at most MAX_CONCURRENT_CPU_JOBS heavy steps at once.

uvicorn runs with a single worker, so an asyncio semaphore plus a thread pool
gives in-process concurrency control. Saturated callers get an immediate
409 "job slot busy, retry" instead of an unbounded queue.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable

from fastapi import FastAPI

from .errors import JobBusy

logger = logging.getLogger("medusa_web.jobs")


class CpuGate:
    """Atomic non-blocking gate: at most ``max_jobs`` CPU jobs at once.

    A plain ``asyncio.Semaphore`` cannot be probed non-blockingly on
    Python 3.9 (``wait_for(acquire(), 0)`` always cancels), so the gate
    uses a thread-safe counter.
    """

    def __init__(self, max_jobs: int):
        self.max_jobs = max(1, int(max_jobs))
        self._busy = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._busy >= self.max_jobs:
                return False
            self._busy += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._busy = max(0, self._busy - 1)

    @property
    def busy(self) -> int:
        with self._lock:
            return self._busy


async def run_cpu_job(app: FastAPI, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` in a worker thread; raises JobBusy (HTTP 409) if the pool is full."""
    gate: CpuGate = app.state.cpu_gate
    if not gate.try_acquire():
        logger.warning("CPU job slot busy; returning 409")
        raise JobBusy()
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    finally:
        gate.release()
