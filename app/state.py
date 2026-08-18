"""In-memory session state: one dict entry per analysis session.

A *session* is created when a spectrum is loaded (from the server folder or an
upload) and holds:

* the raw spectrum (m/z + intensities, float64) and a ``Spectrum`` object;
* per-step results, each cached by a hash of the step parameters (re-running
  a step with unchanged parameters is a no-op);
* a small ring buffer of log messages (visible in the UI log area);
* a per-session ``threading.Lock`` guarding cache writes.

All state is process-local; uvicorn must run with a single worker.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .errors import NotFound, SessionLimit

logger = logging.getLogger("medusa_web.state")

LOG_BUFFER_SIZE = 200


@dataclass
class LogEntry:
    ts: float
    level: str
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return {"ts": self.ts, "level": self.level, "message": self.message}


class Session:
    """State of one analysis session (one mzXML file, first scan)."""

    __slots__ = (
        "id", "file_name", "source", "path",
        "masses", "ints", "spectrum",
        "n_points", "mz_min", "mz_max", "n_scans", "load_time_s",
        "created_at", "last_access",
        "logs", "lock",
        # step 2: deisotoping
        "deisotope_cache", "ion_id", "ion_info", "ion_charge", "last_deisotope_hash",
        # step 3: elements
        "elements_cache", "ion_probs", "ion_probs_for_deiso", "elements_element",
        # step 4/5: threshold + highlight
        "knee_cache", "threshold_state", "point_probs", "point_probs_element",
        # step 6: formulas
        "formulas_cache",
    )

    def __init__(self, session_id: str, file_name: str, source: str, path: str):
        self.id = session_id
        self.file_name = file_name
        self.source = source
        self.path = path

        self.masses: Optional[np.ndarray] = None
        self.ints: Optional[np.ndarray] = None
        self.spectrum = None  # mass_automation.experiment.Spectrum

        self.n_points: Optional[int] = None
        self.mz_min: Optional[float] = None
        self.mz_max: Optional[float] = None
        self.n_scans: Optional[int] = None
        self.load_time_s: Optional[float] = None

        self.created_at = time.time()
        self.last_access = self.created_at
        self.logs: Deque[LogEntry] = deque(maxlen=LOG_BUFFER_SIZE)
        self.lock = threading.Lock()

        self.deisotope_cache: Dict[str, Dict[str, Any]] = {}
        self.ion_id: Optional[np.ndarray] = None          # per-point int32, -1 = noise
        self.ion_info: List[Dict[str, Any]] = []          # per-ion summary (latest run)
        self.ion_charge: Optional[np.ndarray] = None      # per-ion mean charge
        self.last_deisotope_hash: Optional[str] = None

        self.elements_cache: Dict[str, Dict[str, Any]] = {}
        self.ion_probs: Optional[np.ndarray] = None       # (n_ions, 119), NaN = skipped
        self.ion_probs_for_deiso: Optional[str] = None
        self.elements_element: Optional[str] = None

        self.knee_cache: Dict[str, Dict[str, Any]] = {}
        self.threshold_state: Dict[str, Dict[str, Any]] = {}
        self.point_probs: Optional[np.ndarray] = None     # per-point float32 probabilities
        self.point_probs_element: Optional[str] = None

        self.formulas_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    def log(self, level: str, message: str) -> None:
        entry = LogEntry(time.time(), level, message)
        self.logs.append(entry)
        log_fn = logger.info if level == "info" else (logger.warning if level == "warning" else logger.error)
        log_fn("[%s] %s", self.id, message)

    def recent_logs(self, limit: int = 25) -> List[Dict[str, Any]]:
        return [e.as_dict() for e in list(self.logs)[-limit:]]

    def mark_loaded(self, data: Dict[str, Any]) -> None:
        self.masses = data["masses"]
        self.ints = data["ints"]
        self.spectrum = data["spectrum"]
        self.n_points = data["n_points"]
        self.mz_min = data["mz_min"]
        self.mz_max = data["mz_max"]
        self.n_scans = data["n_scans"]
        self.load_time_s = data["load_time_s"]

    def is_loaded(self) -> bool:
        return self.masses is not None

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.id,
            "file_name": self.file_name,
            "source": self.source,
            "path": self.path,
            "n_points": self.n_points,
            "mz_range": [self.mz_min, self.mz_max],
            "n_scans": self.n_scans,
            "load_time_s": self.load_time_s,
            "created_at": self.created_at,
            "n_ions": len(self.ion_info) if self.ion_id is not None else None,
            "element": self.elements_element,
            "completed_steps": {
                "deisotope": bool(self.deisotope_cache),
                "elements": bool(self.elements_cache),
                "threshold": bool(self.threshold_state),
                "formulas": bool(self.formulas_cache),
            },
        }


class SessionStore:
    """dict session_id -> Session with LRU tracking and a size limit."""

    def __init__(self, max_active_sessions: int = 8):
        self.max_active_sessions = max_active_sessions
        self._sessions: Dict[str, Session] = {}
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def create(self, file_name: str, source: str, path: str) -> Session:
        with self._lock:
            if len(self._sessions) >= self.max_active_sessions:
                raise SessionLimit(
                    f"max active sessions reached ({self.max_active_sessions}); "
                    "delete an existing session first (New session button)"
                )
            session = Session(
                session_id=uuid.uuid4().hex[:12],
                file_name=file_name,
                source=source,
                path=path,
            )
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise NotFound(f"session {session_id!r} not found (container restart?)")
        session.last_access = time.time()
        return session

    def delete(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise NotFound(f"session {session_id!r} not found")
        # drop references so large arrays can be freed
        session.masses = None
        session.ints = None
        session.spectrum = None
        session.ion_id = None
        session.ion_probs = None
        session.point_probs = None
        logger.info("session %s deleted", session_id)
        return session

    def clear(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                session.masses = None
                session.ints = None
                session.spectrum = None
            self._sessions.clear()

    def lru_session(self) -> Optional[Session]:
        with self._lock:
            if not self._sessions:
                return None
            return min(self._sessions.values(), key=lambda s: s.last_access)
