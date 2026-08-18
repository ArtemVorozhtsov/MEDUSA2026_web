"""Environment-based settings for the MEDUSA 2026 web service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # repo convention: honor a .env file in the CWD when present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _repo_root() -> Path:
    """Directory that contains both medusa_web/ and MEDUSA2026/ (dev checkout)."""
    return Path(__file__).resolve().parents[2]


def _default_model(name: str, repo_rel: str) -> str:
    """Prefer an env override, then the baked-in image path, then the dev checkout."""
    if os.environ.get(name):
        return os.environ[name]
    baked = Path("/app/models") / Path(repo_rel).name
    if baked.exists():
        return str(baked)
    return str(_repo_root() / repo_rel)


@dataclass(frozen=True)
class Settings:
    spectra_dir: str
    upload_dir: str
    max_upload_mb: int
    max_active_sessions: int
    max_concurrent_cpu_jobs: int
    port: int
    cgb_model: str
    transformer_ckpt: str


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        spectra_dir=os.environ.get("SPECTRA_DIR", "/data/spectra"),
        upload_dir=os.environ.get("UPLOAD_DIR", "/data/uploads"),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 300),
        max_active_sessions=_env_int("MAX_ACTIVE_SESSIONS", 8),
        max_concurrent_cpu_jobs=_env_int("MAX_CONCURRENT_CPU_JOBS", 2),
        port=_env_int("PORT", 8000),
        cgb_model=_default_model("CGB_MODEL", "MEDUSA2026/data/models/charge1_big_optuna150.pkl"),
        transformer_ckpt=_default_model("TRANSFORMER_CKPT", "MEDUSA2026/nn_models/transfomer_classifier.ckpt"),
    )


def reset_settings_cache() -> None:
    """Test helper: re-read environment variables."""
    get_settings.cache_clear()
