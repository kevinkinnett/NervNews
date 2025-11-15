"""ASGI entrypoint for the NervNews dashboard."""
from __future__ import annotations

from src.config.settings import load_settings
from src.config.store import ensure_seed_data
from src.db.session import create_engine_from_url, create_session_factory, init_db
from .app import create_app

_settings = load_settings()
_engine = create_engine_from_url(_settings.database_url)
init_db(_engine)
_session_factory = create_session_factory(_engine)
ensure_seed_data(_session_factory, _settings)

app = create_app(_session_factory)

__all__ = ["app"]
