"""Factory for the FastAPI dashboard application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, sessionmaker

from .routes import admin, dashboard, data_viewer


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def create_app(session_factory: sessionmaker[Session]) -> FastAPI:
    """Create a configured FastAPI application instance."""

    app = FastAPI(title="NervNews Dashboard", version="0.1.0")
    app.state.session_factory = session_factory
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.include_router(dashboard.router)
    app.include_router(data_viewer.router)
    app.include_router(admin.router, prefix="/admin")

    return app


__all__ = ["create_app"]
