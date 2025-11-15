"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, sessionmaker


def get_session(request: Request) -> Session:
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates


__all__ = ["get_session", "get_templates"]
