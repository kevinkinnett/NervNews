"""Administrative views and form handlers."""
from __future__ import annotations

import json
import logging
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from src.config.settings import load_settings
from src.config.store import (
    CONFIG_LLM_BASE_URL,
    CONFIG_LLM_MODEL,
    CONFIG_LLM_MODEL_PATH,
    CONFIG_LLM_PROVIDER,
    CONFIG_SUMMARIZATION_INTERVAL,
    CONFIG_ACTIVE_PROFILE_ID,
    build_llm_settings,
    get_config,
    set_config,
)
from src.db.models import Feed, UserProfile
from src.llm import LLMClient, LLMClientError
from src.web.dependencies import get_session, get_templates

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "on", "yes", "y"}


def _redirect(path: str, message: Optional[str] = None) -> RedirectResponse:
    url = path
    if message:
        query = urlencode({"msg": message})
        url = f"{path}?{query}"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/", response_class=HTMLResponse)
def admin_home(
    request: Request,
    session: Session = Depends(get_session),
):
    templates = get_templates(request)
    feeds = session.query(Feed).order_by(Feed.name.asc()).all()
    active_profile = (
        session.query(UserProfile)
        .filter(UserProfile.is_active.is_(True))
        .order_by(UserProfile.updated_at.desc())
        .first()
    )
    interval = get_config(
        session,
        CONFIG_SUMMARIZATION_INTERVAL,
        3600,
    )
    provider = get_config(session, CONFIG_LLM_PROVIDER, "ollama")
    model_name = get_config(session, CONFIG_LLM_MODEL, "qwen3:30b")
    if model_name is None:
        model_name = get_config(session, CONFIG_LLM_MODEL_PATH, "qwen3:30b")
    base_url = get_config(session, CONFIG_LLM_BASE_URL, "http://127.0.0.1:11434")
    message = request.query_params.get("msg")

    context = {
        "request": request,
        "feeds": feeds,
        "profile": active_profile,
        "interval": interval,
        "provider": provider,
        "model": model_name,
        "base_url": base_url,
        "message": message,
    }
    return templates.TemplateResponse("admin/index.html", context)


@router.post("/feeds", response_class=RedirectResponse)
def create_feed(
    name: str = Form(...),
    url: str = Form(...),
    schedule_seconds: int = Form(...),
    enabled: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    clean_name = name.strip()
    clean_url = url.strip()
    if not clean_name or not clean_url:
        raise HTTPException(status_code=400, detail="Feed name and URL are required")
    if schedule_seconds < 60:
        raise HTTPException(status_code=400, detail="Schedule must be at least 60 seconds")

    existing = session.query(Feed).filter(Feed.url == clean_url).one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Feed URL already exists")

    feed = Feed(
        name=clean_name,
        url=clean_url,
        schedule_seconds=schedule_seconds,
        enabled=_parse_bool(enabled),
    )
    session.add(feed)
    return _redirect("/admin", "Feed created")


@router.post("/feeds/{feed_id}/update", response_class=RedirectResponse)
def update_feed(
    feed_id: int,
    name: str = Form(...),
    url: str = Form(...),
    schedule_seconds: int = Form(...),
    enabled: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    feed = session.get(Feed, feed_id)
    if feed is None:
        raise HTTPException(status_code=404, detail="Feed not found")
    if schedule_seconds < 60:
        raise HTTPException(status_code=400, detail="Schedule must be at least 60 seconds")

    feed.name = name.strip()
    feed.url = url.strip()
    conflict = (
        session.query(Feed)
        .filter(Feed.url == feed.url, Feed.id != feed_id)
        .one_or_none()
    )
    if conflict:
        raise HTTPException(status_code=400, detail="Another feed already uses this URL")
    feed.schedule_seconds = schedule_seconds
    feed.enabled = _parse_bool(enabled)
    session.add(feed)
    return _redirect("/admin", "Feed updated")


@router.post("/feeds/{feed_id}/delete", response_class=RedirectResponse)
def delete_feed(
    feed_id: int,
    session: Session = Depends(get_session),
):
    feed = session.get(Feed, feed_id)
    if feed is None:
        raise HTTPException(status_code=404, detail="Feed not found")
    session.delete(feed)
    return _redirect("/admin", "Feed removed")


@router.post("/settings/scheduler", response_class=RedirectResponse)
def update_scheduler_interval(
    interval_seconds: int = Form(...),
    session: Session = Depends(get_session),
):
    if interval_seconds < 300:
        raise HTTPException(status_code=400, detail="Summary interval must be >= 300 seconds")
    set_config(session, CONFIG_SUMMARIZATION_INTERVAL, interval_seconds)
    return _redirect("/admin", "Summarisation interval updated")


@router.post("/settings/llm", response_class=RedirectResponse)
def update_llm_settings(
    provider: str = Form(...),
    model: str = Form(...),
    base_url: str = Form(...),
    session: Session = Depends(get_session),
):
    clean_provider = provider.strip()
    clean_model = model.strip()
    clean_base_url = base_url.strip()
    if not clean_provider or not clean_model or not clean_base_url:
        raise HTTPException(
            status_code=400,
            detail="Provider, model, and base URL are required",
        )
    set_config(session, CONFIG_LLM_PROVIDER, clean_provider)
    set_config(session, CONFIG_LLM_MODEL, clean_model)
    set_config(session, CONFIG_LLM_BASE_URL, clean_base_url)
    return _redirect("/admin", "LLM configuration saved")


@router.post("/profile", response_class=RedirectResponse)
def update_profile(
    title: str = Form(...),
    content: str = Form(...),
    session: Session = Depends(get_session),
):
    clean_title = title.strip() or "Audience Profile"
    clean_content = content.strip()
    if len(clean_content) < 40:
        raise HTTPException(status_code=400, detail="Profile content must be at least 40 characters")

    active = (
        session.query(UserProfile)
        .filter(UserProfile.is_active.is_(True))
        .order_by(UserProfile.updated_at.desc())
        .first()
    )
    if active:
        active.title = clean_title
        active.content = clean_content
        profile_id = active.id
    else:
        session.query(UserProfile).update({UserProfile.is_active: False})
        new_profile = UserProfile(title=clean_title, content=clean_content, is_active=True)
        session.add(new_profile)
        session.flush()
        profile_id = new_profile.id
    set_config(session, CONFIG_ACTIVE_PROFILE_ID, profile_id)
    return _redirect("/admin", "Profile updated")


@router.post("/settings/llm/test", response_class=JSONResponse)
def test_llm_configuration(session: Session = Depends(get_session)) -> JSONResponse:
    """Perform a round-trip check against the configured LLM runtime."""

    try:
        settings = load_settings()
        logger.info("Loaded settings")

        provider = get_config(session, CONFIG_LLM_PROVIDER, settings.llm.provider)
        model_name = get_config(session, CONFIG_LLM_MODEL, settings.llm.model)
        if model_name is None:
            model_name = get_config(session, CONFIG_LLM_MODEL_PATH, settings.llm.model)
        base_url = get_config(session, CONFIG_LLM_BASE_URL, settings.llm.base_url)
        logger.info(f"Config: provider={provider}, model={model_name}, base_url={base_url}")

        llm_settings = build_llm_settings(
            settings.llm,
            provider,
            model_name,
            base_url,
        )
        logger.info("Built LLM settings")

        llm_client = LLMClient(llm_settings)
        logger.info("Created LLM client")

        reply = llm_client.ping("Status check: please confirm you are reachable.")
        logger.info("Ping successful")
    except LLMClientError as exc:
        error_msg = str(exc)
        if hasattr(exc, 'debug_info') and exc.debug_info:
            debug_str = json.dumps(exc.debug_info, indent=2)
            error_msg += f"\n\nDebug Info:\n{debug_str}"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ok": False, "error": error_msg},
        )
    except Exception as exc:
        logger.exception("Unexpected error in LLM test")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"ok": False, "error": f"Unexpected error: {exc}"},
        )

    return JSONResponse(content={"ok": True, "response": reply})


__all__ = ["router"]
