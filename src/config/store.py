"""Helpers for persisting runtime configuration overrides."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, List, Optional

from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import AppSettings, FeedSettings, LLMSettings, SummarizationSettings
from src.db.models import AppConfig, Feed, UserProfile
from src.db.session import session_scope

CONFIG_SUMMARIZATION_INTERVAL = "summarization.interval_seconds"
CONFIG_LLM_PROVIDER = "llm.provider"
CONFIG_LLM_MODEL = "llm.model"
CONFIG_LLM_BASE_URL = "llm.base_url"
CONFIG_LLM_MODEL_PATH = "llm.model_path"  # legacy key retained for migrations
CONFIG_ACTIVE_PROFILE_ID = "user_profile.active_id"


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def get_config(session: Session, key: str, default: Any = None) -> Any:
    record = session.query(AppConfig).filter(AppConfig.key == key).one_or_none()
    if not record or record.value_json is None:
        return default
    try:
        return json.loads(record.value_json)
    except json.JSONDecodeError:
        return default


def set_config(session: Session, key: str, value: Any) -> None:
    record = session.query(AppConfig).filter(AppConfig.key == key).one_or_none()
    payload = _to_json(value)
    if record is None:
        record = AppConfig(key=key, value_json=payload)
        session.add(record)
    else:
        record.value_json = payload
    session.add(record)


def ensure_seed_data(session_factory: sessionmaker[Session], settings: AppSettings) -> None:
    """Populate database tables with defaults from YAML configuration."""

    with session_scope(session_factory) as session:
        # Seed feeds if none exist.
        if session.query(Feed).count() == 0 and settings.feeds:
            for feed_cfg in settings.feeds:
                feed = Feed(
                    name=feed_cfg.name,
                    url=feed_cfg.url,
                    schedule_seconds=feed_cfg.schedule_seconds,
                    enabled=feed_cfg.enabled,
                    metadata_json=_to_json(feed_cfg.metadata) if feed_cfg.metadata else None,
                )
                session.add(feed)

        # Seed user profile if absent.
        if session.query(UserProfile).count() == 0 and settings.user_profile and settings.user_profile.content:
            profile = UserProfile(
                title=settings.user_profile.title,
                content=settings.user_profile.content,
                is_active=settings.user_profile.is_active,
            )
            session.add(profile)

        # Seed summarisation interval override to align with base settings.
        if session.query(AppConfig).filter(AppConfig.key == CONFIG_SUMMARIZATION_INTERVAL).count() == 0:
            set_config(session, CONFIG_SUMMARIZATION_INTERVAL, settings.summarization.interval_seconds)

        if session.query(AppConfig).filter(AppConfig.key == CONFIG_LLM_PROVIDER).count() == 0:
            set_config(session, CONFIG_LLM_PROVIDER, settings.llm.provider)

        if session.query(AppConfig).filter(AppConfig.key == CONFIG_LLM_MODEL).count() == 0:
            set_config(session, CONFIG_LLM_MODEL, settings.llm.model)

        if session.query(AppConfig).filter(AppConfig.key == CONFIG_LLM_BASE_URL).count() == 0:
            set_config(session, CONFIG_LLM_BASE_URL, settings.llm.base_url)

        if session.query(AppConfig).filter(AppConfig.key == CONFIG_ACTIVE_PROFILE_ID).count() == 0:
            active_profile = (
                session.query(UserProfile)
                .filter(UserProfile.is_active.is_(True))
                .order_by(UserProfile.updated_at.desc())
                .first()
            )
            if active_profile:
                set_config(session, CONFIG_ACTIVE_PROFILE_ID, active_profile.id)


def load_feed_settings(session: Session) -> List[FeedSettings]:
    feeds = session.query(Feed).order_by(Feed.name.asc()).all()
    results: List[FeedSettings] = []
    for feed in feeds:
        metadata = {}
        if feed.metadata_json:
            try:
                metadata = json.loads(feed.metadata_json)
            except json.JSONDecodeError:
                metadata = {}
        results.append(
            FeedSettings(
                id=feed.id,
                name=feed.name,
                url=feed.url,
                schedule_seconds=feed.schedule_seconds,
                enabled=feed.enabled,
                metadata=metadata,
            )
        )
    return results


def build_llm_settings(
    base: LLMSettings,
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
) -> LLMSettings:
    updated = base
    if provider:
        updated = replace(updated, provider=str(provider))
    if model:
        updated = replace(updated, model=str(model))
    if base_url:
        updated = replace(updated, base_url=str(base_url))
    return updated


def build_summarization_settings(base: SummarizationSettings, interval_seconds: Optional[int]) -> SummarizationSettings:
    if not interval_seconds:
        return base
    return replace(base, interval_seconds=int(interval_seconds))


__all__ = [
    "CONFIG_SUMMARIZATION_INTERVAL",
    "CONFIG_LLM_PROVIDER",
    "CONFIG_LLM_MODEL_PATH",
    "CONFIG_LLM_MODEL",
    "CONFIG_LLM_BASE_URL",
    "CONFIG_ACTIVE_PROFILE_ID",
    "ensure_seed_data",
    "load_feed_settings",
    "get_config",
    "set_config",
    "build_llm_settings",
    "build_summarization_settings",
]
