"""Application settings management for NervNews."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
ENV_SETTINGS_PATH = "NERVNEWS_SETTINGS"


@dataclass
class FeedSettings:
    """Configuration for a single RSS/Atom feed."""

    name: str
    url: str
    schedule_seconds: int = 900
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppSettings:
    """Top-level application settings loaded from YAML."""

    database_url: str = "sqlite:///nervnews.db"
    feeds: List[FeedSettings] = field(default_factory=list)
    request_timeout: int = 10
    user_agent: str = "NervNewsBot/0.1"


class SettingsError(RuntimeError):
    """Raised when there is an issue loading settings."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SettingsError(f"Settings file not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SettingsError("Settings file must define a mapping at the root level")
    return data


def _parse_feed(entry: Dict[str, Any]) -> FeedSettings:
    if "name" not in entry or "url" not in entry:
        raise SettingsError("Each feed entry must define at least 'name' and 'url'")
    schedule = int(entry.get("schedule_seconds") or entry.get("schedule") or 900)
    enabled = bool(entry.get("enabled", True))
    metadata = dict(entry.get("metadata", {}))
    return FeedSettings(
        name=str(entry["name"]),
        url=str(entry["url"]),
        schedule_seconds=schedule,
        enabled=enabled,
        metadata=metadata,
    )


def load_settings(path: Optional[Path] = None) -> AppSettings:
    """Load application settings from YAML into ``AppSettings``.

    ``path`` defaults to the value of the ``NERVNEWS_SETTINGS`` environment variable
    and falls back to ``config/settings.yaml`` relative to the project root.
    """

    if path is None:
        env_path = os.environ.get(ENV_SETTINGS_PATH)
        if env_path:
            path = Path(env_path)
        else:
            path = DEFAULT_SETTINGS_PATH

    data = _load_yaml(path)

    feeds_raw = data.get("feeds", [])
    if not isinstance(feeds_raw, list):
        raise SettingsError("'feeds' must be a list of feed definitions")

    feeds = [_parse_feed(item) for item in feeds_raw]

    database_url = str(data.get("database_url", "sqlite:///nervnews.db"))
    request_timeout = int(data.get("request_timeout", 10))
    user_agent = str(data.get("user_agent", "NervNewsBot/0.1"))

    return AppSettings(
        database_url=database_url,
        feeds=feeds,
        request_timeout=request_timeout,
        user_agent=user_agent,
    )


__all__ = ["AppSettings", "FeedSettings", "SettingsError", "load_settings"]
