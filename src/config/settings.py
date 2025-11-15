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

    id: Optional[int] = None
    name: str
    url: str
    schedule_seconds: int = 900
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMSettings:
    """Configuration for the local LLM backend used for enrichment."""

    provider: str = "llama_cpp"
    model_path: str = "models/model.gguf"
    quantization: Optional[str] = None
    context_window: int = 4096
    max_output_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    threads: Optional[int] = None
    max_retries: int = 3


@dataclass
class SummarizationSettings:
    """Configuration for reporter-style summarisation cycles."""

    interval_seconds: int = 3600
    context_window_chars: int = 6000
    max_iterations: int = 3
    historical_days: int = 7
    max_recent_articles: int = 15
    max_historical_per_topic: int = 3


@dataclass
class UserProfileSettings:
    """Default newsroom user profile description."""

    title: str = "Default Audience"
    content: str = ""
    is_active: bool = True


@dataclass
class AppSettings:
    """Top-level application settings loaded from YAML."""

    database_url: str = "sqlite:///nervnews.db"
    feeds: List[FeedSettings] = field(default_factory=list)
    request_timeout: int = 10
    user_agent: str = "NervNewsBot/0.1"
    llm: LLMSettings = field(default_factory=LLMSettings)
    summarization: SummarizationSettings = field(default_factory=SummarizationSettings)
    user_profile: Optional[UserProfileSettings] = None


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
        id=entry.get("id"),
        name=str(entry["name"]),
        url=str(entry["url"]),
        schedule_seconds=schedule,
        enabled=enabled,
        metadata=metadata,
    )


def _parse_llm(entry: Dict[str, Any]) -> LLMSettings:
    if not isinstance(entry, dict):
        raise SettingsError("'llm' must be a mapping of configuration values")

    threads_value = entry.get("threads")
    threads = None if threads_value in (None, "") else int(threads_value)

    return LLMSettings(
        provider=str(entry.get("provider", "llama_cpp")),
        model_path=str(entry.get("model_path", "models/model.gguf")),
        quantization=entry.get("quantization"),
        context_window=int(entry.get("context_window", 4096)),
        max_output_tokens=int(entry.get("max_output_tokens", 256)),
        temperature=float(entry.get("temperature", 0.2)),
        top_p=float(entry.get("top_p", 0.95)),
        repeat_penalty=float(entry.get("repeat_penalty", 1.1)),
        threads=threads,
        max_retries=int(entry.get("max_retries", 3)),
    )


def _parse_summarization(entry: Dict[str, Any]) -> SummarizationSettings:
    if not isinstance(entry, dict):
        raise SettingsError("'summarization' must be a mapping of configuration values")

    return SummarizationSettings(
        interval_seconds=int(entry.get("interval_seconds", 3600)),
        context_window_chars=int(entry.get("context_window_chars", 6000)),
        max_iterations=max(1, int(entry.get("max_iterations", 3))),
        historical_days=max(0, int(entry.get("historical_days", 7))),
        max_recent_articles=max(1, int(entry.get("max_recent_articles", 15))),
        max_historical_per_topic=max(1, int(entry.get("max_historical_per_topic", 3))),
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

    llm_raw = data.get("llm", {})
    llm_settings = _parse_llm(llm_raw) if llm_raw else LLMSettings()

    summarization_raw = data.get("summarization", {})
    summarization_settings = (
        _parse_summarization(summarization_raw)
        if summarization_raw
        else SummarizationSettings()
    )

    profile_entry = data.get("user_profile")
    profile_settings: Optional[UserProfileSettings] = None
    if isinstance(profile_entry, dict) and profile_entry.get("content"):
        profile_settings = UserProfileSettings(
            title=str(profile_entry.get("title", "Default Audience")),
            content=str(profile_entry.get("content", "")),
            is_active=bool(profile_entry.get("is_active", True)),
        )

    return AppSettings(
        database_url=database_url,
        feeds=feeds,
        request_timeout=request_timeout,
        user_agent=user_agent,
        llm=llm_settings,
        summarization=summarization_settings,
        user_profile=profile_settings,
    )


__all__ = [
    "AppSettings",
    "FeedSettings",
    "LLMSettings",
    "SummarizationSettings",
    "UserProfileSettings",
    "SettingsError",
    "load_settings",
]
