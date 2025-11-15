"""Structured logging configuration utilities."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

_LOGGING_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class StructuredFormatter(logging.Formatter):
    """Render log records as JSON for easier ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _LOGGING_FIELDS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Configure application logging with optional structured output."""

    level_name = os.getenv("NERVNEWS_LOG_LEVEL", "INFO").upper()
    try:
        level = getattr(logging, level_name)
    except AttributeError:
        level = logging.INFO

    fmt = os.getenv("NERVNEWS_LOG_FORMAT", "plain").lower()
    structured = fmt in {"json", "structured"}

    handler = logging.StreamHandler()
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]
    root_logger.propagate = False


__all__ = ["configure_logging", "StructuredFormatter"]
