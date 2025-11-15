"""Prompt for identifying the main topic of an article."""
from __future__ import annotations

from .base import JsonPromptTemplate


TOPIC_IDENTIFICATION_PROMPT = JsonPromptTemplate(
    name="article_topic_identification",
    system_prompt=(
        "You are a news analyst who summarises the central topic of an article."
        " Return structured JSON with the requested fields."
    ),
    user_template="""
    Title: {title}
    Summary: {summary}
    Body:
    {content}

    State the primary topic in under ten words.
    """,
    response_schema={
        "type": "object",
        "properties": {
            "topic": {"type": ["string", "null"]},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "supporting_points": {"type": ["string", "null"]},
        },
        "required": ["topic", "confidence"],
    },
)


__all__ = ["TOPIC_IDENTIFICATION_PROMPT"]
