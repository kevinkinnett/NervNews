"""Prompt for extracting the dominant geographic focus of an article."""
from __future__ import annotations

from .base import JsonPromptTemplate


LOCATION_EXTRACTION_PROMPT = JsonPromptTemplate(
    name="article_location_extraction",
    system_prompt=(
        "You are a geoparsing expert."
        " Given article text, identify the single most relevant location discussed."
        " Respond with compact JSON following the provided schema."
        " If no location is evident, return nulls while keeping the JSON structure."
    ),
    user_template="""
    Title: {title}
    Summary: {summary}
    Body:
    {content}

    Provide the dominant location focus for this article.
    """,
    response_schema={
        "type": "object",
        "properties": {
            "location_name": {"type": ["string", "null"]},
            "country": {"type": ["string", "null"]},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "justification": {"type": ["string", "null"]},
        },
        "required": ["location_name", "confidence"],
    },
)


__all__ = ["LOCATION_EXTRACTION_PROMPT"]
