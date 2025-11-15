"""Prompt for mapping an article to high-level categories."""
from __future__ import annotations

from .base import JsonPromptTemplate


CATEGORY_CLASSIFICATION_PROMPT = JsonPromptTemplate(
    name="article_category_classification",
    system_prompt=(
        "You assign newsroom taxonomy labels."
        " Choose the best matching category and subcategory from the provided options."
        " Answer strictly in JSON."
    ),
    user_template="""
    Allowed categories:
      - Politics
        - Elections
        - Policy
        - Diplomacy
      - Business
        - Markets
        - Companies
        - Economy
      - Technology
        - AI
        - Gadgets
        - Cybersecurity
      - Culture
        - Entertainment
        - Art
        - Lifestyle
      - Science
        - Space
        - Environment
        - Health

    Title: {title}
    Summary: {summary}
    Body:
    {content}

    Return the best category and subcategory.
    """,
    response_schema={
        "type": "object",
        "properties": {
            "category": {"type": ["string", "null"]},
            "subcategory": {"type": ["string", "null"]},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "rationale": {"type": ["string", "null"]},
        },
        "required": ["category", "subcategory"],
    },
)


__all__ = ["CATEGORY_CLASSIFICATION_PROMPT"]
