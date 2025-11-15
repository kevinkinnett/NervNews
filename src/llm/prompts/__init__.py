"""Prompt templates for article enrichment tasks."""
from .base import JsonPromptTemplate
from .classification import CATEGORY_CLASSIFICATION_PROMPT
from .location import LOCATION_EXTRACTION_PROMPT
from .topic import TOPIC_IDENTIFICATION_PROMPT

__all__ = [
    "JsonPromptTemplate",
    "CATEGORY_CLASSIFICATION_PROMPT",
    "LOCATION_EXTRACTION_PROMPT",
    "TOPIC_IDENTIFICATION_PROMPT",
]
