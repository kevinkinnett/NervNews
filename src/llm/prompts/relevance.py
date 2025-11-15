"""Prompt template that rates summary key points for relevance."""
from __future__ import annotations

from textwrap import dedent

from .base import JsonPromptTemplate

SUMMARY_RELEVANCE_PROMPT = JsonPromptTemplate(
    name="summary_relevance_rating",
    system_prompt=dedent(
        """
        You are an intelligence editor tasked with evaluating newsroom summaries
        for a specific audience profile. Rate each key point on relevance to the
        profile and operational criticality. Use calibrated language and justify
        the ratings with concise reasoning.
        """
    ).strip(),
    user_template=dedent(
        """
        Audience profile:
        {profile}

        Summary headline: {headline}
        Summary paragraph: {summary}

        Key points:
        {key_points}

        Produce JSON containing:
        - overall_relevance: {"score": 0-5 integer, "label": string, "explanation": string}
        - overall_criticality: same schema as overall_relevance
        - items: list matching key_points order with objects containing:
            * key_point: string (copied)
            * relevance: {"score": 0-5 integer, "label": string}
            * criticality: {"score": 0-5 integer, "label": string}
            * explanation: string giving 1-2 sentence rationale
            * escalation: string with guidance ("monitor", "escalate", or "inform")
        Use "High", "Medium", or "Low" as labels. Default to score 0 and label "Low"
        when information is insufficient. Ensure list lengths match.
        """
    ).strip(),
    response_schema={
        "type": "object",
        "properties": {
            "overall_relevance": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "label": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["score", "label", "explanation"],
            },
            "overall_criticality": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "label": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["score", "label", "explanation"],
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key_point": {"type": "string"},
                        "relevance": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer"},
                                "label": {"type": "string"},
                            },
                            "required": ["score", "label"],
                        },
                        "criticality": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer"},
                                "label": {"type": "string"},
                            },
                            "required": ["score", "label"],
                        },
                        "explanation": {"type": "string"},
                        "escalation": {"type": "string"},
                    },
                    "required": [
                        "key_point",
                        "relevance",
                        "criticality",
                        "explanation",
                        "escalation",
                    ],
                },
            },
        },
        "required": ["overall_relevance", "overall_criticality", "items"],
    },
)

__all__ = ["SUMMARY_RELEVANCE_PROMPT"]
