"""Prompt templates used for summarisation workflows."""
from __future__ import annotations

from textwrap import dedent

from .base import JsonPromptTemplate

ARTICLE_BRIEF_PROMPT = JsonPromptTemplate(
    name="article_brief",
    system_prompt=dedent(
        """
        You are an assistant that writes concise, news-style capsules about a single
        article. Capture the essential facts in at most three sentences while keeping
        neutral tone and avoiding speculation.
        """
    ).strip(),
    user_template=
    """
    Title: {title}
    Summary: {summary}
    Content: {content}

    Produce a 2-3 sentence brief (<= 70 words) that highlights the who, what, when,
    where, and why if available.
    """,
    response_schema={
        "type": "object",
        "properties": {
            "brief": {"type": "string"},
        },
        "required": ["brief"],
    },
)

REPORTER_SUMMARY_PROMPT = JsonPromptTemplate(
    name="reporter_hourly_summary",
    system_prompt=dedent(
        """
        You are "Alex Chen", a seasoned newsroom reporter. Compile a cohesive hourly
        desk update that prioritises clarity, accuracy, and actionable insights for
        editors. Write with calm authority and keep the tone factual yet vivid.
        """
    ).strip(),
    user_template=
    """
    Time window: {time_window}

    Recent coverage:
    {recent_context}

    Historical background:
    {historical_context}

    Editor feedback to address (if any):
    {feedback}

    Craft an hourly update that includes a sharp headline, a tight summary paragraph,
    and a list of 3-6 bullet points covering key developments, context, and any
    outstanding questions. Stay within newsroom voice guidelines and avoid repetition.
    """,
    response_schema={
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["headline", "summary", "key_points"],
    },
)

CRITIC_REVIEW_PROMPT = JsonPromptTemplate(
    name="editorial_critic_review",
    system_prompt=dedent(
        """
        You are a meticulous newsroom editor. Review reporter drafts, flag factual or
        structural issues, and suggest precise improvements. Focus on accuracy,
        completeness, and narrative flow.
        """
    ).strip(),
    user_template=
    """
    Time window: {time_window}

    Draft under review:
    Headline: {draft_headline}
    Summary: {draft_summary}
    Key points:
    {draft_key_points}

    Context supplied to the reporter:
    {context_digest}

    Provide structured feedback. Mark should_revise true only if revisions are
    necessary. Offer concise strengths, issues, and specific revision guidance.
    """,
    response_schema={
        "type": "object",
        "properties": {
            "should_revise": {"type": "boolean"},
            "strengths": {"type": "string"},
            "issues": {"type": "string"},
            "revision_guidance": {"type": "string"},
        },
        "required": [
            "should_revise",
            "strengths",
            "issues",
            "revision_guidance",
        ],
    },
)

__all__ = [
    "ARTICLE_BRIEF_PROMPT",
    "REPORTER_SUMMARY_PROMPT",
    "CRITIC_REVIEW_PROMPT",
]
