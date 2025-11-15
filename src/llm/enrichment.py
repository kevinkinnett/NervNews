"""Sequential enrichment pipeline that annotates newly ingested articles."""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Dict, Sequence

from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Article
from src.db.session import session_scope
from src.llm.client import LLMClient, LLMClientError
from src.llm.prompts import (
    ARTICLE_BRIEF_PROMPT,
    CATEGORY_CLASSIFICATION_PROMPT,
    JsonPromptTemplate,
    LOCATION_EXTRACTION_PROMPT,
    TOPIC_IDENTIFICATION_PROMPT,
)
from src.telemetry import metrics

logger = logging.getLogger(__name__)


class ArticleEnrichmentService:
    """Run prompt-based enrichment for a batch of article identifiers."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        llm_client: LLMClient,
    ) -> None:
        self._session_factory = session_factory
        self._llm_client = llm_client

    def enrich_articles(self, article_ids: Sequence[int]) -> None:
        ids = list(article_ids)
        if not ids:
            return

        batch_start = time.perf_counter()
        successes = 0
        failures = 0
        attempted = 0
        with session_scope(self._session_factory) as session:
            articles = (
                session.query(Article)
                .filter(Article.id.in_(ids))
                .all()
            )
            for article in articles:
                if article.enriched_at:
                    logger.debug("Skipping article %s already enriched", article.id)
                    continue
                attempted += 1
                try:
                    self._enrich_single(session, article)
                    successes += 1
                except Exception as exc:
                    logger.exception("Failed to enrich article %s: %s", article.id, exc)
                    failures += 1

        duration = time.perf_counter() - batch_start
        if attempted:
            metrics.record_enrichment_batch(
                attempted=attempted,
                successes=successes,
                failures=failures,
                duration_seconds=duration,
            )

        if successes:
            avg = duration / max(successes, 1)
            logger.info(
                "Enriched %d/%d articles in %.2fs (avg %.2fs per article, failures=%d)",
                successes,
                len(ids),
                duration,
                avg,
                failures,
                extra={
                    "event": "enrichment.batch_completed",
                    "attempted": attempted,
                    "successes": successes,
                    "failures": failures,
                    "duration_seconds": duration,
                },
            )
        else:
            logger.warning(
                "No articles were successfully enriched for ids: %s",
                ids,
                extra={
                    "event": "enrichment.batch_empty",
                    "attempted": attempted,
                    "failures": failures,
                    "duration_seconds": duration,
                },
            )

    def _collect_variables(self, article: Article) -> Dict[str, str | None]:
        return {
            "title": article.title,
            "summary": article.summary,
            "content": article.content,
        }

    def _enrich_single(self, session: Session, article: Article) -> None:
        timer_start = time.perf_counter()
        variables = self._collect_variables(article)

        if not any(variables.values()):
            raise ValueError("Article has no textual content to enrich")

        location = self._safe_invoke(LOCATION_EXTRACTION_PROMPT, variables)
        topic = self._safe_invoke(TOPIC_IDENTIFICATION_PROMPT, variables)
        classification = self._safe_invoke(CATEGORY_CLASSIFICATION_PROMPT, variables)
        brief = self._safe_invoke(ARTICLE_BRIEF_PROMPT, variables)

        article.location_name = location.get("location_name")
        article.location_country = location.get("country")
        article.location_confidence = self._to_float(location.get("confidence"))
        article.location_justification = location.get("justification")

        article.topic = topic.get("topic")
        article.topic_confidence = self._to_float(topic.get("confidence"))
        article.topic_supporting_points = topic.get("supporting_points")

        article.category = classification.get("category")
        article.subcategory = classification.get("subcategory")
        article.category_confidence = self._to_float(classification.get("confidence"))
        article.category_rationale = classification.get("rationale")

        article.brief_summary = brief.get("brief")

        article.enriched_at = datetime.utcnow()

        duration = time.perf_counter() - timer_start
        logger.info(
            "Article %s enriched in %.2fs (topic=%s[%.2f], category=%s/%s[%.2f], location_conf=%.2f)",
            article.id,
            duration,
            article.topic,
            article.topic_confidence or 0.0,
            article.category,
            article.subcategory,
            article.category_confidence or 0.0,
            article.location_confidence or 0.0,
        )
        session.add(article)

    def _safe_invoke(
        self,
        prompt: JsonPromptTemplate,
        variables: Dict[str, str | None],
    ) -> Dict[str, object]:
        prepared_variables, _ = self._fit_prompt_to_context(prompt, variables)
        try:
            return self._llm_client.generate_structured(prompt, prepared_variables)
        except LLMClientError as exc:
            logger.warning("Primary enrichment call failed: %s", exc)
            fallback_variables = dict(variables)
            fallback_variables["content"] = variables.get("summary") or variables.get("title")
            safe_fallback, _ = self._fit_prompt_to_context(prompt, fallback_variables)
            return self._llm_client.generate_structured(
                prompt,
                safe_fallback,
                max_retries=2,
            )

    def _fit_prompt_to_context(
        self,
        prompt: JsonPromptTemplate,
        variables: Dict[str, str | None],
    ) -> tuple[Dict[str, str | None], str | None]:
        """Ensure the prompt inputs stay within the model context window."""

        approx_chars_per_token = 4
        context_limit = max(self._llm_client.settings.context_window, 0)
        prepared: Dict[str, str | None] = dict(variables)

        if context_limit <= 0:
            return prepared, None

        keys = set(prepared.keys()) | {"content", "summary", "title"}

        def _prompt_values(data: Dict[str, str | None]) -> Dict[str, str]:
            values: Dict[str, str] = {}
            for key in keys:
                raw = data.get(key)
                if raw is None:
                    values[key] = ""
                elif isinstance(raw, str):
                    values[key] = raw
                else:
                    values[key] = str(raw)
            return values

        def _measure(data: Dict[str, str | None]) -> tuple[int, int]:
            prompt_vars = _prompt_values(data)
            rendered = prompt.render_user_prompt(**prompt_vars)
            total_chars = len(prompt.system_prompt or "") + len(rendered)
            if total_chars <= 0:
                return 0, 0
            estimated_tokens = max(1, math.ceil(total_chars / approx_chars_per_token))
            return total_chars, estimated_tokens

        total_chars, estimated_tokens = _measure(prepared)
        if estimated_tokens <= context_limit:
            return prepared, None

        log_extra = {
            "event": "enrichment.prompt.compaction",
            "template": prompt.name,
            "prompt_chars": total_chars,
            "prompt_tokens": estimated_tokens,
            "context_window": context_limit,
        }

        summary_text = prepared.get("summary") or ""
        content_text = prepared.get("content") or ""
        title_text = prepared.get("title") or ""

        if summary_text:
            prepared["content"] = summary_text
            total_chars, estimated_tokens = _measure(prepared)
            if estimated_tokens <= context_limit:
                logger.warning(
                    "Using article summary for %s due to oversized prompt (chars=%d, tokens~%d, limit=%d)",
                    prompt.name,
                    total_chars,
                    estimated_tokens,
                    context_limit,
                    extra=log_extra,
                )
                prepared.setdefault("summary", summary_text)
                return prepared, "summary"
            prepared["content"] = content_text

        prepared["content"] = ""
        static_chars, static_tokens = _measure(prepared)
        prepared["content"] = content_text

        if static_tokens > context_limit:
            logger.warning(
                "Prompt for %s exceeds context window even without article content (chars=%d, tokens~%d, limit=%d)",
                prompt.name,
                static_chars,
                static_tokens,
                context_limit,
                extra=log_extra,
            )
            prepared["content"] = ""
            return prepared, "empty"

        max_chars = context_limit * approx_chars_per_token
        allowed_chars = max(0, max_chars - static_chars)

        source_text = content_text or summary_text or title_text
        truncated_text = (source_text or "")[:allowed_chars]
        prepared["content"] = truncated_text
        if not summary_text:
            prepared.setdefault("summary", truncated_text)
        keys.update(prepared.keys())

        total_chars, estimated_tokens = _measure(prepared)
        logger.warning(
            "Truncated article content for %s to %d chars (~%d tokens) to fit context window %d",
            prompt.name,
            len(truncated_text),
            estimated_tokens,
            context_limit,
            extra=log_extra,
        )
        return prepared, "truncated"

    @staticmethod
    def _to_float(value) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


__all__ = ["ArticleEnrichmentService"]
