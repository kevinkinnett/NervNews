"""Data viewer routes for operational insight."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from src.config.settings import FeedSettings, load_settings
from src.config.store import (
    CONFIG_LLM_BASE_URL,
    CONFIG_LLM_MODEL,
    CONFIG_LLM_MODEL_PATH,
    CONFIG_LLM_PROVIDER,
    build_llm_settings,
    get_config,
    load_feed_settings,
)
from src.db.models import Article, ArticleIngestionLog, Feed, Summary, SummaryEvaluation
from src.ingestion.extractor import ArticleExtractor
from src.ingestion.rss import RSSIngestionService
from src.llm import ArticleEnrichmentService, LLMClient
from src.web.dependencies import get_session, get_templates

logger = logging.getLogger(__name__)

router = APIRouter()


def _redirect(path: str, message: Optional[str] = None) -> RedirectResponse:
    url = path
    if message:
        from urllib.parse import urlencode

        query = urlencode({"msg": message})
        url = f"{path}?{query}"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


def _build_manual_ingestion_stack(
    request: Request,
    session: Session,
) -> tuple[List[FeedSettings], RSSIngestionService, ArticleEnrichmentService]:
    settings = load_settings()
    feeds = load_feed_settings(session)
    if not feeds:
        feeds = settings.feeds

    session_factory = request.app.state.session_factory
    extractor = ArticleExtractor(timeout=settings.request_timeout, user_agent=settings.user_agent)
    ingestion_service = RSSIngestionService(session_factory=session_factory, extractor=extractor)

    provider_override = get_config(session, CONFIG_LLM_PROVIDER, settings.llm.provider)
    model_override = get_config(session, CONFIG_LLM_MODEL, settings.llm.model)
    if model_override is None:
        model_override = get_config(session, CONFIG_LLM_MODEL_PATH, settings.llm.model)
    base_url_override = get_config(session, CONFIG_LLM_BASE_URL, settings.llm.base_url)
    llm_settings = build_llm_settings(
        settings.llm,
        provider_override,
        model_override,
        base_url_override,
    )
    llm_client = LLMClient(llm_settings)
    enrichment_service = ArticleEnrichmentService(session_factory=session_factory, llm_client=llm_client)

    return list(feeds), ingestion_service, enrichment_service


def _parse_summary_payload(summary: Optional[Summary]) -> Dict[str, Any]:
    payload = {"headline": "", "summary": ""}
    if summary is None or not summary.final_json:
        return payload
    try:
        data = json.loads(summary.final_json)
    except json.JSONDecodeError:
        return payload
    payload["headline"] = data.get("headline", "")
    payload["summary"] = data.get("summary", "")
    return payload


def _article_count(summary: Summary) -> int:
    try:
        data = json.loads(summary.article_ids_json or "[]")
        if isinstance(data, list):
            return len(data)
    except json.JSONDecodeError:
        return 0
    return 0


@router.get("/data", response_class=HTMLResponse)
def data_overview(
    request: Request,
    session: Session = Depends(get_session),
):
    templates = get_templates(request)

    article_total = session.query(func.count(Article.id)).scalar() or 0
    enriched_total = (
        session.query(func.count(Article.id))
        .filter(Article.enriched_at.isnot(None))
        .scalar()
        or 0
    )
    ingestion_events = session.query(func.count(ArticleIngestionLog.id)).scalar() or 0

    feed_rows = (
        session.query(
            Feed,
            func.count(Article.id).label("article_count"),
            func.max(Article.published_at).label("last_published_at"),
            func.max(Article.fetched_at).label("last_fetched_at"),
        )
        .outerjoin(Article, Feed.id == Article.feed_id)
        .group_by(Feed.id)
        .order_by(Feed.name.asc())
        .all()
    )

    recent_articles: List[Article] = (
        session.query(Article)
        .order_by(Article.fetched_at.desc())
        .limit(25)
        .all()
    )

    recent_summaries: List[Summary] = (
        session.query(Summary)
        .order_by(Summary.created_at.desc())
        .limit(10)
        .all()
    )

    context = {
        "request": request,
        "stats": {
            "article_total": article_total,
            "enriched_total": enriched_total,
            "ingestion_events": ingestion_events,
        },
        "feed_rows": feed_rows,
        "recent_articles": recent_articles,
        "recent_summaries": [
            {
                "summary": summary,
                "article_count": _article_count(summary),
                "payload": _parse_summary_payload(summary),
                "has_evaluation": bool(summary.evaluation),
            }
            for summary in recent_summaries
        ],
        "message": request.query_params.get("msg"),
    }
    return templates.TemplateResponse("data_viewer.html", context)


@router.get("/data/articles/{article_id}", response_class=HTMLResponse)
def article_detail(
    request: Request,
    article_id: int,
    session: Session = Depends(get_session),
):
    templates = get_templates(request)
    article = (
        session.query(Article)
        .options(
            selectinload(Article.feed),
            selectinload(Article.ingestion_logs).selectinload(ArticleIngestionLog.feed),
        )
        .filter(Article.id == article_id)
        .one_or_none()
    )
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")

    ingestion_history = sorted(
        article.ingestion_logs,
        key=lambda entry: entry.processed_at or entry.id,
        reverse=True,
    )

    context = {
        "request": request,
        "article": article,
        "ingestion_history": ingestion_history,
    }
    return templates.TemplateResponse("article_detail.html", context)


@router.post("/data/articles/{article_id}/delete", response_class=RedirectResponse)
def delete_article(
    article_id: int,
    session: Session = Depends(get_session),
):
    article = session.get(Article, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    session.delete(article)
    logger.info("Article %s manually deleted via data viewer", article_id)
    return _redirect("/data", f"Article #{article_id} deleted")


@router.post("/data/actions/delete-all", response_class=RedirectResponse)
def delete_all_data(
    session: Session = Depends(get_session),
):
    # Remove dependent tables explicitly to avoid FK constraints during bulk deletes.
    session.query(SummaryEvaluation).delete()
    session.query(Summary).delete()
    session.query(ArticleIngestionLog).delete()
    session.query(Article).delete()
    logger.warning("All article and summary data purged via data viewer")
    return _redirect("/data", "All article and summary data deleted")


@router.post("/data/actions/reindex", response_class=RedirectResponse)
def reindex_feeds(
    request: Request,
    session: Session = Depends(get_session),
):
    try:
        feeds, ingestion_service, enrichment_service = _build_manual_ingestion_stack(request, session)
    except Exception as exc:  # pragma: no cover - configuration errors
        logger.exception("Failed to prepare manual ingestion stack: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to prepare ingestion services") from exc

    feed_list = feeds
    total_new = 0
    enriched_batches: List[int] = []
    for feed in feed_list:
        try:
            new_ids = ingestion_service.ingest(feed)
            total_new += len(new_ids)
            if new_ids:
                enrichment_service.enrich_articles(new_ids)
                enriched_batches.append(len(new_ids))
        except Exception as exc:  # pragma: no cover - runtime ingestion failure
            logger.exception("Manual ingestion for feed %s failed: %s", feed.name, exc)
            raise HTTPException(status_code=500, detail=f"Failed to ingest feed {feed.name}") from exc

    message = "Manual re-index completed"
    if total_new:
        message = f"Manual re-index created {total_new} new article(s)"
    logger.info(
        "Manual re-index executed for %d feed(s), new_articles=%d, enriched_batches=%s",
        len(feed_list),
        total_new,
        enriched_batches,
    )
    return _redirect("/data", message)


__all__ = ["router"]
