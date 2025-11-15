"""Data viewer routes for operational insight."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.db.models import Article, ArticleIngestionLog, Feed, Summary
from src.web.dependencies import get_session, get_templates

router = APIRouter()


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
    }
    return templates.TemplateResponse("data_viewer.html", context)


__all__ = ["router"]
