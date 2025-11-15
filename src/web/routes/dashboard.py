"""Dashboard views for summarised intelligence."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from src.db.models import Summary
from src.web.dependencies import get_session, get_templates

router = APIRouter()


def _parse_summary_payload(summary: Optional[Summary]) -> Dict[str, Any]:
    payload = {"headline": "", "summary": "", "key_points": []}
    if summary is None:
        return payload
    if summary.final_json:
        try:
            data = json.loads(summary.final_json)
            payload.update({
                "headline": data.get("headline", ""),
                "summary": data.get("summary", ""),
                "key_points": data.get("key_points", []) or [],
            })
        except json.JSONDecodeError:
            pass
    return payload


def _parse_evaluation(summary: Optional[Summary]) -> Optional[Dict[str, Any]]:
    if summary is None:
        return None
    if not summary.evaluation or not summary.evaluation.ratings_json:
        return None
    try:
        return json.loads(summary.evaluation.ratings_json)
    except json.JSONDecodeError:
        return None


def _article_count(summary: Summary) -> int:
    try:
        data = json.loads(summary.article_ids_json or "[]")
        if isinstance(data, list):
            return len(data)
    except json.JSONDecodeError:
        return 0
    return 0


@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    session: Session = Depends(get_session),
):
    templates = get_templates(request)
    latest: Optional[Summary] = (
        session.query(Summary)
        .order_by(Summary.created_at.desc())
        .first()
    )
    history: List[Summary] = (
        session.query(Summary)
        .order_by(Summary.created_at.desc())
        .limit(6)
        .all()
    )
    history_view = [
        {
            "summary": item,
            "article_count": _article_count(item),
        }
        for item in history
    ]

    context = {
        "request": request,
        "latest": latest,
        "latest_payload": _parse_summary_payload(latest) if latest else None,
        "latest_evaluation": _parse_evaluation(latest) if latest else None,
        "history": history_view,
    }
    return templates.TemplateResponse("dashboard.html", context)


@router.get("/summaries/{summary_id}", response_class=HTMLResponse)
def summary_detail(
    request: Request,
    summary_id: int,
    session: Session = Depends(get_session),
):
    templates = get_templates(request)
    summary = session.get(Summary, summary_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Summary not found")

    context = {
        "request": request,
        "summary": summary,
        "payload": _parse_summary_payload(summary),
        "evaluation": _parse_evaluation(summary),
    }
    return templates.TemplateResponse("summary_detail.html", context)


__all__ = ["router"]
