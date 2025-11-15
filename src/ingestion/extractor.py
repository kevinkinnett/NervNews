"""Article extraction utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests
from readability import Document
from lxml import html

logger = logging.getLogger(__name__)


@dataclass
class ExtractedArticle:
    """Normalized representation of scraped article content."""

    title: Optional[str]
    text: Optional[str]
    summary: Optional[str]


class ArticleExtractor:
    """Extract and normalize article content from raw HTML pages."""

    def __init__(self, timeout: int = 10, user_agent: str = "NervNewsBot/0.1") -> None:
        self._timeout = timeout
        self._headers = {"User-Agent": user_agent}

    def extract(self, url: str) -> ExtractedArticle:
        """Fetch and extract article content from the provided URL."""
        try:
            response = requests.get(url, timeout=self._timeout, headers=self._headers)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network error path
            logger.warning("Failed to download article %s: %s", url, exc)
            raise

        doc = Document(response.text)
        title = doc.short_title()
        summary_html = doc.summary(html_partial=True)

        try:
            tree = html.fromstring(summary_html)
            text = tree.text_content().strip()
        except Exception as exc:  # pragma: no cover - lxml parsing path
            logger.debug("Failed to parse cleaned HTML for %s: %s", url, exc)
            text = None

        return ExtractedArticle(title=title, text=text, summary=summary_html)


__all__ = ["ArticleExtractor", "ExtractedArticle"]
