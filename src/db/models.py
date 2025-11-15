"""SQLAlchemy models for NervNews storage."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship

from .base import Base


class Feed(Base):
    __tablename__ = "feeds"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    url = Column(String(1024), nullable=False, unique=True)
    schedule_seconds = Column(Integer, nullable=False, default=900)
    enabled = Column(Boolean, nullable=False, default=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_polled_at = Column(DateTime, nullable=True)

    articles = relationship("Article", back_populates="feed", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Feed name={self.name!r} url={self.url!r}>"


class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (
        UniqueConstraint("feed_id", "guid", name="uq_article_feed_guid"),
        UniqueConstraint("feed_id", "url", name="uq_article_feed_url"),
        Index("ix_articles_published_at", "published_at"),
    )

    id = Column(Integer, primary_key=True)
    feed_id = Column(Integer, ForeignKey("feeds.id", ondelete="CASCADE"), nullable=False)
    guid = Column(String(1024), nullable=True)
    url = Column(String(2048), nullable=False)
    title = Column(String(512), nullable=True)
    summary = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    brief_summary = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    enriched_at = Column(DateTime, nullable=True)

    location_name = Column(String(255), nullable=True)
    location_country = Column(String(255), nullable=True)
    location_confidence = Column(Float, nullable=True)
    location_justification = Column(Text, nullable=True)

    topic = Column(String(255), nullable=True)
    topic_confidence = Column(Float, nullable=True)
    topic_supporting_points = Column(Text, nullable=True)

    category = Column(String(255), nullable=True)
    subcategory = Column(String(255), nullable=True)
    category_confidence = Column(Float, nullable=True)
    category_rationale = Column(Text, nullable=True)

    feed = relationship("Feed", back_populates="articles")
    ingestion_logs = relationship(
        "ArticleIngestionLog",
        back_populates="article",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Article id={self.id} url={self.url!r}>"


class ArticleIngestionLog(Base):
    __tablename__ = "article_ingestion_logs"
    __table_args__ = (
        Index("ix_ingestion_logs_article_id", "article_id"),
        Index("ix_ingestion_logs_processed_at", "processed_at"),
    )

    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    feed_id = Column(Integer, ForeignKey("feeds.id", ondelete="CASCADE"), nullable=False)
    processed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    article = relationship("Article", back_populates="ingestion_logs")
    feed = relationship("Feed")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ArticleIngestionLog article_id={self.article_id}>"


class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    article_ids_json = Column(Text, nullable=False)
    draft_json = Column(Text, nullable=True)
    final_json = Column(Text, nullable=True)
    critic_feedback_json = Column(Text, nullable=True)
    iteration_count = Column(Integer, nullable=False, default=0)
    status = Column(String(50), nullable=False, default="draft")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Summary id={self.id} window=({self.window_start}, {self.window_end}) status={self.status}>"


__all__ = ["Feed", "Article", "ArticleIngestionLog", "Summary"]
