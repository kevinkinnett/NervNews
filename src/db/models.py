"""SQLAlchemy models for NervNews storage."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
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
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

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


__all__ = ["Feed", "Article", "ArticleIngestionLog"]
