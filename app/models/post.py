"""Post ORM model."""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import BigInteger, Text, String, Boolean, ForeignKey, DateTime, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Post(Base):
    """Channel post model."""
    __tablename__ = "posts"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    telegram_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    media_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    media_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # ML score (0.0 - 1.0) - relevance score for recommendations
    relevance_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    
    # Clustering field - posts are grouped into clusters for optimized search
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    
    posted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    
    # Soft delete fields
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    channel: Mapped["Channel"] = relationship(back_populates="posts")
    interactions: Mapped[List["Interaction"]] = relationship(back_populates="post", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_post_channel_message", "channel_id", "telegram_message_id", unique=True),
        Index("idx_post_relevance", "relevance_score"),
        Index("idx_post_is_deleted", "is_deleted"),
        Index("idx_post_cluster_id", "cluster_id"),
    )

