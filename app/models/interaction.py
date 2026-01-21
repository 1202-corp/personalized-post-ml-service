"""Interaction ORM model."""
from datetime import datetime
from typing import List
from sqlalchemy import Boolean, ForeignKey, DateTime, Enum, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import enum


class InteractionType(str, enum.Enum):
    """Type of user interaction with post."""
    LIKE = "like"
    DISLIKE = "dislike"
    SKIP = "skip"


class Interaction(Base):
    """User interaction with post."""
    __tablename__ = "interactions"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), nullable=False)
    
    interaction_type: Mapped[InteractionType] = mapped_column(
        Enum(InteractionType),
        nullable=False
    )
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="interactions")
    post: Mapped["Post"] = relationship(back_populates="interactions")
    
    __table_args__ = (
        Index("idx_interaction_user_post", "user_id", "post_id", unique=True),
    )

