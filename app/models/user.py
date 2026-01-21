"""User ORM model."""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, BigInteger, Boolean, DateTime, Enum, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import enum


class UserStatus(str, enum.Enum):
    """User funnel status."""
    NEW = "new"
    ONBOARDING = "onboarding"
    TRAINING = "training"
    TRAINED = "trained"
    ACTIVE = "active"
    CHURNED = "churned"


class User(Base):
    """Telegram user model."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    status: Mapped[UserStatus] = mapped_column(
        Enum(UserStatus), 
        default=UserStatus.NEW,
        nullable=False
    )
    is_trained: Mapped[bool] = mapped_column(Boolean, default=False)
    bonus_channels_count: Mapped[int] = mapped_column(default=0)
    initial_best_post_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    language: Mapped[str] = mapped_column(String(10), default="en_US")
    
    # ML preference vector cache (computed from user's liked/disliked posts)
    preference_vector_cache: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    preference_vector_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
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
    channels: Mapped[List["UserChannel"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    interactions: Mapped[List["Interaction"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    logs: Mapped[List["UserLog"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_user_status", "status"),
        Index("idx_user_last_activity", "last_activity_at"),
        Index("idx_user_is_deleted", "is_deleted"),
    )

