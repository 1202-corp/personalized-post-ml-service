"""Channel ORM model."""
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String, BigInteger, Boolean, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.taste_cluster import TasteCluster


class Channel(Base):
    """Telegram channel model."""
    __tablename__ = "channels"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

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
    posts: Mapped[List["Post"]] = relationship(back_populates="channel", cascade="all, delete-orphan")
    user_channels: Mapped[List["UserChannel"]] = relationship(back_populates="channel", cascade="all, delete-orphan")
    taste_clusters: Mapped[List["TasteCluster"]] = relationship(
        back_populates="channel", cascade="all, delete-orphan"
    )

