"""UserChannel ORM model."""
from datetime import datetime
from sqlalchemy import Boolean, ForeignKey, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class UserChannel(Base):
    """User-channel relationship model."""
    __tablename__ = "user_channels"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    
    is_for_training: Mapped[bool] = mapped_column(Boolean, default=False)
    is_bonus: Mapped[bool] = mapped_column(Boolean, default=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="channels")
    channel: Mapped["Channel"] = relationship(back_populates="user_channels")
    
    __table_args__ = (
        Index("idx_user_channel_unique", "user_id", "channel_id", unique=True),
    )

