"""UserChannelPreferenceVector — user's taste vector per channel."""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import ForeignKey, JSON, DateTime, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class UserChannelPreferenceVector(Base):
    """User preference vector for one channel (taste for that channel only)."""
    __tablename__ = "user_channel_preference_vectors"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channels.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    preference_vector: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (UniqueConstraint("user_id", "channel_id", name="uq_user_channel_pref"),)
