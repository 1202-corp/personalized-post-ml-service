"""UserChannelTaste — user's taste cluster assignment per channel."""
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class UserChannelTaste(Base):
    """User's taste cluster for one channel (which cluster they belong to for that channel)."""
    __tablename__ = "user_channel_tastes"

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
    taste_cluster_id: Mapped[int] = mapped_column(
        ForeignKey("taste_clusters.id", ondelete="CASCADE"),
        nullable=False,
    )

    __table_args__ = (UniqueConstraint("user_id", "channel_id", name="uq_user_channel_taste"),)
