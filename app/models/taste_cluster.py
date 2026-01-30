"""Taste cluster ORM model — clusters of users by preference vector, per channel."""
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import Integer, JSON, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

if TYPE_CHECKING:
    from app.models.channel import Channel
    from app.models.user import User


class TasteCluster(Base):
    """Cluster of users with similar taste for one channel. Each channel has its own set of clusters."""
    __tablename__ = "taste_clusters"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("channels.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )  # NULL = legacy global cluster (deprecated)
    centroid: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    user_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    channel: Mapped[Optional["Channel"]] = relationship(back_populates="taste_clusters")
    # Legacy: users with User.taste_cluster_id pointing here (deprecated; per-channel uses UserChannelTaste)
    users: Mapped[List["User"]] = relationship(
        back_populates="taste_cluster",
        foreign_keys="User.taste_cluster_id",
    )

