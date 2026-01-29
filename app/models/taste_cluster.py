"""Taste cluster ORM model — clusters of users by preference vector for post-centric delivery."""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Integer, JSON, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class TasteCluster(Base):
    """Cluster of users with similar taste (preference vector). Used to decide who receives a new post."""
    __tablename__ = "taste_clusters"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    centroid: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    user_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    users: Mapped[List["User"]] = relationship(back_populates="taste_cluster")

