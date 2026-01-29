"""UserPreferenceVector ORM model (same table as API)."""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import ForeignKey, JSON, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class UserPreferenceVector(Base):
    """User preference vector cache (1:1 with User)."""
    __tablename__ = "user_preference_vectors"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    preference_vector: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="preference_vector")
