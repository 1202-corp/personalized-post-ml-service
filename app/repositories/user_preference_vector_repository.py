"""UserPreferenceVector repository for ML Service."""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user_preference_vector import UserPreferenceVector


class UserPreferenceVectorRepository:
    """Repository for user preference vector cache."""

    @staticmethod
    async def get_by_user_id(db: AsyncSession, user_id: int) -> Optional[UserPreferenceVector]:
        """Get preference vector row for user."""
        result = await db.execute(
            select(UserPreferenceVector).where(UserPreferenceVector.user_id == user_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def upsert(
        db: AsyncSession,
        user_id: int,
        preference_vector: List[float],
    ) -> bool:
        """Insert or update preference vector for user."""
        now = datetime.utcnow()
        stmt = insert(UserPreferenceVector).values(
            user_id=user_id,
            preference_vector=preference_vector,
            updated_at=now,
        ).on_conflict_do_update(
            index_elements=["user_id"],
            set_={
                "preference_vector": preference_vector,
                "updated_at": now,
            },
        )
        await db.execute(stmt)
        await db.flush()
        return True
