"""UserChannelPreferenceVector repository for ML Service."""
from datetime import datetime
from typing import Optional, List, Tuple
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user_channel_preference_vector import UserChannelPreferenceVector


class UserChannelPreferenceVectorRepository:
    """Repository for per-channel user preference vectors."""

    @staticmethod
    async def get_by_user_and_channel(
        db: AsyncSession, user_id: int, channel_id: int
    ) -> Optional[UserChannelPreferenceVector]:
        """Get preference vector row for user in channel."""
        result = await db.execute(
            select(UserChannelPreferenceVector).where(
                UserChannelPreferenceVector.user_id == user_id,
                UserChannelPreferenceVector.channel_id == channel_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_vectors_by_channel(
        db: AsyncSession, channel_id: int
    ) -> List[Tuple[int, List[float]]]:
        """Get (user_id, preference_vector) for all users that have a vector for this channel."""
        result = await db.execute(
            select(
                UserChannelPreferenceVector.user_id,
                UserChannelPreferenceVector.preference_vector,
            ).where(
                UserChannelPreferenceVector.channel_id == channel_id,
                UserChannelPreferenceVector.preference_vector.isnot(None),
            )
        )
        out = []
        for row in result.all():
            user_id, pv = row[0], row[1]
            if pv and isinstance(pv, list) and len(pv) > 0:
                out.append((user_id, pv))
        return out

    @staticmethod
    async def upsert(
        db: AsyncSession,
        user_id: int,
        channel_id: int,
        preference_vector: List[float],
    ) -> bool:
        """Insert or update preference vector for user in channel."""
        now = datetime.utcnow()
        stmt = insert(UserChannelPreferenceVector).values(
            user_id=user_id,
            channel_id=channel_id,
            preference_vector=preference_vector,
            updated_at=now,
        ).on_conflict_do_update(
            index_elements=["user_id", "channel_id"],
            set_={
                "preference_vector": preference_vector,
                "updated_at": now,
            },
        )
        await db.execute(stmt)
        await db.flush()
        return True
