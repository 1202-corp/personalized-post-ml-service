"""UserChannelTaste repository for ML Service."""
from typing import List
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user_channel_taste import UserChannelTaste


class UserChannelTasteRepository:
    """Repository for per-channel user taste cluster assignment."""

    @staticmethod
    async def get_by_user_and_channel(
        db: AsyncSession, user_id: int, channel_id: int
    ):
        """Get taste cluster assignment for user in channel."""
        result = await db.execute(
            select(UserChannelTaste).where(
                UserChannelTaste.user_id == user_id,
                UserChannelTaste.channel_id == channel_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user_ids_in_clusters(
        db: AsyncSession, channel_id: int, cluster_ids: List[int]
    ) -> List[int]:
        """Get user IDs that belong to any of the given clusters in this channel."""
        if not cluster_ids:
            return []
        result = await db.execute(
            select(UserChannelTaste.user_id).where(
                UserChannelTaste.channel_id == channel_id,
                UserChannelTaste.taste_cluster_id.in_(cluster_ids),
            ).distinct()
        )
        return [row[0] for row in result.all()]

    @staticmethod
    async def upsert(
        db: AsyncSession,
        user_id: int,
        channel_id: int,
        taste_cluster_id: int,
    ) -> bool:
        """Insert or update taste cluster assignment for user in channel."""
        stmt = insert(UserChannelTaste).values(
            user_id=user_id,
            channel_id=channel_id,
            taste_cluster_id=taste_cluster_id,
        ).on_conflict_do_update(
            index_elements=["user_id", "channel_id"],
            set_={"taste_cluster_id": taste_cluster_id},
        )
        await db.execute(stmt)
        await db.flush()
        return True
