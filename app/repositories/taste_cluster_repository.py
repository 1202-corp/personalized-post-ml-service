"""TasteCluster repository for ML Service."""
from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.taste_cluster import TasteCluster


class TasteClusterRepository:
    """Repository for taste cluster operations (per channel)."""

    @staticmethod
    async def get_all(
        db: AsyncSession, channel_id: Optional[int] = None
    ) -> List[TasteCluster]:
        """Get taste clusters with centroids. If channel_id given, only that channel."""
        q = select(TasteCluster).where(TasteCluster.centroid.isnot(None))
        if channel_id is not None:
            q = q.where(TasteCluster.channel_id == channel_id)
        result = await db.execute(q)
        return list(result.scalars().all())

    @staticmethod
    async def get_by_id(db: AsyncSession, cluster_id: int) -> Optional[TasteCluster]:
        """Get taste cluster by ID."""
        result = await db.execute(
            select(TasteCluster).where(TasteCluster.id == cluster_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def delete_all(
        db: AsyncSession, channel_id: Optional[int] = None
    ) -> None:
        """Delete taste clusters. If channel_id given, only that channel's clusters."""
        q = delete(TasteCluster)
        if channel_id is not None:
            q = q.where(TasteCluster.channel_id == channel_id)
        await db.execute(q)
        await db.flush()

    @staticmethod
    async def create(
        db: AsyncSession,
        centroid: List[float],
        user_count: int = 0,
        channel_id: Optional[int] = None,
    ) -> TasteCluster:
        """Create a new taste cluster (optionally for a channel)."""
        cluster = TasteCluster(
            centroid=centroid, user_count=user_count, channel_id=channel_id
        )
        db.add(cluster)
        await db.flush()
        return cluster
