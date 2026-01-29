"""TasteCluster repository for ML Service."""
from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.taste_cluster import TasteCluster
from app.models.user import User


class TasteClusterRepository:
    """Repository for taste cluster operations."""

    @staticmethod
    async def get_all(db: AsyncSession) -> List[TasteCluster]:
        """Get all taste clusters with centroids."""
        result = await db.execute(
            select(TasteCluster).where(TasteCluster.centroid.isnot(None))
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_by_id(db: AsyncSession, cluster_id: int) -> Optional[TasteCluster]:
        """Get taste cluster by ID."""
        result = await db.execute(
            select(TasteCluster).where(TasteCluster.id == cluster_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user_ids_by_cluster_ids(
        db: AsyncSession,
        cluster_ids: List[int],
    ) -> List[int]:
        """Get user IDs that belong to any of the given cluster IDs."""
        if not cluster_ids:
            return []
        result = await db.execute(
            select(User.id).where(
                User.taste_cluster_id.in_(cluster_ids),
                User.is_deleted == False,
            )
        )
        return [row[0] for row in result.all()]

    @staticmethod
    async def delete_all(db: AsyncSession) -> None:
        """Delete all taste clusters (for full recalculate)."""
        await db.execute(delete(TasteCluster))
        await db.flush()

    @staticmethod
    async def create(
        db: AsyncSession,
        centroid: List[float],
        user_count: int = 0,
    ) -> TasteCluster:
        """Create a new taste cluster."""
        cluster = TasteCluster(centroid=centroid, user_count=user_count)
        db.add(cluster)
        await db.flush()
        return cluster
