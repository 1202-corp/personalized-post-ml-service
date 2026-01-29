"""Taste cluster endpoints: cluster users by preference vector for post-centric delivery."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.taste_cluster import TasteCluster
from app.models.user import User
from app.services import taste_cluster_service

router = APIRouter(prefix="/taste-clusters", tags=["taste-clusters"])


@router.post("/recalculate")
async def recalculate_taste_clusters_endpoint(
    session: AsyncSession = Depends(get_session),
):
    """
    Full recalculate of taste clusters.
    All users with a preference vector are clustered; max_size = ceil(N * 0.017).
    """
    try:
        result = await taste_cluster_service.recalculate_taste_clusters(session)
        await session.commit()
        return result
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error recalculating taste clusters: {str(e)}",
        )


@router.get("/stats")
async def get_taste_cluster_stats(session: AsyncSession = Depends(get_session)):
    """Statistics about taste clusters and user assignment."""
    result = await session.execute(
        select(TasteCluster.id, TasteCluster.user_count).where(
            TasteCluster.centroid.isnot(None)
        )
    )
    rows = result.all()
    num_clusters = len(rows)
    users_with_cluster = await session.scalar(
        select(func.count(User.id)).where(
            User.is_deleted == False,
            User.taste_cluster_id.isnot(None),
        )
    )
    total_users = await session.scalar(
        select(func.count(User.id)).where(User.is_deleted == False)
    )
    total_users = total_users or 0
    users_with_cluster = users_with_cluster or 0
    users_without_taste_cluster = total_users - users_with_cluster
    distribution = [{"cluster_id": row[0], "user_count": row[1]} for row in rows]
    avg_users_per_cluster = (sum(r[1] for r in rows) / num_clusters) if num_clusters else 0
    max_users_in_cluster = max((r[1] for r in rows), default=0)
    return {
        "num_clusters": num_clusters,
        "total_users": total_users,
        "users_with_taste_cluster": users_with_cluster,
        "users_without_taste_cluster": users_without_taste_cluster,
        "avg_users_per_cluster": round(avg_users_per_cluster, 1),
        "max_users_in_cluster": max_users_in_cluster,
        "cluster_distribution": distribution,
    }
