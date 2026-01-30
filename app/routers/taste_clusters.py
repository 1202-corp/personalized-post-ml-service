"""Taste cluster endpoints: cluster users by preference vector per channel."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.taste_cluster import TasteCluster
from app.models.user import User
from app.models.user_channel_taste import UserChannelTaste
from app.services import taste_cluster_service

router = APIRouter(prefix="/taste-clusters", tags=["taste-clusters"])


@router.post("/recalculate")
async def recalculate_taste_clusters_endpoint(
    session: AsyncSession = Depends(get_session),
):
    """
    Full recalculate of taste clusters per channel.
    For each channel with UserChannelPreferenceVector rows, runs K-means and assigns users.
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
    """Statistics about taste clusters (per channel) and user assignment (UserChannelTaste)."""
    result = await session.execute(
        select(TasteCluster.id, TasteCluster.channel_id, TasteCluster.user_count).where(
            TasteCluster.centroid.isnot(None)
        )
    )
    rows = result.all()
    num_clusters = len(rows)
    users_with_cluster = await session.scalar(
        select(func.count(func.distinct(UserChannelTaste.user_id)))
    )
    users_with_cluster = users_with_cluster or 0
    total_users = await session.scalar(
        select(func.count(User.id)).where(User.is_deleted == False)
    )
    total_users = total_users or 0
    users_without_taste_cluster = total_users - users_with_cluster
    distribution = [
        {"cluster_id": row[0], "channel_id": row[1], "user_count": row[2]}
        for row in rows
    ]
    avg_users_per_cluster = (sum(r[2] for r in rows) / num_clusters) if num_clusters else 0
    max_users_in_cluster = max((r[2] for r in rows), default=0)
    return {
        "num_clusters": num_clusters,
        "total_users": total_users,
        "users_with_taste_cluster": users_with_cluster,
        "users_without_taste_cluster": users_without_taste_cluster,
        "avg_users_per_cluster": round(avg_users_per_cluster, 1),
        "max_users_in_cluster": max_users_in_cluster,
        "cluster_distribution": distribution,
    }
