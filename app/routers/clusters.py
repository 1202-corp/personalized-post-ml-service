"""Cluster management endpoints for ML Service."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database import get_session
from app.services import cluster_service

router = APIRouter(prefix="/clusters", tags=["clusters"])


class RecalculateClustersRequest(BaseModel):
    """Request model for cluster recalculation."""
    n_clusters: int = 50


@router.post("/recalculate")
async def recalculate_clusters(
    request: RecalculateClustersRequest,
    session: AsyncSession = Depends(get_session)
):
    """Recalculate post clusters for optimized search.
    
    This will group similar posts together based on their embeddings,
    allowing faster search by filtering through clusters first.
    """
    try:
        result = await cluster_service.recalculate_clusters(
            session, 
            n_clusters=request.n_clusters
        )
        await session.commit()
        return result
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error recalculating clusters: {str(e)}"
        )


@router.get("/stats")
async def get_cluster_stats(session: AsyncSession = Depends(get_session)):
    """Get statistics about current post clusters."""
    from sqlalchemy import func, select
    from app.models.post import Post
    
    # Get cluster distribution
    result = await session.execute(
        select(Post.cluster_id, func.count(Post.id).label("count"))
        .where(Post.is_deleted == False, Post.cluster_id.isnot(None))
        .group_by(Post.cluster_id)
    )
    cluster_counts = result.all()
    
    # Get total stats
    total_posts = await session.scalar(
        select(func.count(Post.id)).where(Post.is_deleted == False)
    )
    clustered_posts = await session.scalar(
        select(func.count(Post.id)).where(
            Post.is_deleted == False,
            Post.cluster_id.isnot(None)
        )
    )
    unclustered_posts = (total_posts or 0) - (clustered_posts or 0)
    
    return {
        "total_posts": total_posts or 0,
        "clustered_posts": clustered_posts or 0,
        "unclustered_posts": unclustered_posts,
        "num_clusters": len(cluster_counts),
        "cluster_distribution": [
            {"cluster_id": row[0], "post_count": row[1]} 
            for row in cluster_counts
        ]
    }

