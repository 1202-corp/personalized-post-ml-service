"""Post repository for ML Service."""
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.post import Post
from app.models.channel import Channel


class PostRepository:
    """Repository for post operations (read-only)."""
    
    @staticmethod
    async def get_by_id(db: AsyncSession, post_id: int) -> Optional[Post]:
        """Get post by ID."""
        result = await db.execute(
            select(Post).where(
                Post.id == post_id,
                Post.is_deleted == False
            )
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_all_by_channel(db: AsyncSession, channel_id: int) -> List[Post]:
        """Get all posts for a channel."""
        result = await db.execute(
            select(Post).where(
                Post.channel_id == channel_id,
                Post.is_deleted == False
            ).order_by(Post.posted_at.desc())
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_all(db: AsyncSession) -> List[Post]:
        """Get all posts (excluding deleted)."""
        result = await db.execute(
            select(Post).where(Post.is_deleted == False)
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_by_ids(db: AsyncSession, post_ids: List[int]) -> List[Post]:
        """Get posts by IDs (excluding deleted)."""
        if not post_ids:
            return []
        result = await db.execute(
            select(Post).where(
                Post.id.in_(post_ids),
                Post.is_deleted == False
            )
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def update_relevance_score(
        db: AsyncSession,
        post_id: int,
        relevance_score: float
    ) -> bool:
        """Update post's relevance score (with commit)."""
        from sqlalchemy import update
        result = await db.execute(
            update(Post)
            .where(
                Post.id == post_id,
                Post.is_deleted == False
            )
            .values(relevance_score=relevance_score)
        )
        return result.rowcount > 0

