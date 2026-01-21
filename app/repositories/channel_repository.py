"""Channel repository for ML Service."""
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.channel import Channel


class ChannelRepository:
    """Repository for channel operations (read-only)."""
    
    @staticmethod
    async def get_by_id(db: AsyncSession, channel_id: int) -> Optional[Channel]:
        """Get channel by ID."""
        result = await db.execute(
            select(Channel).where(
                Channel.id == channel_id,
                Channel.is_deleted == False
            )
        )
        return result.scalar_one_or_none()

