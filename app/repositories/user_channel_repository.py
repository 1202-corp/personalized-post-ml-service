"""UserChannel repository for ML Service."""
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user_channel import UserChannel
from app.models.user import User
from app.models.channel import Channel


class UserChannelRepository:
    """Repository for user-channel association operations (read-only)."""
    
    @staticmethod
    async def get_by_user_id(db: AsyncSession, user_id: int) -> List[UserChannel]:
        """Get all channel associations for a user."""
        result = await db.execute(
            select(UserChannel)
            .join(User)
            .where(
                User.id == user_id,
                User.is_deleted == False
            )
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_training_channels_by_user(
        db: AsyncSession,
        user_id: int
    ) -> List[Channel]:
        """Get all training channels for a user."""
        result = await db.execute(
            select(Channel)
            .join(UserChannel)
            .join(User)
            .where(
                User.id == user_id,
                UserChannel.is_for_training == True,
                User.is_deleted == False,
                Channel.is_deleted == False
            )
        )
        return list(result.scalars().all())

