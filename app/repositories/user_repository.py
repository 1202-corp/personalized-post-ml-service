"""User repository for ML Service."""
from typing import Optional, List
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User, UserStatus


class UserRepository:
    """Repository for user operations (read-only + preference_vector_cache update)."""
    
    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
        """Get user by ID."""
        result = await db.execute(
            select(User).where(
                User.id == user_id,
                User.is_deleted == False
            )
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_telegram_id(db: AsyncSession, telegram_id: int) -> Optional[User]:
        """Get user by Telegram ID."""
        result = await db.execute(
            select(User).where(
                User.telegram_id == telegram_id,
                User.is_deleted == False
            )
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_preference_vector(
        db: AsyncSession,
        user_id: int,
        preference_vector: List[float]
    ) -> bool:
        """Update user's preference vector cache (with commit)."""
        from datetime import datetime
        result = await db.execute(
            update(User)
            .where(
                User.id == user_id,
                User.is_deleted == False
            )
            .values(
                preference_vector_cache=preference_vector,
                preference_vector_updated_at=datetime.utcnow()
            )
        )
        return result.rowcount > 0

