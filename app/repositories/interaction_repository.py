"""Interaction repository for ML Service."""
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.interaction import Interaction, InteractionType
from app.models.user import User


class InteractionRepository:
    """Repository for interaction operations (read-only)."""
    
    @staticmethod
    async def get_by_user_id(db: AsyncSession, user_id: int) -> List[Interaction]:
        """Get all interactions for a user."""
        result = await db.execute(
            select(Interaction)
            .join(User)
            .where(
                User.id == user_id,
                User.is_deleted == False
            )
            .order_by(Interaction.created_at.desc())
        )
        return list(result.scalars().all())

