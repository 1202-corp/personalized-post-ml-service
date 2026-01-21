"""ML endpoints for ML Service."""
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database import get_session
from app.services import ml_service

router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    """Request model for training."""
    user_telegram_id: int


class TrainResponse(BaseModel):
    """Response model for training."""
    success: bool
    message: str
    training_time: float


class PredictRequest(BaseModel):
    """Request model for predictions."""
    user_telegram_id: int
    post_ids: List[int]


class PredictResponse(BaseModel):
    """Response model for predictions."""
    predictions: dict


class RecommendationsRequest(BaseModel):
    """Request model for recommendations."""
    user_telegram_id: int
    limit: int = 10
    exclude_interacted: bool = True


class RecommendationItem(BaseModel):
    """Recommendation item."""
    post_id: int
    score: float
    payload: dict = {}


class RecommendationsResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[RecommendationItem]


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Train ML model for a user using embeddings and vector similarity.
    
    Process:
    1. Gets user's liked/disliked posts
    2. Generates embeddings via OpenAI-compatible API
    3. Stores embeddings in Qdrant vector database
    4. Computes user preference vector
    5. Scores all posts in user's channels
    """
    success, message, training_time = await ml_service.train_model(
        session, request.user_telegram_id
    )
    return TrainResponse(
        success=success,
        message=message,
        training_time=round(training_time, 2)
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Get ML predictions for posts.
    Returns relevance scores based on user preference vector similarity.
    """
    predictions = await ml_service.predict(
        session,
        request.user_telegram_id,
        request.post_ids
    )
    return PredictResponse(predictions=predictions)


@router.post("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(
    request: RecommendationsRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Get personalized post recommendations for a user.
    Uses vector similarity search to find posts similar to user's preferences.
    """
    recommendations = await ml_service.get_recommended_posts(
        session,
        request.user_telegram_id,
        limit=request.limit,
        exclude_interacted=request.exclude_interacted
    )
    return RecommendationsResponse(
        recommendations=[
            RecommendationItem(
                post_id=r['post_id'],
                score=r['score'],
                payload=r.get('payload', {})
            )
            for r in recommendations
        ]
    )


@router.get("/eligibility/{telegram_id}")
async def check_eligibility(
    telegram_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Check if user is eligible to start ML training."""
    eligible, message = await ml_service.check_training_eligibility(session, telegram_id)
    return {"eligible": eligible, "message": message}

