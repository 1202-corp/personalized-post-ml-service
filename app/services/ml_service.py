"""
ML Service - Real implementation using embeddings and Qdrant vector search.
"""

import logging
import time
from typing import List, Dict, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
import httpx

from app.config import get_settings
from app.services import embedding_service, qdrant_service
from app.repositories.user_repository import UserRepository
from app.repositories.user_preference_vector_repository import UserPreferenceVectorRepository
from app.repositories.post_repository import PostRepository
from app.repositories.interaction_repository import InteractionRepository
from app.repositories.user_channel_repository import UserChannelRepository
from app.repositories.channel_repository import ChannelRepository
from app.models.user import UserStatus, User
from app.models.user_channel import UserChannel
from app.models.post import Post
from app.models.channel import Channel
from app.models.interaction import InteractionType
from app.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

_redis_client: Optional[aioredis.Redis] = None


async def _get_redis() -> aioredis.Redis:
    """Lazy Redis client (same URL as API, read-only for post content)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _redis_client


async def get_post_texts_from_redis(post_ids: List[int]) -> Dict[int, str]:
    """
    Fetch post text for given post IDs from Redis (same cache as API).
    Key format: post:{post_id}:content, hash field 'text'.
    Returns dict post_id -> text (empty string if missing).
    """
    if not post_ids:
        return {}
    try:
        redis_client = await _get_redis()
        pipe = redis_client.pipeline()
        for pid in post_ids:
            pipe.hget(f"post:{pid}:content", "text")
        raw = await pipe.execute()
        result = {}
        for pid, val in zip(post_ids, raw):
            if val is not None:
                try:
                    result[pid] = val.decode("utf-8") if isinstance(val, bytes) else str(val)
                except Exception:
                    result[pid] = ""
            else:
                result[pid] = ""
        return result
    except Exception as e:
        logger.warning(f"Failed to get post texts from Redis: {e}")
        return {pid: "" for pid in post_ids}


async def fetch_post_text_from_api(post_id: int) -> str:
    """
    Fetch post content from API (API will load from Redis or fetch from user-bot).
    Returns text or empty string on failure.
    """
    try:
        base = settings.core_api_url.rstrip("/")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{base}/api/v1/posts/{post_id}/content")
            if resp.status_code == 200:
                data = resp.json()
                return (data.get("text") or "").strip()
    except Exception as e:
        logger.debug(f"API fetch post content failed (post_id={post_id}): {e}")
    return ""


async def train_model(session: AsyncSession, user_telegram_id: int) -> tuple[bool, str, float]:
    """
    Train ML model for a user using embeddings and Qdrant.
    
    Process:
    1. Get user's liked and disliked posts
    2. Generate embeddings for posts that don't have them
    3. Store embeddings in Qdrant
    4. Compute user preference vector
    5. Calculate relevance scores for all posts in user's channels
    
    Returns (success, message, training_time).
    """
    start_time = time.time()
    
    # Check if user has enough interactions
    user = await UserRepository.get_by_telegram_id(session, user_telegram_id)
    if not user:
        return False, "User not found", time.time() - start_time
    
    interactions = await InteractionRepository.get_by_user_id(session, user.id)
    interaction_count = len(interactions)
    
    # Require at least one interaction to train; full training flow уже гарантирует достаточное количество
    if interaction_count == 0:
        training_time = time.time() - start_time
        return False, "Need at least 1 interaction to train", training_time
    
    try:
        # Get user's interactions with posts
        liked_posts, disliked_posts = await _get_user_interaction_posts(session, user.id)
        
        # Generate and store embeddings for all interacted posts
        all_posts = liked_posts + disliked_posts
        await _ensure_post_embeddings(session, all_posts)
        
        # Get embeddings for liked and disliked posts
        liked_embeddings = await _get_embeddings_for_posts([p.id for p in liked_posts])
        disliked_embeddings = await _get_embeddings_for_posts([p.id for p in disliked_posts])
        
        # Compute user preference vector
        preference_vector = await qdrant_service.get_user_preference_vector(
            liked_embeddings,
            disliked_embeddings if disliked_embeddings else None
        )
        
        if not preference_vector:
            return False, "Could not compute preference vector", time.time() - start_time
        
        # Save preference vector to user cache
        await UserRepository.update_preference_vector(session, user.id, preference_vector)
        await session.flush()

        # Recalculate taste clusters (creates clusters when none, or N=1; assigns all users)
        from app.services import taste_cluster_service
        await taste_cluster_service.recalculate_taste_clusters(session)

        # Feed scoring is done on the fly via predict() when get_best_posts is called; no Post.relevance_score write.
        
        # Commit all changes
        await session.commit()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed for user {user_telegram_id} in {training_time:.2f}s")
        
        return True, "Model trained successfully", training_time
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Training failed for user {user_telegram_id}: {e}", exc_info=True)
        return False, f"Training failed: {str(e)}", time.time() - start_time


async def predict(
    session: AsyncSession,
    user_telegram_id: int,
    post_ids: List[int]
) -> Dict[int, float]:
    """
    Get relevance predictions for specific posts.
    Uses user's preference vector and post embeddings.
    """
    try:
        user = await UserRepository.get_by_telegram_id(session, user_telegram_id)
        if not user:
            return {}
        
        # Try to use cached preference vector first (from user_preference_vectors table)
        pv_row = await UserPreferenceVectorRepository.get_by_user_id(session, user.id)
        preference_vector = pv_row.preference_vector if pv_row else None
        
        # If no cache, recalculate
        if not preference_vector:
            # Get user's liked posts for preference vector
            liked_posts, disliked_posts = await _get_user_interaction_posts(session, user.id)
            
            liked_embeddings = await _get_embeddings_for_posts([p.id for p in liked_posts])
            disliked_embeddings = await _get_embeddings_for_posts([p.id for p in disliked_posts])
            
            preference_vector = await qdrant_service.get_user_preference_vector(
                liked_embeddings,
                disliked_embeddings if disliked_embeddings else None
            )
            
            # Cache the vector if computed
            if preference_vector:
                await UserPreferenceVectorRepository.upsert(session, user.id, preference_vector)
                await session.commit()
        
        if not preference_vector:
            # Fallback to neutral scores
            return {pid: 0.5 for pid in post_ids}
        
        # Get posts and ensure they have embeddings
        posts = []
        for post_id in post_ids:
            post = await PostRepository.get_by_id(session, post_id)
            if post:
                posts.append(post)
        
        await _ensure_post_embeddings(session, posts)
        
        # Calculate similarity scores
        predictions = {}
        post_embeddings = await qdrant_service.get_post_embeddings_batch(post_ids)
        
        for post_id in post_ids:
            if post_id in post_embeddings:
                score = _cosine_similarity(preference_vector, post_embeddings[post_id])
                # Normalize to 0-1 range (cosine similarity is -1 to 1)
                score = (score + 1) / 2
                predictions[post_id] = round(score, 4)
            else:
                predictions[post_id] = 0.5

        return predictions
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return {pid: 0.5 for pid in post_ids}


async def get_recommended_posts(
    session: AsyncSession,
    user_telegram_id: int,
    limit: int = 10,
    exclude_interacted: bool = True
) -> List[Dict]:
    """
    Get recommended posts for user using vector similarity search.
    """
    try:
        user = await UserRepository.get_by_telegram_id(session, user_telegram_id)
        if not user:
            return []
        
        # Try to use cached preference vector first (from user_preference_vectors table)
        pv_row = await UserPreferenceVectorRepository.get_by_user_id(session, user.id)
        preference_vector = pv_row.preference_vector if pv_row else None
        
        # If no cache, try to compute from interactions
        if not preference_vector:
            liked_posts, disliked_posts = await _get_user_interaction_posts(session, user.id)
            
            if not liked_posts:
                # No liked posts yet - return recent posts
                return []
            
            liked_embeddings = await _get_embeddings_for_posts([p.id for p in liked_posts])
            disliked_embeddings = await _get_embeddings_for_posts([p.id for p in disliked_posts])
            
            preference_vector = await qdrant_service.get_user_preference_vector(
                liked_embeddings,
                disliked_embeddings if disliked_embeddings else None
            )
            
            # Cache if computed
            if preference_vector:
                await UserPreferenceVectorRepository.upsert(session, user.id, preference_vector)
                await session.commit()
        
        if not preference_vector:
            return []
        
        # Get IDs of posts to exclude
        exclude_ids = set()
        if exclude_interacted:
            interactions = await InteractionRepository.get_by_user_id(session, user.id)
            exclude_ids = {i.post_id for i in interactions}
        
        # Direct Qdrant search (post-clusters removed; delivery is post-centric via taste clusters)
        search_limit = limit + len(exclude_ids)
        results = await qdrant_service.search_similar_posts(
            query_vector=preference_vector,
            limit=search_limit,
            score_threshold=settings.default_score_threshold,
        )
        
        recommended = []
        for r in results:
            if r['id'] not in exclude_ids and len(recommended) < limit:
                recommended.append({
                    'post_id': r['id'],
                    'score': r['score'],
                    'payload': r.get('payload', {}),
                })
        
        return recommended
        
    except Exception as e:
        logger.error(f"Get recommendations failed: {e}", exc_info=True)
        return []


async def check_training_eligibility(session: AsyncSession, user_telegram_id: int) -> tuple[bool, str]:
    """Check if user is eligible to start training."""
    user = await UserRepository.get_by_telegram_id(session, user_telegram_id)
    
    if not user:
        return False, "User not found"
    
    interactions = await InteractionRepository.get_by_user_id(session, user.id)
    interaction_count = len(interactions)
    
    if interaction_count == 0:
        return False, "Need at least 1 interaction to start training"
    
    return True, "Ready for training"


# ==================== Helper Functions ====================

async def _get_user_interaction_posts(
    session: AsyncSession,
    user_id: int
) -> tuple[List[Post], List[Post]]:
    """Get user's liked and disliked posts."""
    interactions = await InteractionRepository.get_by_user_id(session, user_id)
    
    liked_post_ids = [
        i.post_id for i in interactions 
        if i.interaction_type == InteractionType.LIKE
    ]
    disliked_post_ids = [
        i.post_id for i in interactions 
        if i.interaction_type == InteractionType.DISLIKE
    ]
    
    liked_posts = []
    for post_id in liked_post_ids:
        post = await PostRepository.get_by_id(session, post_id)
        if post:
            liked_posts.append(post)
    
    disliked_posts = []
    for post_id in disliked_post_ids:
        post = await PostRepository.get_by_id(session, post_id)
        if post:
            disliked_posts.append(post)
    
    return liked_posts, disliked_posts


async def _ensure_post_embeddings(session: AsyncSession, posts: List[Post]) -> None:
    """Ensure all posts have embeddings in Qdrant."""
    # Check which posts need embeddings
    post_ids = [p.id for p in posts]
    existing = await qdrant_service.get_post_embeddings_batch(post_ids)
    
    posts_needing_embeddings = [p for p in posts if p.id not in existing]
    
    if not posts_needing_embeddings:
        return
    
    # Get channel info for context
    channel_cache = {}
    for post in posts_needing_embeddings:
        if post.channel_id not in channel_cache:
            channel = await ChannelRepository.get_by_id(session, post.channel_id)
            channel_cache[post.channel_id] = channel.title if channel else None
    
    # Fetch post text: first Redis, then API (API may fetch from user-bot)
    post_texts = await get_post_texts_from_redis([p.id for p in posts_needing_embeddings])
    missing = [p for p in posts_needing_embeddings if not (post_texts.get(p.id) or "").strip()]
    for p in missing:
        text = await fetch_post_text_from_api(p.id)
        if text:
            post_texts[p.id] = text
    
    # Never send posts without text to embedding — only posts with text
    posts_with_text = [p for p in posts_needing_embeddings if (post_texts.get(p.id) or "").strip()]
    if not posts_with_text:
        logger.info("No posts with text to embed; skipping embedding step")
        return
    skipped = len(posts_needing_embeddings) - len(posts_with_text)
    if skipped:
        logger.info(f"Skipped {skipped} post(s) without text (no embedding)")
    
    # Prepare texts for embedding (post text + channel context)
    texts = [
        embedding_service.prepare_post_text(
            (post_texts.get(p.id) or "").strip(),
            channel_cache.get(p.channel_id)
        )
        for p in posts_with_text
    ]
    
    # Get embeddings
    embeddings = await embedding_service.get_embeddings_batch(texts)
    
    # Store in Qdrant (text_preview for search/debug)
    points = []
    for post, emb in zip(posts_with_text, embeddings):
        if emb:
            raw_text = (post_texts.get(post.id) or "").strip()
            points.append({
                'id': post.id,
                'vector': emb,
                'payload': {
                    'channel_id': post.channel_id,
                    'text_preview': raw_text[:200],
                }
            })
    
    if points:
        await qdrant_service.upsert_post_embeddings_batch(points)
        logger.info(f"Stored {len(points)} post embeddings in Qdrant")


async def _get_embeddings_for_posts(post_ids: List[int]) -> List[List[float]]:
    """Get embeddings for posts from Qdrant."""
    if not post_ids:
        return []
    
    embeddings_dict = await qdrant_service.get_post_embeddings_batch(post_ids)
    return [embeddings_dict[pid] for pid in post_ids if pid in embeddings_dict]


from app.services.utils import cosine_similarity as _cosine_similarity


async def maybe_recalc_taste_after_interaction(
    session: AsyncSession,
    user_telegram_id: int,
) -> bool:
    """
    After every 2 reactions (like/dislike), recalc user preference vector and assign to nearest cluster.
    Returns True if recalculated.
    """
    user = await UserRepository.get_by_telegram_id(session, user_telegram_id)
    if not user:
        return False
    interactions = await InteractionRepository.get_by_user_id(session, user.id)
    count = len(interactions)
    if count < 2 or count % 2 != 0:
        return False
    liked_posts, disliked_posts = await _get_user_interaction_posts(session, user.id)
    if not liked_posts:
        return False
    all_posts = liked_posts + disliked_posts
    await _ensure_post_embeddings(session, all_posts)
    liked_embeddings = await _get_embeddings_for_posts([p.id for p in liked_posts])
    disliked_embeddings = await _get_embeddings_for_posts([p.id for p in disliked_posts])
    preference_vector = await qdrant_service.get_user_preference_vector(
        liked_embeddings,
        disliked_embeddings if disliked_embeddings else None,
    )
    if not preference_vector:
        return False
    await UserPreferenceVectorRepository.upsert(session, user.id, preference_vector)
    from app.services import taste_cluster_service
    await taste_cluster_service.assign_user_to_nearest_cluster(
        session, user.id, preference_vector
    )
    await session.flush()
    logger.info(f"Taste recalculated for user {user_telegram_id} after {count} interactions")
    return True


async def get_post_recipients(
    session: AsyncSession,
    post_id: int,
    post_text: Optional[str] = None,
) -> List[int]:
    """
    Post-centric delivery: return telegram_ids of users who should receive this post.
    Users must be in a taste cluster matching the post embedding and have mailing_enabled for this channel.
    """
    post = await PostRepository.get_by_id(session, post_id)
    if not post:
        return []

    # Get or create post embedding
    existing = await qdrant_service.get_post_embeddings_batch([post_id])
    if post_id in existing:
        post_embedding = existing[post_id]
    else:
        if not post_text or not post_text.strip():
            return []
        channel = await ChannelRepository.get_by_id(session, post.channel_id)
        channel_title = channel.title if channel else None
        text = embedding_service.prepare_post_text(post_text, channel_title)
        embeddings = await embedding_service.get_embeddings_batch([text])
        if not embeddings or embeddings[0] is None:
            return []
        post_embedding = embeddings[0]
        await qdrant_service.upsert_post_embeddings_batch([{
            "id": post_id,
            "vector": post_embedding,
            "payload": {"channel_id": post.channel_id},
        }])

    from app.services import taste_cluster_service
    cluster_ids = await taste_cluster_service.get_cluster_ids_matching_post(
        session, post_embedding
    )
    if not cluster_ids:
        return []

    result = await session.execute(
        select(User.telegram_id).join(
            UserChannel,
            (User.id == UserChannel.user_id)
            & (UserChannel.channel_id == post.channel_id)
            & (UserChannel.mailing_enabled == True),
        ).where(
            User.taste_cluster_id.in_(cluster_ids),
            User.is_deleted == False,
            User.status.in_([UserStatus.ACTIVE, UserStatus.TRAINED]),
        ).distinct()
    )
    return [row[0] for row in result.all()]

