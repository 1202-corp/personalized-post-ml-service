"""
ML Service - Real implementation using embeddings and Qdrant vector search.
"""

import logging
import time
import random
from typing import List, Dict, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
import httpx

from app.config import get_settings
from app.services import embedding_service, qdrant_service
from app.repositories.user_repository import UserRepository
from app.repositories.user_preference_vector_repository import UserPreferenceVectorRepository
from app.repositories.user_channel_preference_vector_repository import UserChannelPreferenceVectorRepository
from app.repositories.user_channel_taste_repository import UserChannelTasteRepository
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
        # Per-channel: group interactions by channel, compute vector per channel, assign to cluster per channel
        by_channel = await _get_user_interaction_posts_by_channel(session, user.id)
        all_posts = []
        for liked, disliked in by_channel.values():
            all_posts.extend(liked)
            all_posts.extend(disliked)
        await _ensure_post_embeddings(session, all_posts)

        from app.services import taste_cluster_service
        for channel_id, (liked_posts, disliked_posts) in by_channel.items():
            if not liked_posts:
                continue
            liked_embeddings = await _get_embeddings_for_posts([p.id for p in liked_posts])
            disliked_embeddings = await _get_embeddings_for_posts([p.id for p in disliked_posts])
            preference_vector = await qdrant_service.get_user_preference_vector(
                liked_embeddings,
                disliked_embeddings if disliked_embeddings else None,
            )
            if not preference_vector:
                continue
            await UserChannelPreferenceVectorRepository.upsert(
                session, user.id, channel_id, preference_vector
            )
            await taste_cluster_service.assign_user_to_nearest_cluster(
                session, user.id, preference_vector, channel_id=channel_id
            )
        await session.flush()

        # Recalculate taste clusters per channel (so cluster centroids are up to date)
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
    post_ids: List[int],
    channel_id: Optional[int] = None,
) -> Dict[int, float]:
    """
    Get relevance predictions for specific posts.
    Uses user's preference vector per channel and post embeddings.
    If channel_id is given, uses only that channel's vector; otherwise for each post uses post.channel_id.
    """
    try:
        user = await UserRepository.get_by_telegram_id(session, user_telegram_id)
        if not user:
            return {}

        posts = []
        for post_id in post_ids:
            post = await PostRepository.get_by_id(session, post_id)
            if post:
                posts.append(post)
        await _ensure_post_embeddings(session, posts)
        post_embeddings = await qdrant_service.get_post_embeddings_batch(post_ids)

        predictions = {}
        for post_id in post_ids:
            if post_id not in post_embeddings:
                predictions[post_id] = 0.0
                continue
            post = next((p for p in posts if p.id == post_id), None)
            cid = channel_id if channel_id is not None else (post.channel_id if post else None)
            if cid is None:
                predictions[post_id] = 0.0
                continue
            pv_row = await UserChannelPreferenceVectorRepository.get_by_user_and_channel(
                session, user.id, cid
            )
            preference_vector = pv_row.preference_vector if pv_row else None
            if not preference_vector:
                predictions[post_id] = 0.0
                continue
            score = _cosine_similarity(preference_vector, post_embeddings[post_id])
            predictions[post_id] = round(score, 4)

        return predictions

    except Exception as e:
        await session.rollback()
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return {pid: 0.0 for pid in post_ids}


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
        
        # Prefer per-channel vectors: average across user's channels for feed search
        from app.models.user_channel_preference_vector import UserChannelPreferenceVector
        result = await session.execute(
            select(UserChannelPreferenceVector.preference_vector).where(
                UserChannelPreferenceVector.user_id == user.id,
                UserChannelPreferenceVector.preference_vector.isnot(None),
            )
        )
        channel_vectors = [row[0] for row in result.all() if row[0] and isinstance(row[0], list) and len(row[0]) > 0]
        if channel_vectors:
            dim = len(channel_vectors[0])
            avg = [0.0] * dim
            for v in channel_vectors:
                for i in range(dim):
                    avg[i] += v[i]
            n = len(channel_vectors)
            avg = [x / n for x in avg]
            mag = sum(x * x for x in avg) ** 0.5
            preference_vector = [x / mag for x in avg] if mag > 0 else None
        else:
            preference_vector = None
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


async def _get_user_interaction_posts_by_channel(
    session: AsyncSession, user_id: int
) -> Dict[int, tuple[List[Post], List[Post]]]:
    """Get user's liked and disliked posts grouped by channel_id. Returns {channel_id: (liked_posts, disliked_posts)}."""
    liked_posts, disliked_posts = await _get_user_interaction_posts(session, user_id)
    by_channel: Dict[int, tuple[List[Post], List[Post]]] = {}
    for post in liked_posts + disliked_posts:
        cid = post.channel_id
        if cid not in by_channel:
            by_channel[cid] = ([], [])
        liked, disliked = by_channel[cid]
        if post in liked_posts:
            liked.append(post)
        else:
            disliked.append(post)
    return by_channel


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
    After every 2 reactions (like/dislike), recalc user preference vector per channel and assign to nearest cluster per channel.
    Returns True if recalculated.
    """
    user = await UserRepository.get_by_telegram_id(session, user_telegram_id)
    if not user:
        return False
    interactions = await InteractionRepository.get_by_user_id(session, user.id)
    count = len(interactions)
    if count < 2 or count % 2 != 0:
        return False
    by_channel = await _get_user_interaction_posts_by_channel(session, user.id)
    all_posts = []
    for liked, disliked in by_channel.values():
        all_posts.extend(liked)
        all_posts.extend(disliked)
    await _ensure_post_embeddings(session, all_posts)

    from app.services import taste_cluster_service
    recalculated = False
    for channel_id, (liked_posts, disliked_posts) in by_channel.items():
        if not liked_posts:
            continue
        liked_embeddings = await _get_embeddings_for_posts([p.id for p in liked_posts])
        disliked_embeddings = await _get_embeddings_for_posts([p.id for p in disliked_posts])
        preference_vector = await qdrant_service.get_user_preference_vector(
            liked_embeddings,
            disliked_embeddings if disliked_embeddings else None,
        )
        if not preference_vector:
            continue
        await UserChannelPreferenceVectorRepository.upsert(
            session, user.id, channel_id, preference_vector
        )
        await taste_cluster_service.assign_user_to_nearest_cluster(
            session, user.id, preference_vector, channel_id=channel_id
        )
        recalculated = True
    await session.flush()
    if recalculated:
        logger.info(f"Taste recalculated for user {user_telegram_id} after {count} interactions (per channel)")
    return recalculated


async def get_post_recipients(
    session: AsyncSession,
    post_id: int,
    post_text: Optional[str] = None,
) -> List[int]:
    """
    Post-centric delivery: return telegram_ids of users who should receive this post.
    Users must be in a taste cluster matching the post embedding and have mailing_enabled for this channel.
    """
    from app.services import taste_cluster_service
    from app.models.interaction import Interaction
    from app.services.utils import cosine_similarity
    from app.repositories.taste_cluster_repository import TasteClusterRepository
    
    logger.info(f"[POST_RECIPIENTS] === НАЧАЛО обработки post_id={post_id}, post_text={post_text[:100] if post_text else None!r}")
    
    post = await PostRepository.get_by_id(session, post_id)
    if not post:
        logger.warning(f"[POST_RECIPIENTS] post_id={post_id}: пост не найден в БД")
        return []

    # Get or create post embedding
    existing = await qdrant_service.get_post_embeddings_batch([post_id])
    if post_id in existing:
        post_embedding = existing[post_id]
        logger.info(f"[POST_RECIPIENTS] post_id={post_id}: эмбеддинг найден в Qdrant")
    else:
        if not post_text or not post_text.strip():
            logger.warning(f"[POST_RECIPIENTS] post_id={post_id}: post_text отсутствует или пустой, невозможно создать эмбеддинг")
            return []
        logger.info(f"[POST_RECIPIENTS] post_id={post_id}: создаю новый эмбеддинг из текста")
        channel = await ChannelRepository.get_by_id(session, post.channel_id)
        channel_title = channel.title if channel else None
        text = embedding_service.prepare_post_text(post_text, channel_title)
        embeddings = await embedding_service.get_embeddings_batch([text])
        if not embeddings or embeddings[0] is None:
            logger.error(f"[POST_RECIPIENTS] post_id={post_id}: не удалось получить эмбеддинг от embedding_service")
            return []
        post_embedding = embeddings[0]
        await qdrant_service.upsert_post_embeddings_batch([{
            "id": post_id,
            "vector": post_embedding,
            "payload": {"channel_id": post.channel_id},
        }])
        logger.info(f"[POST_RECIPIENTS] post_id={post_id}: эмбеддинг создан и сохранен в Qdrant")

    # Логирование: текст поста
    display_text = (post_text or "").strip()[:200]  # Первые 200 символов
    logger.info(f"[POST_RECIPIENTS] post_id={post_id}, text={display_text!r}")

    # Получаем все кластеры с их cosine scores
    # Для коротких текстов cosine similarity с кластером может быть низкой,
    # но predict score может быть высоким. Поэтому используем более мягкий порог для кластеров
    # или проверяем все кластеры с пользователями через predict score
    similarity_threshold = settings.taste_cluster_similarity_threshold
    channel_id = post.channel_id
    clusters = await TasteClusterRepository.get_all(session, channel_id=channel_id)
    
    logger.info(f"[POST_RECIPIENTS] post_id={post_id}: channel_id={channel_id}, найдено кластеров в БД: {len(clusters)}, порог similarity: {similarity_threshold}")
    
    # Собираем все кластеры с их scores (даже если они ниже порога)
    # Predict score будет финальным фильтром
    all_cluster_scores = []
    cluster_ids_to_check = []
    for c in clusters:
        if not c.centroid or len(c.centroid) != len(post_embedding):
            logger.debug(f"[POST_RECIPIENTS] post_id={post_id}: кластер {c.id} пропущен (нет centroid или размер не совпадает: {len(c.centroid) if c.centroid else 0} vs {len(post_embedding)})")
            continue
        cosine = cosine_similarity(post_embedding, c.centroid)
        all_cluster_scores.append((c.id, cosine, c.user_count))
        # Используем более мягкий порог для кластеров: 0.26 вместо 0.5
        # Финальная фильтрация будет по predict score
        if cosine >= 0.26:  # Мягкий порог для кластеров
            cluster_ids_to_check.append(c.id)
    
    # Логируем все scores для отладки
    if all_cluster_scores:
        scores_info = ", ".join([f"cluster_{cid}(cosine={cos:.4f}, users={ucnt})" for cid, cos, ucnt in all_cluster_scores])
        logger.info(f"[POST_RECIPIENTS] post_id={post_id}: все кластеры и их cosine scores: {scores_info}")
    
    if not cluster_ids_to_check:
        # Нет кластеров выше мягкого порога 0.26. Однако есть идея «мутации»:
        # с небольшим шансом (например, 20%) всё же отправлять пост,
        # если он хоть как‑то похож на кластер (cosine > 0).
        positive_clusters = [c for c in all_cluster_scores if c[1] > 0.0]
        if not positive_clusters:
            logger.info(f"[POST_RECIPIENTS] post_id={post_id}: нет кластеров даже с cosine>0, ничего не отправляем")
            return []
        
        mutation_prob = 0.20
        roll = random.random()
        if roll >= mutation_prob:
            logger.info(
                f"[POST_RECIPIENTS] post_id={post_id}: все кластеры имеют cosine < 0.26, "
                f"cosine>0 есть, но мутация не сработала (roll={roll:.3f} >= {mutation_prob})"
            )
            return []

        # Мутация сработала: выбираем все кластеры с cosine>0 и шлём пост всем их пользователям
        soft_cluster_ids = [cid for cid, cos, _ in positive_clusters]
        logger.info(
            f"[POST_RECIPIENTS] post_id={post_id}: МУТАЦИЯ АКТИВНА (p={mutation_prob}), "
            f"рассылаем пост кластерам с cosine>0: {soft_cluster_ids}"
        )
        cluster_ids_to_check = soft_cluster_ids
    
    # Логирование: кластеры, которые будем проверять
    matching_info = ", ".join([f"cluster_{cid}" for cid in cluster_ids_to_check])
    logger.info(f"[POST_RECIPIENTS] post_id={post_id}: проверяем кластеры (cosine >= 0.26): {matching_info}")

    # Candidates: users in these clusters for this channel (UserChannelTaste) + mailing_enabled + not interacted with this post
    candidate_user_ids = await UserChannelTasteRepository.get_user_ids_in_clusters(
        session, channel_id, cluster_ids_to_check
    )
    if not candidate_user_ids:
        logger.info(f"[POST_RECIPIENTS] post_id={post_id}: нет кандидатов после фильтрации по clusters (channel_id={channel_id})")
        return []
    result = await session.execute(
        select(User.telegram_id, User.id)
        .join(
            UserChannel,
            (User.id == UserChannel.user_id)
            & (UserChannel.channel_id == post.channel_id)
            & (UserChannel.mailing_enabled == True),
        )
        .outerjoin(
            Interaction,
            (User.id == Interaction.user_id) & (Interaction.post_id == post_id)
        )
        .where(
            User.id.in_(candidate_user_ids),
            User.is_deleted == False,
            User.status.in_([UserStatus.ACTIVE, UserStatus.TRAINED]),
            Interaction.id.is_(None),
        )
        .distinct()
    )
    candidate_users = result.all()
    if not candidate_users:
        logger.info(f"[POST_RECIPIENTS] post_id={post_id}: нет кандидатов после фильтрации по interactions")
        return []
    
    # Логирование: пользователи в кластерах (до фильтрации по predict)
    candidate_tg_ids = [tg_id for tg_id, _ in candidate_users]
    logger.info(f"[POST_RECIPIENTS] post_id={post_id}: кандидаты из кластеров (до predict): {len(candidate_tg_ids)} пользователей: {candidate_tg_ids[:10]}{'...' if len(candidate_tg_ids) > 10 else ''}")
    
    # Additional filter: use predict score to ensure post is relevant for user
    # This catches cases where taste cluster match is too broad (e.g. similar short phrases)
    # Even if cluster matches, we require predict(user, post) >= threshold
    predict_threshold = settings.post_recipient_predict_threshold
    
    recipients = []
    user_ids_by_telegram = {tg_id: user_id for tg_id, user_id in candidate_users}
    user_scores = {}
    
    # Check predict score for each user (per-channel vector; filters out posts that don't match user taste for this channel)
    for tg_id, user_id in user_ids_by_telegram.items():
        user_predictions = await predict(
            session, tg_id, [post_id], channel_id=channel_id
        )
        score = user_predictions.get(post_id, 0.0)
        user_scores[tg_id] = score
        if score >= predict_threshold:
            recipients.append(tg_id)
    
    # Логирование: финальный список получателей с их predict scores
    if recipients:
        recipient_info = ", ".join([f"user_{tg_id}(predict={user_scores[tg_id]:.4f})" for tg_id in recipients[:10]])
        if len(recipients) > 10:
            recipient_info += f", ... (всего {len(recipients)})"
        logger.info(f"[POST_RECIPIENTS] post_id={post_id}: финальные получатели (после predict>={predict_threshold}): {recipient_info}")
    else:
        logger.info(f"[POST_RECIPIENTS] post_id={post_id}: нет получателей после фильтрации по predict (порог {predict_threshold})")
        # Логируем несколько примеров низких scores
        if user_scores:
            low_scores = sorted(user_scores.items(), key=lambda x: x[1])[:5]
            low_info = ", ".join([f"user_{tg_id}({score:.4f})" for tg_id, score in low_scores])
            logger.info(f"[POST_RECIPIENTS] post_id={post_id}: примеры низких predict scores: {low_info}")
    
    return recipients

