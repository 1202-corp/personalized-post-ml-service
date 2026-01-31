"""
Taste cluster service: cluster users by preference vector per channel for post-centric delivery.
max_cluster_size = ceil(N * 0.017); K = max(K_min, ceil(N / max_size)); postprocess to enforce max_size.
"""

import logging
import math
from typing import List, Optional, Dict, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.taste_cluster import TasteCluster
from app.repositories.taste_cluster_repository import TasteClusterRepository
from app.repositories.user_channel_preference_vector_repository import UserChannelPreferenceVectorRepository
from app.repositories.user_channel_taste_repository import UserChannelTasteRepository
from app.config import get_settings

try:
    import numpy as np
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)
settings = get_settings()


async def get_users_with_preference_vectors_for_channel(
    session: AsyncSession, channel_id: int
) -> List[Tuple[int, List[float]]]:
    """Get (user_id, preference_vector) for all users that have a vector for this channel."""
    return await UserChannelPreferenceVectorRepository.get_vectors_by_channel(
        session, channel_id
    )


def _compute_centroid(vectors: List[List[float]]) -> List[float]:
    """Average of vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    centroid = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            centroid[i] += v[i]
    n = len(vectors)
    return [x / n for x in centroid]


async def recalculate_taste_clusters_for_channel(
    session: AsyncSession, channel_id: int
) -> Dict[str, int]:
    """
    Recalculate taste clusters for one channel. Users with preference vector for this channel are clustered.
    """
    if not HAS_SKLEARN:
        logger.error("sklearn not available, cannot recalculate taste clusters")
        return {"status": "error", "error": "sklearn not available", "clusters_created": 0}

    users_with_vectors = await get_users_with_preference_vectors_for_channel(
        session, channel_id
    )
    N = len(users_with_vectors)
    if N < 1:
        return {"status": "no_users", "total_users": 0, "clusters_created": 0}

    user_ids = [uid for uid, _ in users_with_vectors]
    vectors = [v for _, v in users_with_vectors]

    # Single user: one cluster
    if N == 1:
        await TasteClusterRepository.delete_all(session, channel_id=channel_id)
        cluster = await TasteClusterRepository.create(
            session, centroid=vectors[0], user_count=1, channel_id=channel_id
        )
        await UserChannelTasteRepository.upsert(
            session, user_ids[0], channel_id, cluster.id
        )
        await session.flush()
        logger.info(f"Taste clustering channel_id={channel_id}: 1 user -> 1 cluster")
        return {"status": "success", "total_users": 1, "clusters_created": 1}

    max_size = max(1, math.ceil(N * settings.max_cluster_size_ratio))
    K = min(math.ceil(N / max_size), N)
    vectors_array = np.array(vectors, dtype=np.float64)

    try:
        kmeans = KMeans(
            n_clusters=K,
            random_state=settings.kmeans_random_state,
            n_init=settings.kmeans_n_init,
        )
        labels = kmeans.fit_predict(vectors_array)
    except Exception as e:
        logger.error(f"K-means failed for channel_id={channel_id}: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "clusters_created": 0}

    cluster_to_user_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        cid = int(label)
        if cid not in cluster_to_user_indices:
            cluster_to_user_indices[cid] = []
        cluster_to_user_indices[cid].append(idx)

    final_clusters: List[List[int]] = []
    for indices in cluster_to_user_indices.values():
        if len(indices) <= max_size:
            final_clusters.append(indices)
            continue
        sub_vectors = vectors_array[indices]
        try:
            sub_kmeans = KMeans(n_clusters=2, random_state=settings.kmeans_random_state, n_init=3)
            sub_labels = sub_kmeans.fit_predict(sub_vectors)
            part_a = [indices[i] for i in range(len(indices)) if sub_labels[i] == 0]
            part_b = [indices[i] for i in range(len(indices)) if sub_labels[i] == 1]
            final_clusters.append(part_a)
            final_clusters.append(part_b)
        except Exception:
            final_clusters.append(indices)

    await TasteClusterRepository.delete_all(session, channel_id=channel_id)
    await session.flush()

    for indices in final_clusters:
        if not indices:
            continue
        cluster_vectors = [vectors[i] for i in indices]
        centroid = _compute_centroid(cluster_vectors)
        cluster = await TasteClusterRepository.create(
            session,
            centroid=centroid,
            user_count=len(indices),
            channel_id=channel_id,
        )
        for i in indices:
            await UserChannelTasteRepository.upsert(
                session, user_ids[i], channel_id, cluster.id
            )
        cluster.user_count = len(indices)
    await session.flush()

    logger.info(
        f"Taste clustering channel_id={channel_id}: {N} users -> {len(final_clusters)} clusters"
    )
    return {
        "status": "success",
        "total_users": N,
        "clusters_created": len(final_clusters),
        "max_cluster_size": max_size,
    }


async def recalculate_taste_clusters(session: AsyncSession) -> Dict[str, int]:
    """
    Recalculate taste clusters per channel: for each channel that has UserChannelPreferenceVector rows,
    run recalculate_taste_clusters_for_channel.
    """
    from app.models.user_channel_preference_vector import UserChannelPreferenceVector
    result = await session.execute(
        select(UserChannelPreferenceVector.channel_id).distinct()
    )
    channel_ids = [row[0] for row in result.all()]
    if not channel_ids:
        return {"status": "no_channels", "total_channels": 0, "clusters_created": 0}
    total_clusters = 0
    for cid in channel_ids:
        r = await recalculate_taste_clusters_for_channel(session, cid)
        total_clusters += r.get("clusters_created", 0)
    return {
        "status": "success",
        "total_channels": len(channel_ids),
        "clusters_created": total_clusters,
    }


async def assign_user_to_nearest_cluster(
    session: AsyncSession,
    user_id: int,
    preference_vector: List[float],
    channel_id: int,
) -> Optional[int]:
    """
    Assign user to the nearest taste cluster by centroid similarity (cosine).
    Uses per-channel clusters and UserChannelTaste. Returns cluster_id or None if no clusters exist.
    """
    from app.services.utils import cosine_similarity

    clusters = await TasteClusterRepository.get_all(session, channel_id=channel_id)
    if not clusters:
        return None

    best_id = None
    best_sim = -2.0
    for c in clusters:
        if not c.centroid or len(c.centroid) != len(preference_vector):
            continue
        sim = cosine_similarity(preference_vector, c.centroid)
        if sim > best_sim:
            best_sim = sim
            best_id = c.id

    if best_id is None:
        return None

    await UserChannelTasteRepository.upsert(session, user_id, channel_id, best_id)
    await session.flush()
    return best_id


async def get_cluster_ids_matching_post(
    session: AsyncSession,
    post_embedding: List[float],
    similarity_threshold: Optional[float] = None,
    channel_id: Optional[int] = None,
) -> List[int]:
    """
    Return taste cluster IDs whose centroid is similar enough to the post embedding.
    If channel_id is given, only clusters for that channel are considered (per-channel delivery).
    """
    from app.services.utils import cosine_similarity

    if similarity_threshold is None:
        similarity_threshold = settings.taste_cluster_similarity_threshold

    clusters = await TasteClusterRepository.get_all(session, channel_id=channel_id)
    matching = []
    for c in clusters:
        if not c.centroid or len(c.centroid) != len(post_embedding):
            continue
        if cosine_similarity(post_embedding, c.centroid) >= similarity_threshold:
            matching.append(c.id)
    return matching
