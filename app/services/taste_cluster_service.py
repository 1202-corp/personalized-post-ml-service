"""
Taste cluster service: cluster users by preference vector for post-centric delivery.
max_cluster_size = ceil(N * 0.017); K = max(K_min, ceil(N / max_size)); postprocess to enforce max_size.
"""

import logging
import math
from typing import List, Optional, Dict, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.user_preference_vector import UserPreferenceVector
from app.models.taste_cluster import TasteCluster
from app.repositories.taste_cluster_repository import TasteClusterRepository
from app.config import get_settings

try:
    import numpy as np
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)
settings = get_settings()


async def get_users_with_preference_vectors(session: AsyncSession) -> List[Tuple[User, List[float]]]:
    """Get all non-deleted users that have a preference vector. Returns list of (User, vector)."""
    result = await session.execute(
        select(User, UserPreferenceVector.preference_vector).join(
            UserPreferenceVector,
            User.id == UserPreferenceVector.user_id,
        ).where(
            User.is_deleted == False,
            UserPreferenceVector.preference_vector.isnot(None),
        )
    )
    rows = result.all()
    out = []
    for user, pv in rows:
        if pv and isinstance(pv, list) and len(pv) > 0:
            out.append((user, pv))
    return out


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


async def recalculate_taste_clusters(session: AsyncSession) -> Dict[str, int]:
    """
    Full recalculate of taste clusters: all users with preference vector are clustered.
    max_size = ceil(N * 0.017); K = max(K_min, ceil(N / max_size)).
    Postprocess: split any cluster exceeding max_size.
    """
    if not HAS_SKLEARN:
        logger.error("sklearn not available, cannot recalculate taste clusters")
        return {"status": "error", "error": "sklearn not available", "clusters_created": 0}

    users_with_vectors = await get_users_with_preference_vectors(session)
    N = len(users_with_vectors)
    if N < 1:
        return {"status": "no_users", "total_users": 0, "clusters_created": 0}

    # Single user: create one cluster with their vector as centroid so they get assigned
    if N == 1:
        user, vec = users_with_vectors[0]
        await TasteClusterRepository.delete_all(session)
        user.taste_cluster_id = None
        await session.flush()
        cluster = await TasteClusterRepository.create(session, centroid=vec, user_count=1)
        user.taste_cluster_id = cluster.id
        cluster.user_count = 1
        await session.flush()
        logger.info("Taste clustering: 1 user -> 1 cluster")
        return {"status": "success", "total_users": 1, "clusters_created": 1, "max_cluster_size": 1}

    max_size = max(1, math.ceil(N * settings.max_cluster_size_ratio))
    K = min(math.ceil(N / max_size), N)  # at most N clusters; with small N, max_size=1 => one cluster per user

    users = [u for u, _ in users_with_vectors]
    vectors = [v for _, v in users_with_vectors]
    vectors_array = np.array(vectors, dtype=np.float64)

    try:
        kmeans = KMeans(
            n_clusters=K,
            random_state=settings.kmeans_random_state,
            n_init=settings.kmeans_n_init,
        )
        labels = kmeans.fit_predict(vectors_array)
    except Exception as e:
        logger.error(f"K-means failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "clusters_created": 0}

    # Group user indices by cluster label
    cluster_to_user_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        cid = int(label)
        if cid not in cluster_to_user_indices:
            cluster_to_user_indices[cid] = []
        cluster_to_user_indices[cid].append(idx)

    # Postprocess: split clusters that exceed max_size
    final_clusters: List[List[int]] = []  # list of user indices per cluster
    for cid, indices in cluster_to_user_indices.items():
        if len(indices) <= max_size:
            final_clusters.append(indices)
            continue
        # Split this cluster with K-means K=2
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

    # If any cluster still exceeds max_size, split again (simple recursive would be better; here we do one more pass)
    expanded = True
    while expanded:
        expanded = False
        new_final: List[List[int]] = []
        for indices in final_clusters:
            if len(indices) <= max_size:
                new_final.append(indices)
                continue
            sub_vectors = vectors_array[indices]
            try:
                sub_kmeans = KMeans(n_clusters=2, random_state=settings.kmeans_random_state + 1, n_init=3)
                sub_labels = sub_kmeans.fit_predict(sub_vectors)
                part_a = [indices[i] for i in range(len(indices)) if sub_labels[i] == 0]
                part_b = [indices[i] for i in range(len(indices)) if sub_labels[i] == 1]
                new_final.append(part_a)
                new_final.append(part_b)
                expanded = True
            except Exception:
                new_final.append(indices)
        final_clusters = new_final

    # Clear existing taste clusters and user assignments
    await TasteClusterRepository.delete_all(session)
    for u in users:
        u.taste_cluster_id = None
    await session.flush()

    # Create TasteCluster rows and assign users
    for indices in final_clusters:
        if not indices:
            continue
        cluster_vectors = [vectors[i] for i in indices]
        centroid = _compute_centroid(cluster_vectors)
        cluster = await TasteClusterRepository.create(session, centroid=centroid, user_count=len(indices))
        for i in indices:
            users[i].taste_cluster_id = cluster.id
        cluster.user_count = len(indices)
    await session.flush()

    logger.info(
        f"Taste clustering completed: {N} users -> {len(final_clusters)} clusters (max_size={max_size})"
    )
    return {
        "status": "success",
        "total_users": N,
        "clusters_created": len(final_clusters),
        "max_cluster_size": max_size,
    }


async def assign_user_to_nearest_cluster(
    session: AsyncSession,
    user_id: int,
    preference_vector: List[float],
) -> Optional[int]:
    """
    Assign user to the nearest taste cluster by centroid similarity (cosine).
    Does not run full recluster; only updates this user's taste_cluster_id.
    Returns cluster_id or None if no clusters exist.
    """
    from app.services.utils import cosine_similarity

    clusters = await TasteClusterRepository.get_all(session)
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

    user = await session.get(User, user_id)
    if user:
        old_id = user.taste_cluster_id
        user.taste_cluster_id = best_id
        await session.flush()
        # Optionally update user_count on old and new cluster (simplified: skip for now; full recalc will fix)
    return best_id


async def get_cluster_ids_matching_post(
    session: AsyncSession,
    post_embedding: List[float],
    similarity_threshold: Optional[float] = None,
) -> List[int]:
    """
    Return taste cluster IDs whose centroid is similar enough to the post embedding.
    Used for post-centric delivery: post -> these clusters -> users in clusters.
    """
    from app.services.utils import cosine_similarity

    if similarity_threshold is None:
        similarity_threshold = settings.taste_cluster_similarity_threshold

    clusters = await TasteClusterRepository.get_all(session)
    matching = []
    for c in clusters:
        if not c.centroid or len(c.centroid) != len(post_embedding):
            continue
        if cosine_similarity(post_embedding, c.centroid) >= similarity_threshold:
            matching.append(c.id)
    return matching
