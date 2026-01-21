"""
Cluster service for grouping posts by embeddings for optimized search.

Uses clustering to group similar posts together, allowing faster search
by filtering through clusters first instead of comparing against all posts.
"""

import logging
from typing import List, Optional, Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.post import Post
from app.repositories.post_repository import PostRepository
from app.services.qdrant_service import get_post_embeddings_batch
from app.config import get_settings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.warning("numpy not available, clustering centroid calculation will be limited")

logger = logging.getLogger(__name__)
settings = get_settings()

# Default clustering parameters
DEFAULT_N_CLUSTERS = 50  # Number of clusters to create
MIN_POSTS_PER_CLUSTER = 10  # Minimum posts needed to form a cluster
CLUSTER_SIMILARITY_THRESHOLD = 0.7  # Minimum cosine similarity to assign post to cluster


async def get_cluster_centroids(
    session: AsyncSession,
    post_ids: List[int]
) -> Dict[int, List[float]]:
    """
    Get cluster centroids (average embeddings) for given posts.
    Returns dict mapping cluster_id to centroid vector.
    """
    # Get all posts with their cluster_ids
    posts = await PostRepository.get_by_ids(session, post_ids)
    if not posts:
        return {}
    
    cluster_embeddings: Dict[int, List[List[float]]] = {}
    
    # Get embeddings for all posts
    post_embeddings = await get_post_embeddings_batch(post_ids)
    
    for post in posts:
        if post.cluster_id is not None and post.id in post_embeddings:
            if post.cluster_id not in cluster_embeddings:
                cluster_embeddings[post.cluster_id] = []
            cluster_embeddings[post.cluster_id].append(post_embeddings[post.id])
    
    # Calculate centroids (average vectors)
    centroids = {}
    for cluster_id, embeddings in cluster_embeddings.items():
        if embeddings:
            # Average all embeddings in cluster
            if HAS_NUMPY:
                centroid = np.mean(embeddings, axis=0).tolist()
            else:
                # Manual calculation if numpy not available
                dim = len(embeddings[0])
                centroid = [0.0] * dim
                for emb in embeddings:
                    for i in range(dim):
                        centroid[i] += emb[i]
                centroid = [x / len(embeddings) for x in centroid]
            centroids[cluster_id] = centroid
    
    return centroids


async def find_similar_clusters(
    query_vector: List[float],
    cluster_centroids: Dict[int, List[float]],
    top_k: int = 5,
    similarity_threshold: float = 0.5
) -> List[Tuple[int, float]]:
    """
    Find clusters most similar to query vector.
    
    Returns list of (cluster_id, similarity_score) tuples sorted by similarity.
    """
    if not cluster_centroids:
        return []
    
    similarities = []
    for cluster_id, centroid in cluster_centroids.items():
        similarity = _cosine_similarity(query_vector, centroid)
        if similarity >= similarity_threshold:
            similarities.append((cluster_id, similarity))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


async def get_posts_from_clusters(
    session: AsyncSession,
    cluster_ids: List[int],
    limit: Optional[int] = None
) -> List[Post]:
    """
    Get posts belonging to specified clusters.
    """
    if not cluster_ids:
        return []
    
    query = (
        select(Post)
        .where(
            Post.cluster_id.in_(cluster_ids),
            Post.is_deleted == False
        )
        .order_by(Post.posted_at.desc())
    )
    
    if limit:
        query = query.limit(limit)
    
    from sqlalchemy.ext.asyncio import AsyncSession as AS
    result = await session.execute(query)
    return list(result.scalars().all())


async def assign_post_to_cluster(
    session: AsyncSession,
    post_id: int,
    cluster_id: int
) -> bool:
    """Assign a post to a cluster."""
    try:
        post = await PostRepository.get_by_id(session, post_id)
        if not post:
            return False
        
        post.cluster_id = cluster_id
        await session.flush()
        return True
    except Exception as e:
        logger.error(f"Error assigning post {post_id} to cluster {cluster_id}: {e}")
        return False


async def recalculate_clusters(
    session: AsyncSession,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_posts: int = MIN_POSTS_PER_CLUSTER
) -> Dict[str, int]:
    """
    Recalculate clusters for all posts using K-means-like approach.
    
    Returns statistics about the clustering operation.
    """
    from sklearn.cluster import KMeans
    
    # Get all posts with embeddings
    all_posts = await PostRepository.get_all(session)
    
    if len(all_posts) < min_posts:
        logger.warning(f"Not enough posts for clustering: {len(posts_with_embeddings)} < {min_posts}")
        return {
            "status": "insufficient_posts",
            "total_posts": len(all_posts),
            "clusters_created": 0
        }
    
    post_ids = [p.id for p in posts_with_embeddings]
    embeddings_dict = await get_post_embeddings_batch(post_ids)
    
    # Filter posts that have embeddings
    valid_posts = []
    valid_embeddings = []
    for post in posts_with_embeddings:
        if post.id in embeddings_dict:
            valid_posts.append(post)
            valid_embeddings.append(embeddings_dict[post.id])
    
    if len(valid_posts) < min_posts:
        logger.warning(f"Not enough posts with embeddings: {len(valid_posts)} < {min_posts}")
        return {
            "status": "insufficient_embeddings",
            "total_posts": len(valid_posts),
            "clusters_created": 0
        }
    
    # Limit number of clusters to number of posts
    actual_n_clusters = min(n_clusters, len(valid_posts))
    
    try:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(valid_embeddings)
        
        # Assign cluster IDs to posts
        clusters_created = set()
        for post, cluster_label in zip(valid_posts, cluster_labels):
            cluster_id = int(cluster_label)
            post.cluster_id = cluster_id
            clusters_created.add(cluster_id)
        
        await session.flush()
        
        logger.info(
            f"Clustering completed: {len(valid_posts)} posts assigned to {len(clusters_created)} clusters"
        )
        
        return {
            "status": "success",
            "total_posts": len(valid_posts),
            "clusters_created": len(clusters_created),
            "posts_clustered": len(valid_posts)
        }
        
    except Exception as e:
        logger.error(f"Error during clustering: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "total_posts": len(valid_posts),
            "clusters_created": 0
        }

