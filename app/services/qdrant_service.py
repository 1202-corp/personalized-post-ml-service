"""
Qdrant vector database service.
Handles storage and similarity search for post embeddings.
"""

import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global Qdrant client
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant vector database client.
    
    The client is singleton - reused across all requests for efficiency.
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,  # Connection timeout in seconds
        )
        logger.info(f"Initialized Qdrant client: {settings.qdrant_host}:{settings.qdrant_port}")
    return _qdrant_client


_collection_created = False

async def ensure_collection_exists() -> bool:
    """
    Ensure the post embeddings collection exists in Qdrant.
    Creates it if it doesn't exist.
    
    Uses a global flag to avoid repeated checks.
    """
    global _collection_created
    if _collection_created:
        return True
        
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if settings.qdrant_collection_name not in collection_names:
            client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                f"Created Qdrant collection: {settings.qdrant_collection_name} "
                f"(dimensions: {settings.embedding_dimensions})"
            )
        else:
            logger.debug(f"Qdrant collection already exists: {settings.qdrant_collection_name}")
        
        _collection_created = True
        return True
    except Exception as e:
        logger.error(f"Error ensuring Qdrant collection exists: {e}", exc_info=True)
        return False


async def upsert_post_embedding(
    post_id: int,
    embedding: List[float],
    payload: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Store or update a post embedding in Qdrant.
    
    Args:
        post_id: Unique post identifier (used as point ID)
        embedding: Vector embedding of the post
        payload: Optional metadata (channel_id, user_interactions, etc.)
    """
    try:
        client = get_qdrant_client()
        await ensure_collection_exists()
        
        point = models.PointStruct(
            id=post_id,
            vector=embedding,
            payload=payload or {},
        )
        
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=[point],
        )
        
        return True
    except Exception as e:
        logger.error(f"Error upserting post embedding {post_id}: {e}")
        return False


async def upsert_post_embeddings_batch(
    points: List[Dict[str, Any]]
) -> bool:
    """
    Batch upsert multiple post embeddings.
    
    Args:
        points: List of dicts with 'id', 'vector', and optional 'payload'
    """
    try:
        client = get_qdrant_client()
        await ensure_collection_exists()
        
        qdrant_points = [
            models.PointStruct(
                id=p['id'],
                vector=p['vector'],
                payload=p.get('payload', {}),
            )
            for p in points
        ]
        
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=qdrant_points,
        )
        
        return True
    except Exception as e:
        logger.error(f"Error batch upserting embeddings: {e}")
        return False


async def search_similar_posts(
    query_vector: List[float],
    limit: int = 10,
    score_threshold: float = 0.0,
    filter_conditions: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for posts similar to the query vector.
    
    Args:
        query_vector: The embedding vector to search with
        limit: Maximum number of results
        score_threshold: Minimum similarity score (0-1 for cosine)
        filter_conditions: Optional Qdrant filter conditions
        
    Returns:
        List of dicts with 'id', 'score', and 'payload'
    """
    try:
        client = get_qdrant_client()
        
        # Build filter if conditions provided
        qdrant_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            if must_conditions:
                qdrant_filter = models.Filter(must=must_conditions)
        
        results = client.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )
        
        return [
            {
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload,
            }
            for hit in results
        ]
    except Exception as e:
        logger.error(f"Error searching similar posts: {e}")
        return []


async def get_user_preference_vector(
    liked_embeddings: List[List[float]],
    disliked_embeddings: Optional[List[List[float]]] = None,
) -> Optional[List[float]]:
    """
    Compute user preference vector from liked/disliked post embeddings.
    
    Simple approach: average of liked embeddings minus weighted average of disliked.
    More sophisticated approaches could use learned weights or contrastive methods.
    """
    if not liked_embeddings:
        return None
    
    # Average liked embeddings
    dim = len(liked_embeddings[0])
    preference_vector = [0.0] * dim
    
    for emb in liked_embeddings:
        for i in range(dim):
            preference_vector[i] += emb[i]
    
    for i in range(dim):
        preference_vector[i] /= len(liked_embeddings)
    
    # Optionally subtract disliked embeddings (with lower weight)
    if disliked_embeddings:
        dislike_weight = 0.3
        dislike_avg = [0.0] * dim
        
        for emb in disliked_embeddings:
            for i in range(dim):
                dislike_avg[i] += emb[i]
        
        for i in range(dim):
            dislike_avg[i] /= len(disliked_embeddings)
            preference_vector[i] -= dislike_weight * dislike_avg[i]
    
    # Normalize the vector
    magnitude = sum(x * x for x in preference_vector) ** 0.5
    if magnitude > 0:
        preference_vector = [x / magnitude for x in preference_vector]
    
    return preference_vector


async def get_post_embedding(post_id: int) -> Optional[List[float]]:
    """Get stored embedding for a post."""
    try:
        client = get_qdrant_client()
        
        results = client.retrieve(
            collection_name=settings.qdrant_collection_name,
            ids=[post_id],
            with_vectors=True,
        )
        
        if results:
            return results[0].vector
        return None
    except Exception as e:
        logger.error(f"Error getting post embedding {post_id}: {e}")
        return None


async def get_post_embeddings_batch(post_ids: List[int]) -> Dict[int, List[float]]:
    """Get stored embeddings for multiple posts."""
    if not post_ids:
        return {}
    
    try:
        # Ensure collection exists first
        await ensure_collection_exists()
        
        client = get_qdrant_client()
        
        results = client.retrieve(
            collection_name=settings.qdrant_collection_name,
            ids=post_ids,
            with_vectors=True,
        )
        
        return {point.id: point.vector for point in results}
    except Exception as e:
        logger.error(f"Error getting batch embeddings: {e}")
        return {}


async def delete_post_embedding(post_id: int) -> bool:
    """Delete a post embedding from Qdrant."""
    try:
        client = get_qdrant_client()
        
        client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=models.PointIdsList(points=[post_id]),
        )
        
        return True
    except Exception as e:
        logger.error(f"Error deleting post embedding {post_id}: {e}")
        return False
