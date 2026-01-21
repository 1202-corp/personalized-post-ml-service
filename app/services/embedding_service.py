"""
Embedding service using OpenAI-compatible API.
Generates text embeddings for posts.
"""

import logging
import httpx
from typing import List, Optional
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# HTTP client for API calls
_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Get or create HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=60.0)
    return _http_client


def get_api_base() -> str:
    """Get the correct API base URL, fixing known issues."""
    base = settings.openai_api_base.rstrip("/")
    
    # Fix bothub.chat URL if using old format
    if "bothub.chat/api/v1" in base and "/v2/openai" not in base:
        base = base.replace("bothub.chat/api/v1", "bothub.chat/api/v2/openai/v1")
        logger.info(f"Auto-corrected API base URL to: {base}")
    
    return base


async def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding for a single text.
    Returns vector of floats or None if failed.
    """
    if not text or not text.strip():
        return None
    
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured, returning None")
        return None
    
    try:
        client = get_http_client()
        
        # Truncate text if too long
        text = text[:8000]
        
        # Call OpenAI-compatible embeddings endpoint
        api_base = get_api_base()
        response = await client.post(
            f"{api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.embedding_model,
                "input": text,
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Embedding API error {response.status_code}: {response.text[:500]}")
            return None
        
        data = response.json()
        return data["data"][0]["embedding"]
        
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None


async def get_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Get embeddings for multiple texts in batch.
    More efficient than calling get_embedding multiple times.
    """
    if not texts:
        return []
    
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured, returning empty embeddings")
        return [None] * len(texts)
    
    try:
        client = get_http_client()
        
        # Filter and truncate texts
        processed_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                processed_texts.append(text[:8000])
                valid_indices.append(i)
        
        if not processed_texts:
            return [None] * len(texts)
        
        # Call OpenAI-compatible embeddings endpoint
        api_base = get_api_base()
        response = await client.post(
            f"{api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.embedding_model,
                "input": processed_texts,
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Embedding API error {response.status_code}: {response.text[:500]}")
            return [None] * len(texts)
        
        data = response.json()
        
        # Map results back to original indices
        results = [None] * len(texts)
        for idx, embedding_data in enumerate(data["data"]):
            original_idx = valid_indices[idx]
            results[original_idx] = embedding_data["embedding"]
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting batch embeddings: {e}")
        return [None] * len(texts)


def prepare_post_text(text: str, channel_title: Optional[str] = None) -> str:
    """
    Prepare post text for embedding.
    Optionally prepends channel title for context.
    """
    if channel_title:
        return f"[{channel_title}] {text}"
    return text
