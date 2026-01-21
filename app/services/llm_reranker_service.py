"""
LLM-based reranking service using GPT-4o-mini.
Used as treatment_b variant in A/B testing for improved post recommendations.
"""

import logging
import json
import httpx
from typing import List, Dict, Optional
from datetime import datetime
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Cost tracking (GPT-4o-mini pricing as of 2024)
# Input: $0.15 per 1M tokens, Output: $0.60 per 1M tokens
COST_PER_1K_INPUT_TOKENS = 0.00015
COST_PER_1K_OUTPUT_TOKENS = 0.0006

# Running cost tracker
_total_cost = 0.0
_request_count = 0


def get_cost_stats() -> Dict:
    """Get current cost statistics."""
    return {
        "total_cost_usd": round(_total_cost, 6),
        "request_count": _request_count,
        "avg_cost_per_request": round(_total_cost / _request_count, 6) if _request_count > 0 else 0,
    }


async def rerank_posts_with_llm(
    user_likes: List[str],
    user_dislikes: List[str],
    candidate_posts: List[Dict],
    top_k: int = 5
) -> List[Dict]:
    """
    Rerank candidate posts using GPT-4o-mini based on user preferences.
    
    Args:
        user_likes: List of post texts the user liked
        user_dislikes: List of post texts the user disliked  
        candidate_posts: List of dicts with 'post_id', 'text', 'score' (from embedding)
        top_k: Number of top posts to return
        
    Returns:
        List of reranked posts with LLM scores
    """
    global _total_cost, _request_count
    
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured, returning original order")
        return candidate_posts[:top_k]
    
    if not candidate_posts:
        return []
    
    # Prepare context - limit to avoid token overflow
    likes_sample = user_likes[:5]  # Last 5 likes
    dislikes_sample = user_dislikes[:3]  # Last 3 dislikes
    candidates_sample = candidate_posts[:15]  # Top 15 candidates to rerank
    
    # Build prompt
    likes_text = "\n".join([f"- {text[:200]}" for text in likes_sample]) if likes_sample else "No likes yet"
    dislikes_text = "\n".join([f"- {text[:200]}" for text in dislikes_sample]) if dislikes_sample else "No dislikes"
    
    candidates_text = ""
    for i, post in enumerate(candidates_sample):
        text = post.get('text', '')[:300]
        candidates_text += f"\n[{i}] {text}"
    
    prompt = f"""You are a content recommendation expert. Rank these posts by how much the user will like them.

USER PREFERENCES:
Posts they LIKED:
{likes_text}

Posts they DISLIKED:
{dislikes_text}

CANDIDATE POSTS TO RANK:
{candidates_text}

Return ONLY a JSON array of post indices in order of predicted preference (most liked first).
Example: [3, 0, 7, 1, 4]

Return the top {top_k} indices only."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            api_base = settings.openai_api_base.rstrip("/")
            
            response = await client.post(
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a content recommendation expert. Respond only with JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 100,
                }
            )
            
            if response.status_code != 200:
                logger.error(f"LLM reranker API error {response.status_code}: {response.text[:500]}")
                return candidate_posts[:top_k]
            
            data = response.json()
            
            # Track costs
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            cost = (input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS + 
                   output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)
            _total_cost += cost
            _request_count += 1
            
            logger.info(f"LLM rerank: {input_tokens} in, {output_tokens} out, cost=${cost:.6f}, total=${_total_cost:.4f}")
            
            # Parse response
            content = data["choices"][0]["message"]["content"].strip()
            
            # Try to extract JSON array
            try:
                # Handle potential markdown formatting
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                
                indices = json.loads(content)
                
                if not isinstance(indices, list):
                    raise ValueError("Response is not a list")
                
                # Map indices back to posts
                reranked = []
                for idx in indices[:top_k]:
                    if isinstance(idx, int) and 0 <= idx < len(candidates_sample):
                        post = candidates_sample[idx].copy()
                        post['llm_rank'] = len(reranked) + 1
                        reranked.append(post)
                
                # Fill remaining slots if needed
                if len(reranked) < top_k:
                    used_ids = {p['post_id'] for p in reranked}
                    for post in candidates_sample:
                        if post['post_id'] not in used_ids and len(reranked) < top_k:
                            reranked.append(post)
                
                return reranked
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM response: {e}, content: {content[:200]}")
                return candidate_posts[:top_k]
            
    except Exception as e:
        logger.error(f"LLM reranker failed: {e}")
        return candidate_posts[:top_k]


async def _get_user_interaction_texts(
    session,
    user_id: int,
    interaction_type: str,
    limit: int
) -> List[str]:
    """Get text from user's interactions of specific type."""
    from app.services import post_service
    from app.repositories.interaction_repository import InteractionRepository
    from app.models.interaction import InteractionType
    
    interaction_enum = InteractionType.LIKE if interaction_type == "like" else InteractionType.DISLIKE
    interactions = await InteractionRepository.get_by_user_id(session, user_id)
    filtered_interactions = [
        i for i in interactions 
        if i.interaction_type == interaction_enum
    ][:limit]
    
    texts = []
    for interaction in filtered_interactions:
        post = await post_service.get_post_by_id(session, interaction.post_id)
        if post and post.text:
            texts.append(post.text[:500])
    
    return texts


async def get_reranked_recommendations(
    session,
    user_telegram_id: int,
    candidate_posts: List[Dict],
    limit: int = 5
) -> List[Dict]:
    """
    Get LLM-reranked recommendations for a user.
    
    This is the main entry point for treatment_b users.
    """
    from app.services import user_service
    
    try:
        user = await user_service.get_user_by_telegram_id(session, user_telegram_id)
        if not user:
            return candidate_posts[:limit]
        
        # Get user's liked and disliked post texts
        user_likes = await _get_user_interaction_texts(session, user.id, "like", 10)
        user_dislikes = await _get_user_interaction_texts(session, user.id, "dislike", 5)
        
        # Rerank with LLM
        return await rerank_posts_with_llm(
            user_likes=user_likes,
            user_dislikes=user_dislikes,
            candidate_posts=candidate_posts,
            top_k=limit
        )
        
    except Exception as e:
        logger.error(f"LLM reranking failed for user {user_telegram_id}: {e}")
        return candidate_posts[:limit]
