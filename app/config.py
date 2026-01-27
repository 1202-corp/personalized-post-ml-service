"""ML Service configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # PostgreSQL database (read-only access to main DB)
    database_url: str = "postgresql+asyncpg://ppp_user:ppp_secret@postgres:5432/ppp_db"
    
    # App settings
    debug: bool = False
    
    # Embedding generation settings
    use_local_embeddings: bool = False  # Use local model instead of cloud API
    # Default uses Docker service name (ollama) - change to localhost if running outside Docker
    local_embedding_model_url: str = "http://ollama:11434/api/embeddings"
    local_embedding_model: str = "all-minilm"  # Model name for local API
    
    # Cloud API settings (when use_local_embeddings=False)
    openai_api_base: str = "https://bothub.chat/api/v2/openai/v1"
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536  # text-embedding-ada-002 dimension
    
    # Qdrant vector database settings
    qdrant_host: str = "vector-db-qdrant"  # Docker service name for vector database
    qdrant_port: int = 6333
    qdrant_collection_name: str = "post_embeddings"
    qdrant_timeout: int = 30  # Connection timeout in seconds
    
    # Embedding settings
    max_text_length: int = 8000  # Maximum text length for embedding generation
    embedding_timeout: float = 60.0  # HTTP timeout for embedding API requests (seconds)
    error_text_limit: int = 500  # Maximum error text length in logs
    
    # ML Training settings
    default_similarity_threshold: float = 0.5  # Default similarity threshold for recommendations
    default_score_threshold: float = 0.3  # Default score threshold for Qdrant search
    cluster_search_multiplier: int = 3  # Multiplier for expanding cluster search
    cluster_search_max_multiplier: int = 5  # Maximum multiplier for cluster search
    
    # Clustering settings
    default_n_clusters: int = 50  # Default number of clusters to create
    min_posts_per_cluster: int = 10  # Minimum posts needed to form a cluster
    cluster_similarity_threshold: float = 0.7  # Minimum cosine similarity to assign post to cluster
    kmeans_random_state: int = 42  # Random state for K-Means (for reproducibility)
    kmeans_n_init: int = 10  # Number of K-Means initialization attempts
    
    # Qdrant search settings
    dislike_weight: float = 0.3  # Weight for dislikes in user preference vector calculation
    
    # LLM Reranker settings
    llm_timeout: float = 30.0  # HTTP timeout for LLM reranker requests (seconds)
    llm_temperature: float = 0.3  # Temperature for LLM reranker
    llm_max_tokens: int = 100  # Maximum tokens for LLM reranker output
    cost_per_1k_input_tokens: float = 0.00015  # Cost per 1K input tokens (USD)
    cost_per_1k_output_tokens: float = 0.0006  # Cost per 1K output tokens (USD)
    
    # Database connection pool settings
    db_pool_size: int = 10  # Database connection pool size
    db_max_overflow: int = 20  # Maximum overflow connections in pool
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()

