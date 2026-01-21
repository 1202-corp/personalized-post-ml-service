"""ML Service configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # PostgreSQL database (read-only access to main DB)
    database_url: str = "postgresql+asyncpg://ppb_user:ppb_secret@postgres:5432/ppb_db"
    
    # App settings
    debug: bool = False
    
    # OpenAI-compatible API settings
    openai_api_base: str = "https://bothub.chat/api/v2/openai/v1"
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536  # text-embedding-ada-002 dimension
    
    # Qdrant vector database settings
    qdrant_host: str = "vector-db-qdrant"  # Docker service name for vector database
    qdrant_port: int = 6333
    qdrant_collection_name: str = "post_embeddings"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()

