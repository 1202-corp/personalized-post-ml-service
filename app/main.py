"""ML Service FastAPI application."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import close_db
from app.config import get_settings
from app.logging_config import setup_logging, get_logger
from app.routers import ml, clusters

# Configure logging
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "/var/log/ppb"),
    log_file="ml-service.log",
)
logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("ML Service starting up...")
    yield
    # Shutdown
    await close_db()
    logger.info("ML Service shutting down...")


app = FastAPI(
    title="Personalized Post Bot - ML Service",
    description="ML and recommendation service for personalized post recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ml.router, prefix="/api/v1")
app.include_router(clusters.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "ml-service"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check - verifies all dependencies are available."""
    from app.services.qdrant_service import get_qdrant_client
    from app.database import async_session_maker
    from sqlalchemy import text
    
    checks = {
        "service": "ml-service",
        "postgres": "unknown",
        "qdrant": "unknown",
    }
    all_healthy = True
    
    # Check PostgreSQL
    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "healthy"
    except Exception as e:
        checks["postgres"] = f"unhealthy: {str(e)[:50]}"
        all_healthy = False
    
    # Check Qdrant
    try:
        client = get_qdrant_client()
        client.get_collections()
        checks["qdrant"] = "healthy"
    except Exception as e:
        checks["qdrant"] = f"unhealthy: {str(e)[:50]}"
        all_healthy = False
    
    checks["status"] = "healthy" if all_healthy else "degraded"
    return checks


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Personalized Post Bot - ML Service",
        "docs": "/docs",
        "health": "/health"
    }

