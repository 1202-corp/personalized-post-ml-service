# ML Service

ML and recommendation service for personalized post recommendations.

## Overview

This service handles all machine learning operations including:
- Post embedding generation
- User preference vector computation
- Post clustering for optimized search
- Recommendation generation using vector similarity
- LLM-based reranking

## API Endpoints

### ML Operations
- `POST /api/v1/ml/train` - Train ML model for a user
- `POST /api/v1/ml/predict` - Get relevance predictions for posts
- `POST /api/v1/ml/recommendations` - Get personalized recommendations
- `GET /api/v1/ml/eligibility/{telegram_id}` - Check training eligibility

### Cluster Management
- `POST /api/v1/clusters/recalculate` - Recalculate post clusters
- `GET /api/v1/clusters/stats` - Get cluster statistics

## Configuration

Environment variables:
- `DATABASE_URL` - PostgreSQL connection string (read-write access)
- `OPENAI_API_BASE` - OpenAI-compatible API base URL
- `OPENAI_API_KEY` - API key for embeddings
- `QDRANT_HOST` - Qdrant vector database host
- `QDRANT_PORT` - Qdrant port

## Architecture

- Uses PostgreSQL for reading/writing user data and posts
- Uses Qdrant for storing and searching post embeddings
- Generates embeddings via OpenAI-compatible API
- Implements clustering for optimized search performance

