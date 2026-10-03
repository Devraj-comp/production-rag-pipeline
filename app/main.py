from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import router as api_router
from qdrant_client import QdrantClient
import redis

app = FastAPI(title=settings.PROJECT_NAME)

# Include our API routes
app.include_router(api_router, prefix="/api", tags=["RAG"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Production RAG Pipeline API"}

@app.get("/health")
async def health_check():
    health = {}

    # Check Qdrant
    try:
        q_client = QdrantClient(url=settings.QDRANT_URL)
        q_client.get_collections()
        health["qdrant"] = "healthy"
    except Exception as e:
        health["qdrant"] = f"unhealthy: {str(e)}"

    # Check Redis
    try:
        r_client = redis.from_url(settings.REDIS_URL)
        r_client.ping()
        health["redis"] = "healthy"
    except Exception as e:
        health["redis"] = f"unhealthy: {str(e)}"

    return health
