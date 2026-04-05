from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check")
async def health_check() -> dict:
    """
    Returns server status and current UTC timestamp.
    Use this endpoint for load balancer / Docker healthchecks.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "rag-parsing-server",
        "version": "1.0.0",
    }
