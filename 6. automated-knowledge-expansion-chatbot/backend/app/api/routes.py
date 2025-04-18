from fastapi import APIRouter
from app.api.endpoints import channels, videos, insights, chat

router = APIRouter()

router.include_router(channels.router, prefix="/channels", tags=["channels"])
router.include_router(videos.router, prefix="/videos", tags=["videos"])
router.include_router(insights.router, prefix="/insights", tags=["insights"])
router.include_router(chat.router, prefix="/chat", tags=["chat"])
