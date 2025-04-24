from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.channel import Channel, ChannelCreate
from app.services.youtube_service import YouTubeService

router = APIRouter()
youtube_service = YouTubeService()

@router.get("/", response_model=List[Channel])
async def get_channels():
    """Get all monitored YouTube channels"""
    try:
        return youtube_service.get_all_channels()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=Channel)
async def add_channel(channel: ChannelCreate):
    """Add a new YouTube channel to monitor"""
    try:
        return youtube_service.add_channel(channel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{channel_id}", response_model=Channel)
async def get_channel(channel_id: str):
    """Get a specific YouTube channel by ID"""
    try:
        channel = youtube_service.get_channel(channel_id)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
        return channel
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{channel_id}")
async def delete_channel(channel_id: str):
    """Remove a YouTube channel from monitoring"""
    try:
        success = youtube_service.delete_channel(channel_id)
        if not success:
            raise HTTPException(status_code=404, detail="Channel not found")
        return {"message": "Channel deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
