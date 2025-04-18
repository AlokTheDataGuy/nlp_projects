from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.models.video import Video
from app.services.youtube_service import YouTubeService

router = APIRouter()
youtube_service = YouTubeService()

@router.get("/", response_model=List[Video])
async def get_videos(
    channel_id: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    skip: int = Query(0, ge=0)
):
    """Get videos, optionally filtered by channel"""
    try:
        return youtube_service.get_videos(channel_id, limit, skip)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{video_id}", response_model=Video)
async def get_video(video_id: str):
    """Get a specific video by ID"""
    try:
        video = youtube_service.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        return video
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{video_id}/transcript")
async def get_video_transcript(video_id: str):
    """Get the transcript for a specific video"""
    try:
        transcript = youtube_service.get_transcript(video_id)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")
        return {"transcript": transcript}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
