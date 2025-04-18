from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
from app.models.insight import Insight
from app.services.insight_service import InsightService

router = APIRouter()
insight_service = InsightService()

@router.get("/", response_model=List[Insight])
async def get_insights(
    channel_id: Optional[str] = None,
    topic: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = Query(10, ge=1, le=50),
    skip: int = Query(0, ge=0)
):
    """Get insights, optionally filtered by channel, topic, and date range"""
    try:
        # Default to last month if no dates provided
        if not from_date:
            from_date = datetime.now() - timedelta(days=30)
        if not to_date:
            to_date = datetime.now()
            
        return insight_service.get_insights(
            channel_id=channel_id,
            topic=topic,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            skip=skip
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{insight_id}", response_model=Insight)
async def get_insight(insight_id: str):
    """Get a specific insight by ID"""
    try:
        insight = insight_service.get_insight(insight_id)
        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")
        return insight
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{video_id}", response_model=List[Insight])
async def get_insights_by_video(video_id: str):
    """Get all insights for a specific video"""
    try:
        insights = insight_service.get_insights_by_video(video_id)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
