from fastapi import APIRouter, HTTPException, Body
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    """Process a chat request and return a response"""
    try:
        response = chat_service.process_query(
            query=request.query,
            channel_id=request.channel_id,
            topic=request.topic,
            from_date=request.from_date,
            to_date=request.to_date
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
