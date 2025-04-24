from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.db.database import get_db
from app.core.query_processor import QueryProcessor
from app.core.response_generator import response_generator
from app.core.vector_store import VectorIndex
import logging

logger = logging.getLogger(__name__)

# Initialize vector index
vector_index = VectorIndex()

# Initialize query processor
query_processor = QueryProcessor(vector_index)

router = APIRouter()

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    response: str
    papers: List[Dict[str, Any]]

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Chat endpoint for interacting with the ArXiv Expert Chatbot.
    """
    try:
        # Process the query
        query_result = query_processor.process_query(request.message, db)

        # Prepare follow-up context from conversation history
        follow_up_context = None
        if request.conversation_history:
            follow_up_context = []
            for i in range(0, len(request.conversation_history), 2):
                if i+1 < len(request.conversation_history):
                    user_msg = request.conversation_history[i]
                    assistant_msg = request.conversation_history[i+1]
                    if user_msg.role == 'user' and assistant_msg.role == 'assistant':
                        follow_up_context.append({
                            'query': user_msg.content,
                            'response': assistant_msg.content
                        })

        # Generate response
        response = response_generator.generate_response(query_result, follow_up_context)

        return {
            'response': response['response'],
            'papers': response['papers']
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
