"""
FastAPI backend for the arXiv Research Assistant.
"""
import os
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from rag.system import RAGSystem
from utils.config import API_HOST, API_PORT, API_WORKERS, FRONTEND_URL
from utils.logger import setup_logger

logger = setup_logger("api", "api.log")

# Initialize FastAPI app
app = FastAPI(
    title="arXiv Research Assistant API",
    description="API for the arXiv Research Assistant",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None

# Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    response_type: str = Field("rag", description="Type of response to generate")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    include_context: bool = Field(True, description="Whether to include context in the response")

class SessionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")

class Response(BaseModel):
    response: str = Field(..., description="Generated response")
    context: Optional[List[Dict[str, Any]]] = Field(None, description="Context used for generation")
    session_id: str = Field(..., description="Session ID")

class SessionList(BaseModel):
    sessions: List[str] = Field(..., description="List of session IDs")

# Dependencies
def get_rag_system():
    global rag_system
    if rag_system is None:
        logger.info("Initializing RAG system")
        rag_system = RAGSystem()
    return rag_system

# Routes
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "arXiv Research Assistant API is running"}

@app.post("/query", response_model=Response, tags=["Query"])
async def query(
    request: QueryRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Process a user query."""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process query
        response = rag_system.process_query(
            query=request.query,
            session_id=session_id,
            response_type=request.response_type
        )
        
        # Add session ID to response
        response["session_id"] = session_id
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=Response, tags=["Chat"])
async def chat(
    request: ChatRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Process a chat message."""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process chat message
        response = rag_system.process_chat(
            message=request.message,
            session_id=session_id,
            include_context=request.include_context
        )
        
        # Add session ID to response
        response["session_id"] = session_id
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_conversation", tags=["Session"])
async def clear_conversation(
    request: SessionRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Clear conversation history."""
    try:
        rag_system.clear_conversation(request.session_id)
        return {"status": "ok", "message": f"Conversation {request.session_id} cleared"}
    
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_conversation", tags=["Session"])
async def delete_conversation(
    request: SessionRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Delete a conversation."""
    try:
        rag_system.delete_conversation(request.session_id)
        return {"status": "ok", "message": f"Conversation {request.session_id} deleted"}
    
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_conversations", response_model=SessionList, tags=["Session"])
async def list_conversations(
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """List all conversations."""
    try:
        sessions = rag_system.list_conversations()
        return {"sessions": sessions}
    
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the API
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        reload=True
    )
