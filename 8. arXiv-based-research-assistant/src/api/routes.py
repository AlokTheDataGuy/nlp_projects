"""
API Routes Module

This module defines the API routes for the arXiv research assistant.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define API models
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[float] = Field(None, description="Timestamp of the message")

class Conversation(BaseModel):
    id: str = Field(..., description="Unique identifier for the conversation")
    user_id: str = Field(..., description="User identifier")
    messages: List[Message] = Field(default=[], description="List of messages in the conversation")
    created_at: float = Field(..., description="Timestamp when the conversation was created")
    updated_at: float = Field(..., description="Timestamp when the conversation was last updated")

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continuing a conversation")
    message: str = Field(..., description="User message")
    user_id: str = Field(..., description="User identifier")

class ChatResponse(BaseModel):
    conversation_id: str = Field(..., description="Conversation ID")
    response: str = Field(..., description="Assistant response")
    citations: Optional[List[Dict[str, Any]]] = Field(None, description="Citations used in the response")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Sources used for the response")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    search_type: str = Field("hybrid", description="Type of search (semantic, keyword, hybrid)")
    limit: int = Field(10, description="Maximum number of results to return")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    search_type: str = Field(..., description="Type of search performed")

class PaperRequest(BaseModel):
    paper_id: str = Field(..., description="Paper ID")

class PaperResponse(BaseModel):
    paper: Dict[str, Any] = Field(..., description="Paper data")
    related_papers: Optional[List[Dict[str, Any]]] = Field(None, description="Related papers")

# Create router
router = APIRouter()

# Dependency to get services
def get_services():
    # This would normally be injected via dependency injection
    # For now, we'll just import them here
    from src.query_processor.query_analyzer import QueryAnalyzer
    from src.query_processor.conversation_manager import ConversationManager
    from src.query_processor.search_engine import SearchEngine
    from src.knowledge_base.vector_store import VectorStore
    from src.knowledge_base.document_store import DocumentStore
    from src.inference_engine.model_loader import ModelLoader
    from src.inference_engine.rag_system import RAGSystem
    from src.inference_engine.response_generator import ResponseGenerator
    
    # Initialize services
    vector_store = VectorStore()
    document_store = DocumentStore()
    model_loader = ModelLoader()
    model_loader.load_model()
    
    query_analyzer = QueryAnalyzer()
    conversation_manager = ConversationManager(document_store=document_store)
    search_engine = SearchEngine(vector_store, document_store)
    rag_system = RAGSystem(vector_store, model_loader)
    response_generator = ResponseGenerator()
    
    return {
        "query_analyzer": query_analyzer,
        "conversation_manager": conversation_manager,
        "search_engine": search_engine,
        "rag_system": rag_system,
        "response_generator": response_generator,
        "model_loader": model_loader
    }

# Chat endpoint
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, services=Depends(get_services)):
    try:
        query_analyzer = services["query_analyzer"]
        conversation_manager = services["conversation_manager"]
        rag_system = services["rag_system"]
        response_generator = services["response_generator"]
        
        # Get or create conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = conversation_manager.create_conversation(request.user_id)
        
        # Add user message to conversation
        conversation_manager.add_message(
            conversation_id,
            {"role": "user", "content": request.message}
        )
        
        # Check if this is a follow-up question
        if conversation_manager.is_follow_up_question(conversation_id, request.message):
            # Resolve follow-up question
            query = conversation_manager.resolve_follow_up(conversation_id, request.message, query_analyzer)
        else:
            query = request.message
        
        # Analyze query
        query_analysis = query_analyzer.classify_question(query)
        
        # Expand query if needed
        if query_analysis["is_cs_relevant"]:
            expanded_query = query_analyzer.expand_query(query)
        else:
            expanded_query = query
        
        # Process query with RAG system
        response_text, documents = rag_system.process_query(expanded_query)
        
        # Format response
        formatted_response = response_generator.format_response(response_text, documents, query)
        
        # Add assistant message to conversation
        conversation_manager.add_message(
            conversation_id,
            {"role": "assistant", "content": formatted_response["response"]}
        )
        
        # Update conversation context
        conversation_manager.update_context(
            conversation_id,
            {
                "last_query": query,
                "last_expanded_query": expanded_query,
                "last_query_type": query_analysis["question_type"],
                "last_documents": [doc.get("paper_id") for doc in documents]
            }
        )
        
        # Create response
        chat_response = ChatResponse(
            conversation_id=conversation_id,
            response=formatted_response["response"],
            citations=formatted_response["citations"],
            sources=formatted_response["sources"]
        )
        
        return chat_response
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoint
@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, services=Depends(get_services)):
    try:
        search_engine = services["search_engine"]
        
        # Perform search
        results = search_engine.search(
            query=request.query,
            search_type=request.search_type,
            limit=request.limit
        )
        
        # Create response
        search_response = SearchResponse(
            results=results,
            query=request.query,
            search_type=request.search_type
        )
        
        return search_response
    
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Paper endpoint
@router.post("/paper", response_model=PaperResponse)
async def get_paper(request: PaperRequest, services=Depends(get_services)):
    try:
        search_engine = services["search_engine"]
        
        # Get paper
        paper = search_engine.search_by_paper_id(request.paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper with ID {request.paper_id} not found")
        
        # Get related papers
        related_papers = search_engine.search_related_papers(request.paper_id)
        
        # Create response
        paper_response = PaperResponse(
            paper=paper,
            related_papers=related_papers
        )
        
        return paper_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in paper endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Conversations endpoint
@router.get("/conversations/{user_id}", response_model=List[Conversation])
async def get_conversations(user_id: str, services=Depends(get_services)):
    try:
        document_store = services["document_store"]
        
        # Get conversations for user
        conversations = document_store.find_documents("conversations", {"user_id": user_id})
        
        return conversations
    
    except Exception as e:
        logger.error(f"Error in conversations endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Conversation endpoint
@router.get("/conversation/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str, services=Depends(get_services)):
    try:
        conversation_manager = services["conversation_manager"]
        
        # Get conversation
        conversation = conversation_manager.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
        
        return conversation
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in conversation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
