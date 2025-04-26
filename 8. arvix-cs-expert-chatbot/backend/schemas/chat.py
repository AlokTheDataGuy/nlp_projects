from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class MessageRequest(BaseModel):
    """Request model for sending a message"""
    message: str = Field(..., description="The message text")
    conversation_id: Optional[str] = Field(None, description="The conversation ID (if continuing a conversation)")
    user_id: str = Field(..., description="The user ID")

class MessageResponse(BaseModel):
    """Response model for a message"""
    id: str = Field(..., description="The message ID")
    conversation_id: str = Field(..., description="The conversation ID")
    response: str = Field(..., description="The assistant's response")
    relevant_papers: List[Dict[str, Any]] = Field([], description="List of relevant papers")
    timestamp: str = Field(..., description="The timestamp of the response")

class ConversationModel(BaseModel):
    """Model for a conversation"""
    id: str = Field(..., description="The conversation ID")
    user_id: str = Field(..., description="The user ID")
    title: str = Field(..., description="The conversation title")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    messages: List[Dict[str, Any]] = Field([], description="List of messages in the conversation")

class ConversationListResponse(BaseModel):
    """Response model for listing conversations"""
    conversations: List[ConversationModel] = Field([], description="List of conversations")