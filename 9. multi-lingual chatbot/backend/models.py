"""
Pydantic models for request and response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message")
    session_id: str = Field(..., description="Session ID for conversation tracking")
    language: Optional[str] = Field(None, description="Language code (if manually set)")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    message: str = Field(..., description="Assistant response")
    detected_language: str = Field(..., description="Detected language code")
    success: bool = Field(True, description="Whether the request was successful")


class LanguageSwitchRequest(BaseModel):
    """Request model for language switching endpoint."""
    language: str = Field(..., description="Language code to switch to")
    session_id: str = Field(..., description="Session ID for conversation tracking")


class LanguageSwitchResponse(BaseModel):
    """Response model for language switching endpoint."""
    success: bool = Field(True, description="Whether the request was successful")
    language_name: str = Field(..., description="Name of the language switched to")


class ClearConversationRequest(BaseModel):
    """Request model for clearing conversation endpoint."""
    session_id: str = Field(..., description="Session ID for conversation tracking")


class ClearConversationResponse(BaseModel):
    """Response model for clearing conversation endpoint."""
    success: bool = Field(True, description="Whether the request was successful")


class Language(BaseModel):
    """Model for language information."""
    code: str = Field(..., description="Language code")
    name: str = Field(..., description="Language name")


class Message(BaseModel):
    """Model for conversation messages."""
    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")
    language: str = Field(..., description="Language code of the message")


class Conversation(BaseModel):
    """Model for conversation history."""
    messages: List[Message] = Field(default_factory=list, description="List of messages")
    current_language: str = Field(..., description="Current conversation language")
    last_detected_language: str = Field(..., description="Last detected language")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Overall health status: healthy, degraded, or unhealthy")
    ollama_available: bool = Field(..., description="Whether Ollama is available")
    translation_available: bool = Field(..., description="Whether the translation model is loaded")
    language_detection_available: bool = Field(..., description="Whether the language detection model is loaded")
    transliteration_available: bool = Field(..., description="Whether the transliteration engines are loaded")
    api_version: str = Field(..., description="API version")
    error: Optional[str] = Field(None, description="Error message if status is unhealthy")
