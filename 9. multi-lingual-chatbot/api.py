"""
FastAPI backend for the multilingual chatbot.
"""

import os
import logging
import uuid
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from utils import setup_logging, get_language_name, get_language_code, check_gpu_availability
from chatbot import MultilingualChatbot
from config import SUPPORTED_LANGUAGES
from debug_utils import debug_checkpoint

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual Chatbot API",
    description="API for a chatbot that supports English, Hindi, Bengali, and Marathi",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize chatbot
chatbot = MultilingualChatbot()

# Check GPU availability on startup
@app.on_event("startup")
async def startup_event():
    """Check GPU availability on startup."""
    gpu_info = check_gpu_availability()
    if gpu_info["available"]:
        logger.info(f"GPU available: {gpu_info['device_name']}")
        logger.info(f"GPU memory: {gpu_info['total_memory_mb']:.2f} MB")
    else:
        logger.warning("No GPU available, using CPU")

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: Optional[str] = None
    language: Optional[str] = None

class ChatResponse(BaseModel):
    """Chat response model."""
    session_id: str
    message: str
    detected_language: str
    language_confidence: float
    user_language: str
    english_message: str
    english_response: str

class LanguageSwitchRequest(BaseModel):
    """Language switch request model."""
    language: str
    session_id: str

class LanguageSwitchResponse(BaseModel):
    """Language switch response model."""
    success: bool
    language: str
    language_name: str

class ClearConversationRequest(BaseModel):
    """Clear conversation request model."""
    session_id: str

class ClearConversationResponse(BaseModel):
    """Clear conversation response model."""
    success: bool

class LanguageInfo(BaseModel):
    """Language information model."""
    code: str
    name: str

# API endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """API endpoint for chat messages."""
    try:
        # Clean the message to remove newlines that cause fastText to fail
        cleaned_message = request.message.replace('\n', ' ').strip()
        logger.info(f"Original message: '{request.message}', Cleaned: '{cleaned_message}'")
        debug_checkpoint("API received chat request", {
            "message": cleaned_message,
            "session_id": request.session_id,
            "language": request.language
        })

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Always detect the language first to see if we should override the preferred language
        from language_detection import LanguageDetector
        detector = LanguageDetector()
        detected_lang, confidence = detector.detect(cleaned_message)
        debug_checkpoint("API detected language", {
            "detected_lang": detected_lang,
            "confidence": confidence
        })

        # If detected language is Hindi, Marathi, or Bengali with high confidence,
        # use it as the preferred language regardless of what was passed in
        if detected_lang in ["hin_Deva", "mar_Deva", "ben_Beng"] and confidence > 0.7:
            logger.info(f"API using detected language {detected_lang} with confidence {confidence}")
            preferred_language = detected_lang
            debug_checkpoint("API using detected language", {"preferred_language": preferred_language})
        else:
            # Only use the request language if it was explicitly provided
            preferred_language = request.language

        # Process the message
        response = chatbot.process_message(
            message=cleaned_message,
            session_id=session_id,
            preferred_language=preferred_language
        )

        debug_checkpoint("API processed message", response)
        return response

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        debug_checkpoint("API error processing message", {"error": str(e)})

        # Return a fallback response instead of raising an exception
        fallback_response = {
            "session_id": request.session_id or str(uuid.uuid4()),
            "message": "I'm sorry, I encountered an error processing your message. Please try again.",
            "detected_language": "eng_Latn",
            "language_confidence": 1.0,
            "user_language": request.language or "eng_Latn",
            "english_message": request.message,
            "english_response": "I'm sorry, I encountered an error processing your message. Please try again."
        }

        return fallback_response

@app.post("/api/switch-language", response_model=LanguageSwitchResponse)
async def switch_language(request: LanguageSwitchRequest):
    """API endpoint to switch language."""
    try:
        # Switch language
        success = chatbot.switch_language(request.session_id, request.language)

        if success:
            return {
                "success": True,
                "language": request.language,
                "language_name": get_language_name(request.language)
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid language")

    except Exception as e:
        logger.error(f"Error switching language: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-conversation", response_model=ClearConversationResponse)
async def clear_conversation(request: ClearConversationRequest):
    """API endpoint to clear conversation history."""
    try:
        # Clear conversation
        success = chatbot.clear_conversation(request.session_id)

        if success:
            return {"success": True}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear conversation")

    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/languages", response_model=List[LanguageInfo])
async def get_languages():
    """API endpoint to get supported languages."""
    languages = [{"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES.items()]
    return languages

@app.get("/api/health")
async def health_check():
    """API endpoint to check if the service is healthy."""
    return {"status": "ok"}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True)
