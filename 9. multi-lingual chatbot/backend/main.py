"""
FastAPI application for the multilingual chatbot.
"""
import logging
from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import API_HOST, API_PORT, DEBUG, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, TRANSLATION_MODEL, OLLAMA_MODEL
from models import (
    ChatRequest, ChatResponse,
    LanguageSwitchRequest, LanguageSwitchResponse,
    ClearConversationRequest, ClearConversationResponse,
    HealthCheckResponse,
    Message, Language
)
from language_detection import detect_language
from translation import translate
from indicxlit_transliteration import transliterate_word
from llm import generate_response, check_model_availability, restart_ollama_service
# Set up logging
logging.basicConfig(
    level=logging.INFO if not DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log")
    ]
)

# Session storage (in-memory for simplicity)
_sessions = {}

# Helper functions
def get_language_name(language_code):
    """Get language name from language code."""
    for lang in SUPPORTED_LANGUAGES:
        if lang["code"] == language_code:
            return lang["name"]
    return language_code

def is_supported_language(language_code):
    """Check if language is supported."""
    return any(lang["code"] == language_code for lang in SUPPORTED_LANGUAGES)

# Session management functions
def get_session(session_id):
    """Get session by ID or create a new one."""
    if session_id not in _sessions:
        # Create new session with default language
        _sessions[session_id] = models.Conversation(
            messages=[],
            current_language=DEFAULT_LANGUAGE,
            last_detected_language=DEFAULT_LANGUAGE
        )
    return _sessions[session_id]

def update_session(session_id, conversation):
    """Update session."""
    _sessions[session_id] = conversation
    return True

def clear_session(session_id):
    """Clear session."""
    if session_id in _sessions:
        _sessions[session_id] = models.Conversation(
            messages=[],
            current_language=_sessions[session_id].current_language,
            last_detected_language=_sessions[session_id].last_detected_language
        )
        return True
    return False
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multilingual Chatbot API",
    description="API for a chatbot that supports English, Hindi, Bengali, and Marathi",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting multilingual chatbot API...")
    logger.info(f"API running at http://{API_HOST}:{API_PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Supported languages: {[lang['name'] for lang in SUPPORTED_LANGUAGES]}")
    logger.info(f"Translation model: {TRANSLATION_MODEL}")
    logger.info(f"Ollama model: {OLLAMA_MODEL}")

    # Check if LLaMA3.1 model is available
    model_available = await check_model_availability()
    if not model_available:
        logger.warning(f"LLaMA3.1 model not available in Ollama. Attempting to restart Ollama service...")

        # Try to restart Ollama service
        if restart_ollama_service():
            # Check again after restart
            model_available = await check_model_availability()
            if model_available:
                logger.info("Successfully restarted Ollama service and verified model availability.")
            else:
                logger.warning(f"LLaMA3.1 model still not available after Ollama restart.")
                logger.info(f"You can download it with: ollama pull {OLLAMA_MODEL}")
        else:
            logger.warning(f"Failed to restart Ollama service. Please make sure Ollama is installed and the model is downloaded.")
            logger.info(f"You can download it with: ollama pull {OLLAMA_MODEL}")


@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """Check the health of the API and its dependencies."""
    try:
        # Check if Ollama is available
        ollama_available = await check_model_availability(max_retries=1)

        # Check if translation model is loaded
        from translation import _model as translation_model
        translation_available = translation_model is not None

        # Check if language detection model is loaded
        from language_detection import _model as language_detection_model
        language_detection_available = language_detection_model is not None

        # Check if transliteration engines are loaded
        from indicxlit_transliteration import _en2indic_engine, _indic2en_engine
        transliteration_available = _en2indic_engine is not None and _indic2en_engine is not None

        # Return health status
        return HealthCheckResponse(
            status="healthy" if (ollama_available and translation_available and language_detection_available and transliteration_available) else "degraded",
            ollama_available=ollama_available,
            translation_available=translation_available,
            language_detection_available=language_detection_available,
            transliteration_available=transliteration_available,
            api_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Error checking health: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            ollama_available=False,
            translation_available=False,
            language_detection_available=False,
            transliteration_available=False,
            api_version="1.0.0",
            error=str(e)
        )


@app.get("/api/languages", response_model=List[Language])
async def get_languages():
    """Get supported languages."""
    return SUPPORTED_LANGUAGES


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and generate a response.

    The flow is:
    1. Detect the language of the user message
    2. Transliterate if input is in Latin script but user prefers Indic script
    3. Translate to English if needed
    4. Process with LLaMA3
    5. Translate back to the detected language
    """
    try:
        # Get session
        conversation = get_session(request.session_id)

        # Detect language if not specified
        detected_language = request.language or detect_language(request.message)

        # Update last detected language
        conversation.last_detected_language = detected_language

        # Check if language is supported
        if not is_supported_language(detected_language):
            detected_language = DEFAULT_LANGUAGE



        # Check if the message is in Latin script but the user's preferred language is an Indic language
        # This is for handling transliteration (e.g., "namaste" -> "नमस्ते")
        user_message = request.message
        is_latin_script = detect_language(user_message) == "eng_Latn"

        if is_latin_script and detected_language != "eng_Latn":
            # Try to transliterate the message from Latin to the detected Indic script
            try:
                transliterated_suggestions = transliterate_word(user_message, "eng_Latn", detected_language)
                if transliterated_suggestions and len(transliterated_suggestions) > 0:
                    user_message = transliterated_suggestions[0]  # Use the top suggestion
                    logger.info(f"Transliterated user message from Latin to {detected_language}: {user_message}")
            except Exception as e:
                logger.error(f"Error transliterating user message: {e}")

        # Translate user message to English for processing if needed
        if detected_language != "eng_Latn":
            # We could translate the user message here if needed for better understanding
            # But for now, we'll just process the message as is
            pass

        # Add user message to conversation history
        conversation.messages.append(
            Message(
                role="user",
                content=user_message,  # Use the potentially transliterated message
                language=detected_language
            )
        )

        # Format messages for LLaMA3
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in conversation.messages[-10:]  # Limit context to last 10 messages
        ]

        # Generate response in English
        english_response = await generate_response(
            formatted_messages,
            "eng_Latn",  # Always generate in English first
            max_retries=3,  # Increase retries to handle temporary failures
            temperature=0.7,  # Control randomness
            max_tokens=500  # Limit response length to avoid timeouts
        )

        # Check if we got a valid response
        if not english_response:
            logger.error("Failed to generate response from LLaMA3.1 after retries")

            # Try to restart Ollama service
            logger.warning("Attempting to restart Ollama service due to failed response generation...")
            restart_success = restart_ollama_service()

            if restart_success:
                # Try one more time after restart
                logger.info("Retrying response generation after Ollama restart...")
                english_response = await generate_response(
                    formatted_messages,
                    "eng_Latn",
                    max_retries=2,
                    temperature=0.7,
                    max_tokens=300  # Reduce token count for faster response
                )

                if not english_response:
                    logger.error("Still failed to generate response after Ollama restart")
                    # Use a fallback response instead of failing completely
                    english_response = "I'm sorry, I'm having trouble processing your request right now. The language model service is experiencing high load. Please try again with a simpler question or try again later."
                    logger.warning("Using fallback response due to LLM service failure")
            else:
                # If restart failed, use fallback response
                english_response = "I'm sorry, I'm having trouble connecting to the language model service. This could be due to high demand or a temporary outage. Please try again later."
                logger.warning("Using fallback response due to Ollama restart failure")

        # Log the response for debugging
        logger.info(f"Generated response from LLaMA3.1: {english_response[:50]}...")

        # Translate response to detected language if needed
        response_text = english_response
        if detected_language != "eng_Latn":
            try:
                translated_response = translate(english_response, "eng_Latn", detected_language)
                if translated_response:
                    response_text = translated_response
                    logger.info(f"Translated response to {detected_language}")
                else:
                    logger.warning(f"Translation to {detected_language} failed, using English response")
            except Exception as e:
                logger.error(f"Error translating response: {e}")
                # Continue with English response if translation fails



        # Add assistant response to conversation history
        conversation.messages.append(
            Message(
                role="assistant",
                content=response_text,
                language=detected_language
            )
        )

        # Update session
        update_session(request.session_id, conversation)

        # Return response
        return ChatResponse(
            message=response_text,
            detected_language=detected_language,
            success=True
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/switch-language", response_model=LanguageSwitchResponse)
async def switch_language(request: LanguageSwitchRequest):
    """Switch the conversation language."""
    try:
        # Check if language is supported
        if not is_supported_language(request.language):
            raise HTTPException(status_code=400, detail="Unsupported language")

        # Get session
        conversation = get_session(request.session_id)

        # Update current language
        conversation.current_language = request.language

        # Update session
        update_session(request.session_id, conversation)

        # Return response
        return LanguageSwitchResponse(
            success=True,
            language_name=get_language_name(request.language)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching language: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear-conversation", response_model=ClearConversationResponse)
async def clear_conversation(request: ClearConversationRequest):
    """Clear the conversation history."""
    try:
        # Clear session
        success = clear_session(request.session_id)

        # Return response
        return ClearConversationResponse(success=success)

    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG
    )
