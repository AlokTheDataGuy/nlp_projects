"""
Core chatbot logic.
"""

import logging
import uuid
import time
from typing import Dict, List, Optional, Tuple, Any
from config import CHATBOT, SUPPORTED_LANGUAGES
from language_detection import LanguageDetector
from translation import Translator
from cultural_adaptation import CulturalAdapter
from cache import ConversationCache
from debug_utils import debug_checkpoint

logger = logging.getLogger(__name__)

class MultilingualChatbot:
    """
    Multilingual chatbot that supports English, Hindi, Bengali, and Marathi.
    """

    def __init__(self):
        """Initialize the chatbot components."""
        self.language_detector = LanguageDetector()
        self.translator = Translator()
        self.cultural_adapter = CulturalAdapter()
        self.conversation_cache = ConversationCache()
        self.max_context_length = CHATBOT["max_context_length"]
        self.default_response = CHATBOT["default_response"]

        logger.info("Multilingual chatbot initialized")

    def process_message(self,
                        message: str,
                        session_id: Optional[str] = None,
                        preferred_language: Optional[str] = None) -> Dict:
        """
        Process a user message and generate a response.

        Args:
            message: User message
            session_id: Session identifier (created if None)
            preferred_language: Preferred language code (detected if None)

        Returns:
            Response object with translated message and metadata
        """
        debug_checkpoint("Starting process_message", {
            "message": message,
            "session_id": session_id,
            "preferred_language": preferred_language
        })
        logger.info(f"Processing message: '{message}'")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Preferred language: {preferred_language}")

        # Create session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session ID: {session_id}")

        # Get conversation history
        conversation = self.conversation_cache.get_conversation(session_id)
        logger.info(f"Retrieved conversation history with {len(conversation)} messages")
        debug_checkpoint("Retrieved conversation history", {"conversation_length": len(conversation)})

        # Detect language if not specified
        debug_checkpoint("Before language detection", {"message": message})
        detected_lang, confidence = self.language_detector.detect(message)
        logger.info(f"Detected language: {detected_lang} with confidence {confidence}")
        debug_checkpoint("After language detection", {
            "detected_lang": detected_lang,
            "confidence": confidence
        })

        # Use detected language if no preference is set
        user_lang = preferred_language or detected_lang
        logger.info(f"Using language: {user_lang}")

        # If preferred_language is explicitly set, use it
        # Otherwise, if detected language is Hindi, Marathi, or Bengali with high confidence, use it
        if preferred_language:
            logger.info(f"Using preferred language: {preferred_language}")
            user_lang = preferred_language
            debug_checkpoint("Using preferred language", {"user_lang": user_lang})
        elif detected_lang in ["hin_Deva", "mar_Deva", "ben_Beng"] and confidence > 0.7:
            logger.info(f"Using detected language {detected_lang} with high confidence")
            user_lang = detected_lang
            debug_checkpoint("Using detected language with high confidence", {"user_lang": user_lang})

        # Check if language is supported
        if user_lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Language {user_lang} not supported, defaulting to English")
            user_lang = "eng_Latn"  # Default to English

        debug_checkpoint("Final language selection", {
            "user_lang": user_lang,
            "detected_lang": detected_lang,
            "preferred_language": preferred_language
        })

        # Translate user message to English for processing
        english_message = message
        if user_lang != "eng_Latn":
            logger.info(f"Translating user message from {user_lang} to English")
            debug_checkpoint("Before translation to English", {
                "message": message,
                "source_lang": user_lang
            })
            try:
                english_message = self.translator.translate_to_english(message, user_lang)
                logger.info(f"Translated message: '{english_message}'")
                debug_checkpoint("After translation to English", {"english_message": english_message})
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                logger.info("Using original message as fallback")
                english_message = message
                debug_checkpoint("Translation to English failed", {"error": str(e)})

        # Add user message to conversation history
        user_message = {
            "role": "user",
            "content": message,
            "language": user_lang,
            "english_content": english_message,
            "timestamp": time.time()
        }
        self.conversation_cache.add_message(session_id, user_message)
        logger.info("Added user message to conversation history")
        debug_checkpoint("Added user message to history", user_message)

        # Generate response in English
        logger.info("Generating response in English")
        debug_checkpoint("Before generating response", {
            "english_message": english_message,
            "conversation_length": len(conversation)
        })
        english_response = self._generate_response(english_message, conversation)
        logger.info(f"Generated English response: '{english_response}'")
        debug_checkpoint("After generating response", {"english_response": english_response})

        # Translate response to user's language
        response_content = english_response
        if user_lang != "eng_Latn":
            logger.info(f"Translating response from English to {user_lang}")
            debug_checkpoint("Before translating response from English", {
                "english_response": english_response,
                "target_lang": user_lang
            })
            try:
                response_content = self.translator.translate_from_english(english_response, user_lang)
                logger.info(f"Translated response: '{response_content}'")
                debug_checkpoint("After translating response from English", {"response_content": response_content})

                # Apply cultural adaptations
                logger.info("Applying cultural adaptations")
                debug_checkpoint("Before cultural adaptation", {"response_content": response_content})
                response_content = self.cultural_adapter.adapt_response(response_content, user_lang)
                logger.info(f"Culturally adapted response: '{response_content}'")
                debug_checkpoint("After cultural adaptation", {"response_content": response_content})
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                logger.info("Using English response as fallback")
                response_content = english_response
                debug_checkpoint("Translation from English failed", {"error": str(e)})

        # Add bot response to conversation history
        bot_message = {
            "role": "assistant",
            "content": response_content,
            "language": user_lang,
            "english_content": english_response,
            "timestamp": time.time()
        }
        self.conversation_cache.add_message(session_id, bot_message)
        logger.info("Added bot response to conversation history")
        debug_checkpoint("Added bot response to history", bot_message)

        # Trim conversation history if needed
        if len(conversation) > self.max_context_length * 2:  # *2 because each turn has user and bot messages
            logger.info(f"Trimming conversation history (current length: {len(conversation)})")
            # Keep only the most recent messages
            new_history = conversation[-self.max_context_length * 2:]
            self.conversation_cache.clear_conversation(session_id)
            for msg in new_history:
                self.conversation_cache.add_message(session_id, msg)
            logger.info(f"Trimmed conversation history to {len(new_history)} messages")
            debug_checkpoint("Trimmed conversation history", {"new_length": len(new_history)})

        # Prepare response object
        response = {
            "session_id": session_id,
            "message": response_content,
            "detected_language": detected_lang,
            "language_confidence": confidence,
            "user_language": user_lang,
            "english_message": english_message,
            "english_response": english_response
        }

        logger.info("Returning response object")
        debug_checkpoint("Completed process_message", response)
        return response

    def _generate_response(self, message: str, conversation: List[Dict]) -> str:
        """
        Generate a response in English based on the user message and conversation history.

        Args:
            message: User message in English
            conversation: Conversation history

        Returns:
            Response in English
        """
        # This is a simple rule-based response generation
        # In a real implementation, this would be replaced with a more sophisticated
        # dialogue management system or a language model

        # Extract recent English messages for context
        recent_messages = []
        for msg in conversation[-self.max_context_length * 2:]:
            if msg["role"] == "user":
                recent_messages.append(msg.get("english_content", msg["content"]))
            else:
                recent_messages.append(msg.get("english_content", msg["content"]))

        # Simple keyword-based response generation
        message_lower = message.lower()

        # Check for common Hindi/Marathi/Bengali phrases that might not translate well
        # Log the message for debugging
        debug_checkpoint("Checking for non-English phrases", {"message": message})

        # Hindi greetings and phrases
        if any(phrase in message for phrase in ["नमस्ते", "नमस्कार", "कैसे हैं", "कैसे हो", "आप का नाम", "आप कैसे है"]):
            debug_checkpoint("Found Hindi phrase", {"message": message})
            if "नमस्ते" in message or "नमस्कार" in message:
                return "Hello! How can I help you today?"
            if "कैसे हैं" in message or "कैसे हो" in message or "आप कैसे है" in message:
                return "I'm doing well, thank you for asking! How can I help you today?"
            if "आप का नाम" in message:
                return "I am a multilingual chatbot that can speak English, Hindi, Bengali, and Marathi. You can call me MultiLingua."

        # Marathi greetings and phrases
        if any(phrase in message for phrase in ["नमस्कार", "कसे आहात", "कसे आहेस", "तुझे नाव", "तुमचे नाव", "तुझे नाव काय आहे"]):
            debug_checkpoint("Found Marathi phrase", {"message": message})
            if "नमस्कार" in message:
                return "Hello! How can I help you today?"
            if "कसे आहात" in message or "कसे आहेस" in message:
                return "I'm doing well, thank you for asking! How can I help you today?"
            if "तुझे नाव" in message or "तुमचे नाव" in message or "तुझे नाव काय आहे" in message:
                return "I am a multilingual chatbot that can speak English, Hindi, Bengali, and Marathi. You can call me MultiLingua."

        # Bengali greetings and phrases
        if any(phrase in message for phrase in ["নমস্কার", "কেমন আছেন", "কেমন আছো", "আপনার নাম", "তোমার নাম"]):
            debug_checkpoint("Found Bengali phrase", {"message": message})
            if "নমস্কার" in message:
                return "Hello! How can I help you today?"
            if "কেমন আছেন" in message or "কেমন আছো" in message:
                return "I'm doing well, thank you for asking! How can I help you today?"
            if "আপনার নাম" in message or "তোমার নাম" in message:
                return "I am a multilingual chatbot that can speak English, Hindi, Bengali, and Marathi. You can call me MultiLingua."

        # Greeting detection
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "greetings", "namaste", "namaskar"]):
            return "Hello! How can I help you today?"

        # Question detection
        if any(question in message_lower for question in ["what", "how", "why", "when", "where", "who", "which"]):
            if "your name" in message_lower:
                return "I am a multilingual chatbot that can speak English, Hindi, Bengali, and Marathi. You can call me MultiLingua."
            elif "language" in message_lower or "languages" in message_lower:
                return "I can speak English, Hindi, Bengali, and Marathi. Feel free to use any of these languages with me."
            elif "weather" in message_lower:
                return "I'm sorry, I don't have access to real-time weather information. You might want to check a weather service for that."
            elif "time" in message_lower:
                return "I don't have access to the current time in your location. Could you check your device's clock?"
            elif "help" in message_lower:
                return "I can help you with general information and have conversations in multiple languages. What would you like to talk about?"

        # Farewell detection
        if any(farewell in message_lower for farewell in ["bye", "goodbye", "see you", "farewell", "alvida", "phir milenge"]):
            return "Goodbye! Feel free to chat again anytime."

        # Thank you detection
        if any(thanks in message_lower for thanks in ["thank", "thanks", "dhanyavaad", "shukriya", "dhanyawad"]):
            return "You're welcome! Is there anything else I can help you with?"

        # Default response
        return self.default_response

    def switch_language(self, session_id: str, new_language: str) -> bool:
        """
        Switch the user's preferred language.

        Args:
            session_id: Session identifier
            new_language: New language code

        Returns:
            True if successful, False otherwise
        """
        if new_language not in SUPPORTED_LANGUAGES:
            return False

        # Get conversation history
        conversation = self.conversation_cache.get_conversation(session_id)

        # Update language preference in conversation
        if conversation:
            # Add a system message indicating language change
            system_message = {
                "role": "system",
                "content": f"Language switched to {SUPPORTED_LANGUAGES[new_language]}",
                "language": new_language,
                "english_content": f"Language switched to {SUPPORTED_LANGUAGES[new_language]}",
                "timestamp": time.time()
            }
            self.conversation_cache.add_message(session_id, system_message)

        return True

    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear the conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            self.conversation_cache.clear_conversation(session_id)
            return True
        except Exception as e:
            logger.error(f"Failed to clear conversation: {e}")
            return False
