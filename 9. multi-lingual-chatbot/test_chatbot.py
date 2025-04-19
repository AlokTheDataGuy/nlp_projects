"""
Test script for the multilingual chatbot.
"""

import logging
import unittest
from chatbot import MultilingualChatbot
from language_detection import LanguageDetector
from translation import Translator
from cultural_adaptation import CulturalAdapter
from cache import TranslationCache, ConversationCache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLanguageDetection(unittest.TestCase):
    """Test language detection functionality."""
    
    def setUp(self):
        """Set up the test."""
        self.detector = LanguageDetector()
    
    def test_english_detection(self):
        """Test English language detection."""
        lang, conf = self.detector.detect("Hello, how are you today?")
        self.assertEqual(lang, "eng_Latn")
        self.assertGreaterEqual(conf, 0.7)
    
    def test_hindi_detection(self):
        """Test Hindi language detection."""
        lang, conf = self.detector.detect("नमस्ते, आप कैसे हैं?")
        self.assertEqual(lang, "hin_Deva")
        self.assertGreaterEqual(conf, 0.7)
    
    def test_bengali_detection(self):
        """Test Bengali language detection."""
        lang, conf = self.detector.detect("হ্যালো, আপনি কেমন আছেন?")
        self.assertEqual(lang, "ben_Beng")
        self.assertGreaterEqual(conf, 0.7)
    
    def test_marathi_detection(self):
        """Test Marathi language detection."""
        lang, conf = self.detector.detect("नमस्कार, तुम्ही कसे आहात?")
        self.assertEqual(lang, "mar_Deva")
        self.assertGreaterEqual(conf, 0.7)
    
    def test_empty_text(self):
        """Test detection with empty text."""
        lang, conf = self.detector.detect("")
        self.assertEqual(lang, "eng_Latn")  # Default language
        self.assertEqual(conf, 1.0)
    
    def test_code_mixed(self):
        """Test code-mixed text detection."""
        is_mixed = self.detector.is_code_mixed("Hello नमस्ते")
        self.assertTrue(is_mixed)
        
        is_mixed = self.detector.is_code_mixed("Hello world")
        self.assertFalse(is_mixed)


class TestTranslation(unittest.TestCase):
    """Test translation functionality."""
    
    def setUp(self):
        """Set up the test."""
        self.translator = Translator()
    
    def test_english_to_hindi(self):
        """Test English to Hindi translation."""
        translation = self.translator.translate("Hello", "eng_Latn", "hin_Deva")
        self.assertIn(translation.lower(), ["नमस्ते", "हैलो", "नमस्कार"])
    
    def test_hindi_to_english(self):
        """Test Hindi to English translation."""
        translation = self.translator.translate("नमस्ते", "hin_Deva", "eng_Latn")
        self.assertIn(translation.lower(), ["hello", "hi", "namaste", "greetings"])
    
    def test_same_language(self):
        """Test translation between same languages."""
        text = "Hello world"
        translation = self.translator.translate(text, "eng_Latn", "eng_Latn")
        self.assertEqual(translation, text)
    
    def test_batch_translation(self):
        """Test batch translation."""
        texts = ["Hello", "How are you?"]
        translations = self.translator.translate(texts, "eng_Latn", "hin_Deva")
        self.assertEqual(len(translations), 2)
        self.assertIsInstance(translations, list)


class TestCulturalAdaptation(unittest.TestCase):
    """Test cultural adaptation functionality."""
    
    def setUp(self):
        """Set up the test."""
        self.adapter = CulturalAdapter()
    
    def test_date_formatting(self):
        """Test date formatting."""
        text = "The date is 05/20/2023"
        adapted = self.adapter._format_dates(text)
        self.assertEqual(adapted, "The date is 20-05-2023")
    
    def test_number_formatting(self):
        """Test number formatting."""
        text = "The cost is 1000000 rupees"
        adapted = self.adapter._format_numbers_indian(text)
        self.assertEqual(adapted, "The cost is 10,00,000 rupees")
    
    def test_greeting_adaptation(self):
        """Test greeting adaptation."""
        greeting = self.adapter.adapt_greeting("hin_Deva", "morning")
        self.assertEqual(greeting, "सुप्रभात")
        
        greeting = self.adapter.adapt_greeting("eng_Latn")
        self.assertEqual(greeting, "Hello")


class TestCache(unittest.TestCase):
    """Test caching functionality."""
    
    def setUp(self):
        """Set up the test."""
        self.translation_cache = TranslationCache()
        self.conversation_cache = ConversationCache()
    
    def test_translation_cache(self):
        """Test translation cache."""
        # Add to cache
        self.translation_cache.add("Hello", "eng_Latn", "hin_Deva", "नमस्ते")
        
        # Get from cache
        cached = self.translation_cache.get("Hello", "eng_Latn", "hin_Deva")
        self.assertEqual(cached, "नमस्ते")
        
        # Non-existent entry
        cached = self.translation_cache.get("Goodbye", "eng_Latn", "hin_Deva")
        self.assertIsNone(cached)
    
    def test_conversation_cache(self):
        """Test conversation cache."""
        session_id = "test-session"
        
        # Add message
        self.conversation_cache.add_message(session_id, {"role": "user", "content": "Hello"})
        
        # Get conversation
        conversation = self.conversation_cache.get_conversation(session_id)
        self.assertEqual(len(conversation), 1)
        self.assertEqual(conversation[0]["content"], "Hello")
        
        # Clear conversation
        self.conversation_cache.clear_conversation(session_id)
        conversation = self.conversation_cache.get_conversation(session_id)
        self.assertEqual(len(conversation), 0)


class TestChatbot(unittest.TestCase):
    """Test chatbot functionality."""
    
    def setUp(self):
        """Set up the test."""
        self.chatbot = MultilingualChatbot()
    
    def test_process_message_english(self):
        """Test processing an English message."""
        response = self.chatbot.process_message("Hello", "test-session")
        self.assertIn("session_id", response)
        self.assertIn("message", response)
        self.assertEqual(response["detected_language"], "eng_Latn")
    
    def test_process_message_hindi(self):
        """Test processing a Hindi message."""
        response = self.chatbot.process_message("नमस्ते", "test-session")
        self.assertEqual(response["detected_language"], "hin_Deva")
    
    def test_language_switching(self):
        """Test language switching."""
        session_id = "test-session-switch"
        
        # Start with English
        response = self.chatbot.process_message("Hello", session_id)
        self.assertEqual(response["user_language"], "eng_Latn")
        
        # Switch to Hindi
        self.chatbot.switch_language(session_id, "hin_Deva")
        
        # Send another message
        response = self.chatbot.process_message("Hello again", session_id, "hin_Deva")
        self.assertEqual(response["user_language"], "hin_Deva")
    
    def test_conversation_context(self):
        """Test conversation context maintenance."""
        session_id = "test-session-context"
        
        # First message
        self.chatbot.process_message("Hello", session_id)
        
        # Get conversation history
        conversation = self.chatbot.conversation_cache.get_conversation(session_id)
        self.assertEqual(len(conversation), 2)  # User message + bot response


if __name__ == "__main__":
    unittest.main()
