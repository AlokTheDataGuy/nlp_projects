"""
Caching mechanisms for the chatbot.
"""

import time
import logging
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict
from config import CACHE

logger = logging.getLogger(__name__)

class LRUCache:
    """
    Least Recently Used (LRU) cache with time-to-live (TTL) functionality.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of items in the cache
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()  # {key: (value, timestamp)}
        logger.info(f"Initialized LRU cache with max_size={max_size}, ttl={ttl}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # Check if expired
        if self.ttl > 0 and time.time() - timestamp > self.ttl:
            # Remove expired item
            del self.cache[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # If key exists, update and move to end
        if key in self.cache:
            self.cache.move_to_end(key)
        
        # Add new item with current timestamp
        self.cache[key] = (value, time.time())
        
        # Remove oldest item if cache is full
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def __len__(self) -> int:
        """Get the number of items in the cache."""
        return len(self.cache)


class TranslationCache:
    """
    Cache for translation results.
    """
    
    def __init__(self):
        """Initialize the translation cache."""
        self.enabled = CACHE["enabled"]
        self.cache = LRUCache(
            max_size=CACHE["max_size"],
            ttl=CACHE["ttl"]
        )
        logger.info(f"Translation cache initialized (enabled={self.enabled})")
    
    def _get_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Generate a cache key for the translation.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Cache key
        """
        return f"{source_lang}:{target_lang}:{text}"
    
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Get a cached translation.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Cached translation or None if not found
        """
        if not self.enabled:
            return None
        
        key = self._get_key(text, source_lang, target_lang)
        return self.cache.get(key)
    
    def add(self, text: str, source_lang: str, target_lang: str, translation: str) -> None:
        """
        Add a translation to the cache.
        
        Args:
            text: Original text
            source_lang: Source language code
            target_lang: Target language code
            translation: Translated text
        """
        if not self.enabled:
            return
        
        key = self._get_key(text, source_lang, target_lang)
        self.cache.set(key, translation)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


class ConversationCache:
    """
    Cache for conversation history.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the conversation cache.
        
        Args:
            max_size: Maximum number of conversations to cache
        """
        self.max_size = max_size
        self.conversations = OrderedDict()  # {session_id: conversation_history}
        logger.info(f"Conversation cache initialized with max_size={max_size}")
    
    def get_conversation(self, session_id: str) -> list:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation history as a list of messages
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        else:
            # Move to end (most recently used)
            self.conversations.move_to_end(session_id)
        
        return self.conversations[session_id]
    
    def add_message(self, session_id: str, message: Dict) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            session_id: Session identifier
            message: Message to add
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        else:
            # Move to end (most recently used)
            self.conversations.move_to_end(session_id)
        
        self.conversations[session_id].append(message)
        
        # Remove oldest conversation if cache is full
        if len(self.conversations) > self.max_size:
            self.conversations.popitem(last=False)
    
    def clear_conversation(self, session_id: str) -> None:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.conversations:
            self.conversations[session_id] = []
    
    def clear_all(self) -> None:
        """Clear all conversation histories."""
        self.conversations.clear()
