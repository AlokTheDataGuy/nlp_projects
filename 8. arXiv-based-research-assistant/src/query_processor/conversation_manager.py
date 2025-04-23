"""
Conversation Manager Module

This module maintains dialog context and handles follow-up questions.
"""

import os
import logging
import yaml
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Class for maintaining dialog context and handling follow-up questions.
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml", document_store=None):
        """
        Initialize the ConversationManager.
        
        Args:
            config_path: Path to the configuration file.
            document_store: Document store for persisting conversations.
        """
        self.config = self._load_config(config_path)
        self.document_store = document_store
        
        # UI configuration
        self.max_history = self.config["ui"]["max_history"]
        
        # Active conversations
        self.active_conversations = {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dict containing configuration.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def create_conversation(self, user_id: str) -> str:
        """
        Create a new conversation.
        
        Args:
            user_id: ID of the user.
            
        Returns:
            Conversation ID.
        """
        conversation_id = str(uuid.uuid4())
        
        # Create conversation object
        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": [],
            "context": {}
        }
        
        # Store in active conversations
        self.active_conversations[conversation_id] = conversation
        
        # Persist to database if document store is available
        if self.document_store:
            self.document_store.insert_document("conversations", conversation)
        
        return conversation_id
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation.
            message: Message to add.
            
        Returns:
            True if successful, False otherwise.
        """
        if conversation_id not in self.active_conversations:
            # Try to load from database
            if self.document_store:
                conversation = self.document_store.find_document("conversations", {"id": conversation_id})
                if conversation:
                    self.active_conversations[conversation_id] = conversation
                else:
                    logger.error(f"Conversation {conversation_id} not found")
                    return False
            else:
                logger.error(f"Conversation {conversation_id} not found")
                return False
        
        # Add message
        conversation = self.active_conversations[conversation_id]
        
        # Ensure message has required fields
        if "role" not in message or "content" not in message:
            logger.error("Message must have 'role' and 'content' fields")
            return False
        
        # Add timestamp
        message["timestamp"] = time.time()
        
        # Add message to conversation
        conversation["messages"].append(message)
        conversation["updated_at"] = time.time()
        
        # Limit history size
        if len(conversation["messages"]) > self.max_history:
            conversation["messages"] = conversation["messages"][-self.max_history:]
        
        # Update in database if available
        if self.document_store:
            self.document_store.update_document(
                "conversations",
                {"id": conversation_id},
                {"messages": conversation["messages"], "updated_at": conversation["updated_at"]}
            )
        
        return True
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation.
        
        Args:
            conversation_id: ID of the conversation.
            
        Returns:
            Conversation object or None if not found.
        """
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # Try to load from database
        if self.document_store:
            conversation = self.document_store.find_document("conversations", {"id": conversation_id})
            if conversation:
                self.active_conversations[conversation_id] = conversation
                return conversation
        
        logger.error(f"Conversation {conversation_id} not found")
        return None
    
    def get_conversation_history(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the message history for a conversation.
        
        Args:
            conversation_id: ID of the conversation.
            limit: Maximum number of messages to return (from most recent).
            
        Returns:
            List of messages or empty list if conversation not found.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation["messages"]
        
        if limit is not None and limit > 0:
            messages = messages[-limit:]
        
        return messages
    
    def update_context(self, conversation_id: str, context_updates: Dict[str, Any]) -> bool:
        """
        Update the context for a conversation.
        
        Args:
            conversation_id: ID of the conversation.
            context_updates: Updates to apply to the context.
            
        Returns:
            True if successful, False otherwise.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        # Update context
        conversation["context"].update(context_updates)
        conversation["updated_at"] = time.time()
        
        # Update in database if available
        if self.document_store:
            self.document_store.update_document(
                "conversations",
                {"id": conversation_id},
                {"context": conversation["context"], "updated_at": conversation["updated_at"]}
            )
        
        return True
    
    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get the context for a conversation.
        
        Args:
            conversation_id: ID of the conversation.
            
        Returns:
            Context dictionary or empty dict if conversation not found.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return {}
        
        return conversation["context"]
    
    def is_follow_up_question(self, conversation_id: str, query: str) -> bool:
        """
        Determine if a query is a follow-up question.
        
        Args:
            conversation_id: ID of the conversation.
            query: The query to check.
            
        Returns:
            True if the query is a follow-up question, False otherwise.
        """
        # Get conversation history
        history = self.get_conversation_history(conversation_id, limit=3)
        if not history:
            return False
        
        # Check for follow-up indicators
        follow_up_indicators = [
            "it", "this", "that", "these", "those", "they", "them", "their",
            "the", "he", "she", "his", "her", "its", "their",
            "what about", "how about", "what if", "why", "how", "and", "also",
            "additionally", "furthermore", "moreover", "in addition",
            "tell me more", "explain further", "elaborate", "continue"
        ]
        
        query_lower = query.lower()
        
        # Check if query starts with a follow-up indicator
        for indicator in follow_up_indicators:
            if query_lower.startswith(indicator + " ") or query_lower == indicator:
                return True
        
        # Check if query contains pronouns that might refer to previous context
        for indicator in ["it", "this", "that", "these", "those", "they", "them", "their"]:
            if f" {indicator} " in f" {query_lower} ":
                return True
        
        return False
    
    def get_context_for_follow_up(self, conversation_id: str, query: str) -> Dict[str, Any]:
        """
        Get context information for a follow-up question.
        
        Args:
            conversation_id: ID of the conversation.
            query: The follow-up query.
            
        Returns:
            Context information for the follow-up question.
        """
        # Get conversation history
        history = self.get_conversation_history(conversation_id, limit=3)
        if not history:
            return {"is_follow_up": False, "original_query": query}
        
        # Get the most recent user query and assistant response
        user_queries = [msg for msg in history if msg["role"] == "user"]
        assistant_responses = [msg for msg in history if msg["role"] == "assistant"]
        
        if not user_queries or not assistant_responses:
            return {"is_follow_up": False, "original_query": query}
        
        # Get the previous query and response
        prev_query = user_queries[-1]["content"] if len(user_queries) > 1 else None
        prev_response = assistant_responses[-1]["content"] if assistant_responses else None
        
        # Get conversation context
        context = self.get_context(conversation_id)
        
        # Create follow-up context
        follow_up_context = {
            "is_follow_up": True,
            "original_query": query,
            "previous_query": prev_query,
            "previous_response": prev_response,
            "conversation_context": context
        }
        
        return follow_up_context
    
    def resolve_follow_up(self, conversation_id: str, query: str, query_analyzer) -> str:
        """
        Resolve a follow-up question into a standalone question.
        
        Args:
            conversation_id: ID of the conversation.
            query: The follow-up query.
            query_analyzer: Query analyzer for extracting entities.
            
        Returns:
            Resolved query.
        """
        # Check if this is a follow-up question
        if not self.is_follow_up_question(conversation_id, query):
            return query
        
        # Get follow-up context
        follow_up_context = self.get_context_for_follow_up(conversation_id, query)
        
        if not follow_up_context["is_follow_up"] or not follow_up_context["previous_query"]:
            return query
        
        # Get entities from previous query
        prev_query = follow_up_context["previous_query"]
        prev_entities = query_analyzer.extract_entities(prev_query)
        prev_key_terms = query_analyzer.extract_key_terms(prev_query)
        
        # Get entities from current query
        curr_entities = query_analyzer.extract_entities(query)
        curr_key_terms = query_analyzer.extract_key_terms(query)
        
        # Combine entities and key terms
        all_entities = {}
        for entity_type, entities in prev_entities.items():
            if entity_type not in all_entities:
                all_entities[entity_type] = []
            all_entities[entity_type].extend(entities)
        
        for entity_type, entities in curr_entities.items():
            if entity_type not in all_entities:
                all_entities[entity_type] = []
            all_entities[entity_type].extend(entities)
        
        # Remove duplicates
        for entity_type in all_entities:
            all_entities[entity_type] = list(dict.fromkeys(all_entities[entity_type]))
        
        # Combine key terms
        all_key_terms = list(dict.fromkeys(prev_key_terms + curr_key_terms))
        
        # Simple resolution: combine previous query with current query
        resolved_query = f"{prev_query} {query}"
        
        # Update context with resolved query
        self.update_context(conversation_id, {"resolved_query": resolved_query})
        
        return resolved_query


if __name__ == "__main__":
    # Example usage
    from query_analyzer import QueryAnalyzer
    
    conversation_manager = ConversationManager()
    query_analyzer = QueryAnalyzer()
    
    # Create a conversation
    conversation_id = conversation_manager.create_conversation("user123")
    print(f"Created conversation: {conversation_id}")
    
    # Add messages
    conversation_manager.add_message(
        conversation_id,
        {"role": "user", "content": "What is a transformer in deep learning?"}
    )
    
    conversation_manager.add_message(
        conversation_id,
        {"role": "assistant", "content": "A transformer is a deep learning model architecture introduced in the paper 'Attention Is All You Need'. It relies entirely on attention mechanisms without using recurrence or convolutions."}
    )
    
    # Test follow-up question
    follow_up_query = "How does it compare to RNNs?"
    
    is_follow_up = conversation_manager.is_follow_up_question(conversation_id, follow_up_query)
    print(f"Is follow-up: {is_follow_up}")
    
    resolved_query = conversation_manager.resolve_follow_up(conversation_id, follow_up_query, query_analyzer)
    print(f"Resolved query: {resolved_query}")
    
    # Add follow-up to conversation
    conversation_manager.add_message(
        conversation_id,
        {"role": "user", "content": follow_up_query}
    )
    
    # Get conversation history
    history = conversation_manager.get_conversation_history(conversation_id)
    print("\nConversation history:")
    for message in history:
        print(f"{message['role']}: {message['content']}")
