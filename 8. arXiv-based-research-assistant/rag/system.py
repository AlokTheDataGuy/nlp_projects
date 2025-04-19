"""
Main RAG system for the arXiv Research Assistant.
"""
from typing import List, Dict, Any, Optional, Union
import json

from rag.llm import PhiModel
from rag.retriever import HybridRetriever
from rag.reranker import Reranker
from rag.generator import ResponseGenerator
from utils.memory import ConversationManager, Conversation
from utils.logger import setup_logger

logger = setup_logger("rag_system", "system.log")

class RAGSystem:
    """Main RAG system for the arXiv Research Assistant."""

    def __init__(self):
        """Initialize the RAG system."""
        # Initialize components
        logger.info("Initializing RAG system components")

        self.llm = PhiModel()
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.generator = ResponseGenerator(
            llm=self.llm,
            retriever=self.retriever,
            reranker=self.reranker
        )

        # Initialize conversation manager
        self.conversation_manager = ConversationManager()

        logger.info("RAG system initialized successfully")

    def process_query(self,
                     query: str,
                     session_id: str,
                     response_type: str = "rag") -> Dict[str, Any]:
        """
        Process a user query.

        Args:
            query: User query
            session_id: Session ID for conversation history
            response_type: Type of response to generate

        Returns:
            Response with context and metadata
        """
        # Get conversation
        conversation = self.conversation_manager.get_conversation(session_id)

        # Convert conversation to messages
        messages = self._conversation_to_messages(conversation)

        # Add current query
        messages.append({"role": "user", "content": query})

        # Generate response
        response = self.generator.generate_response(
            query=query,
            conversation_history=messages[:-1],  # Exclude the current query
            response_type=response_type
        )

        # Update conversation
        conversation.add_message("user", query)
        conversation.add_message("assistant", response["response"])

        # Save conversation
        conversation.save()

        return response

    def process_chat(self,
                    message: str,
                    session_id: str,
                    include_context: bool = True) -> Dict[str, Any]:
        """
        Process a chat message.

        Args:
            message: User message
            session_id: Session ID for conversation history
            include_context: Whether to include context in the response

        Returns:
            Response with optional context
        """
        # Get conversation
        conversation = self.conversation_manager.get_conversation(session_id)

        # Add user message
        conversation.add_message("user", message)

        # Convert conversation to messages
        messages = self._conversation_to_messages(conversation)

        # Generate response
        response = self.generator.generate_chat_response(
            messages=messages,
            include_context=include_context
        )

        # Add assistant response to conversation
        conversation.add_message("assistant", response["response"])

        # Save conversation
        conversation.save()

        return response

    def _conversation_to_messages(self, conversation: Conversation) -> List[Dict[str, str]]:
        """Convert conversation to messages format for the LLM."""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specializing in computer science research."}
        ]

        # Add conversation history
        for message in conversation.get_history():
            messages.append({
                "role": message.role,
                "content": message.content
            })

        return messages

    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history."""
        self.conversation_manager.clear_conversation(session_id)

    def delete_conversation(self, session_id: str) -> None:
        """Delete a conversation."""
        self.conversation_manager.delete_conversation(session_id)

    def list_conversations(self) -> List[str]:
        """List all conversations."""
        return self.conversation_manager.list_conversations()


if __name__ == "__main__":
    # Example usage
    rag_system = RAGSystem()

    # Process a query
    query = "Explain the transformer architecture in deep learning"
    session_id = "test_session"

    response = rag_system.process_query(query, session_id)

    print(f"Query: {query}")
    print(f"Response: {response['response']}")
    print("\nContext:")
    for i, doc in enumerate(response["context"]):
        print(f"  [{i+1}] {doc['title']} by {', '.join(doc['authors'])}")
        print(f"      {doc['content']}")

    # Process a follow-up query
    follow_up = "How does attention mechanism work in transformers?"
    follow_up_response = rag_system.process_chat(follow_up, session_id)

    print(f"\nFollow-up: {follow_up}")
    print(f"Response: {follow_up_response['response']}")
