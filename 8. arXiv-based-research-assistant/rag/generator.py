"""
Response generation for the RAG system.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import json

from rag.llm import PhiModel
from rag.retriever import HybridRetriever
from rag.reranker import Reranker
from rag.prompts import (
    SYSTEM_PROMPT, get_rag_prompt, get_summarization_prompt,
    get_concept_explanation_prompt, get_chain_of_thought_prompt
)
from utils.config import NUM_DOCUMENTS
from utils.logger import setup_logger

logger = setup_logger("response_generator", "generator.log")

class ResponseGenerator:
    """Generates responses using the RAG system."""

    def __init__(self,
                 llm: Optional[PhiModel] = None,
                 retriever: Optional[HybridRetriever] = None,
                 reranker: Optional[Reranker] = None,
                 num_documents: int = NUM_DOCUMENTS):
        """
        Initialize the response generator.

        Args:
            llm: Phi-2 model
            retriever: Hybrid retriever
            reranker: Result reranker
            num_documents: Number of documents to retrieve
        """
        self.llm = llm or PhiModel()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.num_documents = num_documents

    def generate_response(self,
                         query: str,
                         conversation_history: Optional[List[Dict[str, str]]] = None,
                         response_type: str = "rag") -> Dict[str, Any]:
        """
        Generate a response to a query.

        Args:
            query: User query
            conversation_history: Previous conversation history
            response_type: Type of response to generate ('rag', 'summarize', 'explain', 'chain_of_thought')

        Returns:
            Response with context and metadata
        """
        conversation_history = conversation_history or []

        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, k=self.num_documents * 2)

        # Rerank documents
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_n=self.num_documents)

        # Generate prompt based on response type
        if response_type == "summarize":
            # For summarization, we assume the query contains a paper ID or title
            # and we retrieve that specific paper
            if len(reranked_docs) > 0:
                doc = reranked_docs[0]
                prompt = get_summarization_prompt(
                    title=doc["title"],
                    authors=doc["authors"],
                    content=doc["content"]
                )
            else:
                prompt = f"I need to summarize a paper about: {query}\nHowever, I couldn't find any relevant papers in my database."

        elif response_type == "explain":
            prompt = get_concept_explanation_prompt(
                concept=query,
                context=reranked_docs
            )

        elif response_type == "chain_of_thought":
            prompt = get_chain_of_thought_prompt(
                query=query,
                context=reranked_docs
            )

        else:  # Default to RAG
            prompt = get_rag_prompt(
                query=query,
                context=reranked_docs
            )

        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Add conversation history
        messages.extend(conversation_history)

        # Add current query and context
        messages.append({"role": "user", "content": prompt})

        # Generate response
        response_text = self.llm.generate_chat(messages)

        # Prepare response with metadata
        response = {
            "query": query,
            "response": response_text,
            "context": [
                {
                    "title": doc["title"],
                    "authors": doc["authors"],
                    "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "paper_id": doc["paper_id"],
                    "chunk_id": doc["chunk_id"]
                }
                for doc in reranked_docs
            ],
            "response_type": response_type
        }

        return response

    def generate_chat_response(self,
                              messages: List[Dict[str, str]],
                              include_context: bool = True) -> Dict[str, Any]:
        """
        Generate a response in a chat conversation.

        Args:
            messages: Chat messages
            include_context: Whether to include context in the response

        Returns:
            Response with optional context
        """
        # Extract the latest user query
        latest_user_message = None
        for message in reversed(messages):
            if message["role"] == "user":
                latest_user_message = message["content"]
                break

        if latest_user_message is None:
            return {"response": "No user message found in the conversation."}

        # Generate response
        response = self.generate_response(
            query=latest_user_message,
            conversation_history=messages[:-1]  # Exclude the latest user message
        )

        # Format response for chat
        chat_response = {
            "response": response["response"]
        }

        # Include context if requested
        if include_context:
            chat_response["context"] = response["context"]

        return chat_response


if __name__ == "__main__":
    # Example usage
    generator = ResponseGenerator()

    # Generate response to a query
    query = "Explain the transformer architecture in deep learning"
    response = generator.generate_response(query)

    print(f"Query: {query}")
    print(f"Response: {response['response']}")
    print("\nContext:")
    for i, doc in enumerate(response["context"]):
        print(f"  [{i+1}] {doc['title']} by {', '.join(doc['authors'])}")
        print(f"      {doc['content']}")

    # Generate chat response
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specializing in computer science research."},
        {"role": "user", "content": "What is the difference between CNN and RNN?"},
        {"role": "assistant", "content": "CNNs (Convolutional Neural Networks) and RNNs (Recurrent Neural Networks) are two different types of neural network architectures..."},
        {"role": "user", "content": "How do transformers improve on RNNs?"}
    ]

    chat_response = generator.generate_chat_response(messages)

    print("\nChat Response:")
    print(f"Response: {chat_response['response']}")
