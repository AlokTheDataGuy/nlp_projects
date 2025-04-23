"""
RAG System Module

This module implements the Retrieval-Augmented Generation (RAG) system.
"""

import os
import logging
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Class for implementing the Retrieval-Augmented Generation (RAG) system.
    """

    def __init__(
        self,
        vector_store,
        model_loader,
        config_path: str = "config/model_config.yaml"
    ):
        """
        Initialize the RAGSystem.

        Args:
            vector_store: Vector store for retrieving documents.
            model_loader: Model loader for generating text.
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.vector_store = vector_store
        self.model_loader = model_loader

        # RAG configuration
        self.retrieval_config = self.config["rag"]["retrieval"]
        self.prompt_template = self.config["rag"]["prompt_template"]
        self.system_prompt = self.config["rag"]["system_prompt"]

        # Check if we're using a fine-tuned model
        self.using_fine_tuned_model = self.model_loader.use_fine_tuned_model

        # Initialize LangChain components
        self.llm_chain = None
        self._init_langchain()

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

    def _init_langchain(self) -> None:
        """
        Initialize LangChain components.
        """
        # Check if model is loaded
        if self.model_loader.model is None:
            logger.warning("Model not loaded. LangChain components will not be initialized.")
            return

        try:
            # Create HuggingFace pipeline for LangChain
            hf_pipeline = HuggingFacePipeline(pipeline=self.model_loader.pipeline)

            # Create prompt template
            prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "question"]
            )

            # Create LLM chain
            self.llm_chain = LLMChain(
                llm=hf_pipeline,
                prompt=prompt
            )

            logger.info("LangChain components initialized")
        except Exception as e:
            logger.error(f"Error initializing LangChain components: {str(e)}")

    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.
            k: Number of documents to retrieve.

        Returns:
            List of retrieved documents.
        """
        if k is None:
            k = self.retrieval_config["top_k"]

        # Determine retrieval method
        retrieval_type = self.retrieval_config.get("retriever_type", "hybrid")

        if retrieval_type == "hybrid":
            # Use hybrid search
            documents = self.vector_store.hybrid_search(
                query,
                k=k,
                semantic_weight=self.retrieval_config.get("semantic_weight", 0.7),
                keyword_weight=self.retrieval_config.get("keyword_weight", 0.3)
            )
        else:
            # Use regular search
            documents = self.vector_store.search(
                query,
                index_type="chunks",
                k=k
            )

        # Apply similarity threshold if specified
        threshold = self.retrieval_config.get("similarity_threshold")
        if threshold is not None:
            documents = [doc for doc in documents if doc.get("score", 0) >= threshold]

        # Apply reranking if specified
        if self.retrieval_config.get("reranking", False) and len(documents) > 1:
            documents = self._rerank_documents(query, documents)

        return documents

    def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents based on relevance to the query.

        Args:
            query: The query.
            documents: List of retrieved documents.

        Returns:
            Reranked list of documents.
        """
        # Simple reranking based on combined score
        # In a real system, this would use a more sophisticated reranking model

        # Calculate query terms
        query_terms = set(query.lower().split())

        for doc in documents:
            # Calculate term overlap
            text_terms = set(doc["text"].lower().split())
            matching_terms = query_terms.intersection(text_terms)
            term_overlap = len(matching_terms) / max(1, len(query_terms))

            # Calculate position score (earlier sections are more important)
            position_score = 1.0
            if "section" in doc:
                section = doc["section"].lower()
                if "abstract" in section:
                    position_score = 1.0
                elif "introduction" in section:
                    position_score = 0.9
                elif "conclusion" in section:
                    position_score = 0.8
                else:
                    position_score = 0.7

            # Combine scores
            rerank_score = (
                0.6 * doc.get("combined_score", doc.get("score", 0)) +
                0.3 * term_overlap +
                0.1 * position_score
            )

            doc["rerank_score"] = rerank_score

        # Sort by rerank score
        reranked_documents = sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked_documents

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            documents: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        context_parts = []

        for i, doc in enumerate(documents):
            # Format document with metadata
            paper_id = doc.get("paper_id", "Unknown")
            title = doc.get("title", "Unknown Title")
            section = doc.get("section", "Unknown Section")
            text = doc.get("text", "")

            formatted_doc = f"[{i+1}] Paper: {title} (ID: {paper_id})\nSection: {section}\nContent: {text}\n"
            context_parts.append(formatted_doc)

        return "\n".join(context_parts)

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using the LLM chain or direct model generation.

        Args:
            query: The query to generate a response for.
            context: The context to use for generation.

        Returns:
            Generated response.
        """
        if self.model_loader.model is None:
            logger.error("Model not loaded.")
            return "I'm sorry, but I'm not able to generate a response at the moment."

        try:
            # If using a fine-tuned model, we might want to use a different prompt format
            # that matches how the model was fine-tuned
            if self.using_fine_tuned_model:
                # Format prompt for fine-tuned model
                prompt = f"### Instruction:\nAnswer the following question based on the provided context.\n\n### Input:\nContext: {context}\n\nQuestion: {query}\n\n### Response:\n"

                # Generate response directly using the model_loader
                response = self.model_loader.generate_text(prompt)
                return response
            else:
                # Use LangChain for non-fine-tuned models
                if self.llm_chain is None:
                    logger.error("LLM chain not initialized. Call _init_langchain() first.")
                    return "I'm sorry, but I'm not able to generate a response at the moment."

                # Generate response using LangChain
                response = self.llm_chain.run(context=context, question=query)
                return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, but I encountered an error while generating a response."

    def process_query(self, query: str, k: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a query using the RAG system.

        Args:
            query: The query to process.
            k: Number of documents to retrieve.

        Returns:
            Tuple of (generated response, retrieved documents).
        """
        # Retrieve documents
        documents = self.retrieve_documents(query, k=k)

        if not documents:
            return "I couldn't find any relevant information to answer your question.", []

        # Format context
        context = self.format_context(documents)

        # Generate response
        response = self.generate_response(query, context)

        return response, documents


if __name__ == "__main__":
    # Example usage
    from model_loader import ModelLoader
    from src.knowledge_base.vector_store import VectorStore

    # Initialize components
    model_loader = ModelLoader()
    model_loader.load_model()

    vector_store = VectorStore()

    rag_system = RAGSystem(vector_store, model_loader)

    # Process a query
    query = "Explain the attention mechanism in transformer models"
    response, documents = rag_system.process_query(query)

    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Retrieved {len(documents)} documents")

    # Unload model
    model_loader.unload_model()
