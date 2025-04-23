"""
Context Manager Module

This module handles context window optimization and document chunking.
"""

import os
import logging
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextManager:
    """
    Class for handling context window optimization and document chunking.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the ContextManager.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        
        # Context window configuration
        self.context_window_config = self.config["llama3"]["context_window"]
        self.max_length = self.context_window_config["max_length"]
        self.truncation_strategy = self.context_window_config["truncation_strategy"]
        self.stride = self.context_window_config["stride"]
    
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
    
    def optimize_context(self, documents: List[Dict[str, Any]], query: str, tokenizer) -> str:
        """
        Optimize the context to fit within the model's context window.
        
        Args:
            documents: List of retrieved documents.
            query: The query being processed.
            tokenizer: The tokenizer to use for counting tokens.
            
        Returns:
            Optimized context string.
        """
        # Estimate token counts
        prompt_template = self.config["rag"]["prompt_template"]
        system_prompt = self.config["rag"]["system_prompt"]
        
        # Calculate tokens for fixed parts
        system_tokens = len(tokenizer.encode(system_prompt))
        template_tokens = len(tokenizer.encode(prompt_template.replace("{context}", "").replace("{question}", "")))
        query_tokens = len(tokenizer.encode(query))
        
        # Reserve tokens for the model's response
        response_tokens = self.config["llama3"]["generation"]["max_new_tokens"]
        
        # Calculate available tokens for context
        available_context_tokens = self.max_length - system_tokens - template_tokens - query_tokens - response_tokens
        
        # Ensure we have a minimum number of tokens for context
        if available_context_tokens < 100:
            logger.warning(f"Very limited context space available: {available_context_tokens} tokens")
            available_context_tokens = 100  # Minimum context size
        
        # Optimize documents to fit in available tokens
        optimized_docs = self._fit_documents_to_token_limit(documents, available_context_tokens, tokenizer)
        
        # Format the optimized context
        context = self._format_context(optimized_docs)
        
        return context
    
    def _fit_documents_to_token_limit(self, documents: List[Dict[str, Any]], token_limit: int, tokenizer) -> List[Dict[str, Any]]:
        """
        Fit documents within the token limit.
        
        Args:
            documents: List of retrieved documents.
            token_limit: Maximum number of tokens allowed.
            tokenizer: The tokenizer to use for counting tokens.
            
        Returns:
            List of optimized documents.
        """
        # Calculate token counts for each document
        for doc in documents:
            # Format document with metadata
            paper_id = doc.get("paper_id", "Unknown")
            title = doc.get("title", "Unknown Title")
            section = doc.get("section", "Unknown Section")
            text = doc.get("text", "")
            
            formatted_doc = f"Paper: {title} (ID: {paper_id})\nSection: {section}\nContent: {text}\n"
            doc["formatted_text"] = formatted_doc
            doc["token_count"] = len(tokenizer.encode(formatted_doc))
        
        # Sort documents by relevance score
        sorted_docs = sorted(documents, key=lambda x: x.get("rerank_score", x.get("combined_score", x.get("score", 0))), reverse=True)
        
        # Apply truncation strategy
        if self.truncation_strategy == "sliding_window":
            return self._apply_sliding_window(sorted_docs, token_limit)
        elif self.truncation_strategy == "start":
            return self._truncate_from_start(sorted_docs, token_limit)
        elif self.truncation_strategy == "end":
            return self._truncate_from_end(sorted_docs, token_limit)
        else:
            # Default to proportional truncation
            return self._truncate_proportionally(sorted_docs, token_limit)
    
    def _apply_sliding_window(self, documents: List[Dict[str, Any]], token_limit: int) -> List[Dict[str, Any]]:
        """
        Apply sliding window truncation strategy.
        
        Args:
            documents: List of documents sorted by relevance.
            token_limit: Maximum number of tokens allowed.
            
        Returns:
            List of optimized documents.
        """
        optimized_docs = []
        current_tokens = 0
        
        # Include documents until we reach the token limit
        for doc in documents:
            if current_tokens + doc["token_count"] <= token_limit:
                optimized_docs.append(doc)
                current_tokens += doc["token_count"]
            else:
                # If this is the first document and it's too large, truncate it
                if not optimized_docs:
                    truncated_doc = self._truncate_document(doc, token_limit)
                    optimized_docs.append(truncated_doc)
                    current_tokens += truncated_doc["token_count"]
                break
        
        return optimized_docs
    
    def _truncate_from_start(self, documents: List[Dict[str, Any]], token_limit: int) -> List[Dict[str, Any]]:
        """
        Truncate documents from the start.
        
        Args:
            documents: List of documents sorted by relevance.
            token_limit: Maximum number of tokens allowed.
            
        Returns:
            List of optimized documents.
        """
        optimized_docs = []
        current_tokens = 0
        
        # Start with the most relevant document
        for doc in documents:
            if current_tokens + doc["token_count"] <= token_limit:
                optimized_docs.append(doc)
                current_tokens += doc["token_count"]
            else:
                # If we can't fit the whole document, truncate it from the start
                remaining_tokens = token_limit - current_tokens
                if remaining_tokens > 100:  # Only truncate if we can include a meaningful amount
                    truncated_doc = self._truncate_document(doc, remaining_tokens, from_start=True)
                    optimized_docs.append(truncated_doc)
                break
        
        return optimized_docs
    
    def _truncate_from_end(self, documents: List[Dict[str, Any]], token_limit: int) -> List[Dict[str, Any]]:
        """
        Truncate documents from the end.
        
        Args:
            documents: List of documents sorted by relevance.
            token_limit: Maximum number of tokens allowed.
            
        Returns:
            List of optimized documents.
        """
        optimized_docs = []
        current_tokens = 0
        
        # Start with the most relevant document
        for doc in documents:
            if current_tokens + doc["token_count"] <= token_limit:
                optimized_docs.append(doc)
                current_tokens += doc["token_count"]
            else:
                # If we can't fit the whole document, truncate it from the end
                remaining_tokens = token_limit - current_tokens
                if remaining_tokens > 100:  # Only truncate if we can include a meaningful amount
                    truncated_doc = self._truncate_document(doc, remaining_tokens, from_start=False)
                    optimized_docs.append(truncated_doc)
                break
        
        return optimized_docs
    
    def _truncate_proportionally(self, documents: List[Dict[str, Any]], token_limit: int) -> List[Dict[str, Any]]:
        """
        Truncate documents proportionally based on their relevance.
        
        Args:
            documents: List of documents sorted by relevance.
            token_limit: Maximum number of tokens allowed.
            
        Returns:
            List of optimized documents.
        """
        # Calculate total tokens
        total_tokens = sum(doc["token_count"] for doc in documents)
        
        # If we're already under the limit, return all documents
        if total_tokens <= token_limit:
            return documents
        
        # Calculate relevance weights
        total_relevance = sum(doc.get("rerank_score", doc.get("combined_score", doc.get("score", 1.0))) for doc in documents)
        
        optimized_docs = []
        remaining_tokens = token_limit
        
        # Allocate tokens proportionally to relevance
        for doc in documents:
            relevance = doc.get("rerank_score", doc.get("combined_score", doc.get("score", 1.0)))
            proportion = relevance / total_relevance if total_relevance > 0 else 1.0 / len(documents)
            
            # Allocate tokens for this document
            allocated_tokens = int(token_limit * proportion)
            
            # Ensure minimum tokens
            allocated_tokens = max(100, min(allocated_tokens, doc["token_count"], remaining_tokens))
            
            if allocated_tokens > 0:
                # Truncate document if necessary
                if allocated_tokens < doc["token_count"]:
                    truncated_doc = self._truncate_document(doc, allocated_tokens)
                    optimized_docs.append(truncated_doc)
                else:
                    optimized_docs.append(doc)
                
                remaining_tokens -= allocated_tokens
            
            # Stop if we've used all tokens
            if remaining_tokens <= 0:
                break
        
        return optimized_docs
    
    def _truncate_document(self, document: Dict[str, Any], token_limit: int, from_start: bool = False) -> Dict[str, Any]:
        """
        Truncate a document to fit within a token limit.
        
        Args:
            document: The document to truncate.
            token_limit: Maximum number of tokens allowed.
            from_start: Whether to truncate from the start (True) or end (False).
            
        Returns:
            Truncated document.
        """
        # Create a copy of the document
        truncated_doc = document.copy()
        
        # Get the text
        text = document.get("text", "")
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Calculate metadata tokens
        paper_id = document.get("paper_id", "Unknown")
        title = document.get("title", "Unknown Title")
        section = document.get("section", "Unknown Section")
        
        metadata_text = f"Paper: {title} (ID: {paper_id})\nSection: {section}\nContent: "
        metadata_tokens = document.get("token_count", 0) - len(tokenizer.encode(text))
        
        # Available tokens for text
        available_text_tokens = token_limit - metadata_tokens
        
        # Truncate text
        truncated_sentences = []
        current_tokens = 0
        
        if from_start:
            # Truncate from the start (keep the end)
            for sentence in reversed(sentences):
                sentence_tokens = len(tokenizer.encode(sentence))
                if current_tokens + sentence_tokens <= available_text_tokens:
                    truncated_sentences.insert(0, sentence)
                    current_tokens += sentence_tokens
                else:
                    break
            
            truncated_text = " ".join(truncated_sentences)
            if truncated_text and sentences and truncated_text != " ".join(sentences):
                truncated_text = "... " + truncated_text
        else:
            # Truncate from the end (keep the start)
            for sentence in sentences:
                sentence_tokens = len(tokenizer.encode(sentence))
                if current_tokens + sentence_tokens <= available_text_tokens:
                    truncated_sentences.append(sentence)
                    current_tokens += sentence_tokens
                else:
                    break
            
            truncated_text = " ".join(truncated_sentences)
            if truncated_text and sentences and truncated_text != " ".join(sentences):
                truncated_text = truncated_text + " ..."
        
        # Update the document
        truncated_doc["text"] = truncated_text
        truncated_doc["formatted_text"] = f"Paper: {title} (ID: {paper_id})\nSection: {section}\nContent: {truncated_text}\n"
        truncated_doc["token_count"] = metadata_tokens + current_tokens
        
        return truncated_doc
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format documents into a context string.
        
        Args:
            documents: List of documents.
            
        Returns:
            Formatted context string.
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Use pre-formatted text if available
            if "formatted_text" in doc:
                formatted_doc = f"[{i+1}] {doc['formatted_text']}"
            else:
                # Format document with metadata
                paper_id = doc.get("paper_id", "Unknown")
                title = doc.get("title", "Unknown Title")
                section = doc.get("section", "Unknown Section")
                text = doc.get("text", "")
                
                formatted_doc = f"[{i+1}] Paper: {title} (ID: {paper_id})\nSection: {section}\nContent: {text}\n"
            
            context_parts.append(formatted_doc)
        
        return "\n".join(context_parts)
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk.
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Overlap between chunks in characters.
            
        Returns:
            List of text chunks.
        """
        if chunk_size is None:
            chunk_size = 512  # Default chunk size
        if chunk_overlap is None:
            chunk_overlap = 50  # Default overlap
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed the chunk size, save the current chunk and start a new one
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep the last few sentences for overlap
                overlap_size = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s) + 1  # +1 for the space
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_size
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for the space
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    context_manager = ContextManager()
    
    # Example documents
    documents = [
        {
            "paper_id": "1234.5678",
            "title": "Attention Is All You Need",
            "section": "Introduction",
            "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "score": 0.95
        },
        {
            "paper_id": "5678.1234",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "section": "Method",
            "text": "BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017). In this work, we denote the number of layers (i.e., Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A.",
            "score": 0.85
        }
    ]
    
    query = "Explain the transformer architecture"
    
    # Optimize context
    optimized_context = context_manager.optimize_context(documents, query, tokenizer)
    
    print(f"Query: {query}")
    print(f"Optimized context:\n{optimized_context}")
    
    # Chunk text
    text = "This is a long text that needs to be chunked into smaller pieces. Each chunk should contain complete sentences as much as possible. The chunks should have some overlap to maintain context across chunks."
    chunks = context_manager.chunk_text(text, chunk_size=50, chunk_overlap=10)
    
    print("\nChunked text:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")
