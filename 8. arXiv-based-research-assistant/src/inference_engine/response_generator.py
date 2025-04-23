"""
Response Generator Module

This module formats final responses with citations and explanations.
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

class ResponseGenerator:
    """
    Class for formatting final responses with citations and explanations.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the ResponseGenerator.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
    
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
    
    def format_response(self, response: str, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Format the response with citations and explanations.
        
        Args:
            response: The raw response from the model.
            documents: The retrieved documents used for the response.
            query: The original query.
            
        Returns:
            Dictionary containing the formatted response and metadata.
        """
        # Extract citations from the response
        citations = self._extract_citations(response)
        
        # Format citations
        formatted_citations = self._format_citations(citations, documents)
        
        # Clean the response
        cleaned_response = self._clean_response(response)
        
        # Add missing citations if necessary
        response_with_citations = self._add_missing_citations(cleaned_response, documents, citations)
        
        # Create the final response
        formatted_response = {
            "query": query,
            "response": response_with_citations,
            "citations": formatted_citations,
            "sources": self._format_sources(documents)
        }
        
        return formatted_response
    
    def _extract_citations(self, response: str) -> List[str]:
        """
        Extract citations from the response.
        
        Args:
            response: The response to extract citations from.
            
        Returns:
            List of citation IDs.
        """
        # Look for citation patterns like [1], (Smith et al., 2020), etc.
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([^)]+?(?:\d{4})[^)]*)\)',  # (Author et al., 2020)
            r'(?:paper|article|study|research) (?:ID|id|Id): (\w+\.\w+)'  # paper ID: 1234.5678
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, response)
            citations.extend(matches)
        
        return citations
    
    def _format_citations(self, citations: List[str], documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format citations with document metadata.
        
        Args:
            citations: List of citation IDs or references.
            documents: The retrieved documents.
            
        Returns:
            List of formatted citation dictionaries.
        """
        formatted_citations = []
        
        # Create a mapping of document indices and IDs
        doc_indices = {str(i+1): doc for i, doc in enumerate(documents)}
        doc_ids = {doc.get("paper_id", ""): doc for doc in documents}
        
        for citation in citations:
            # Try to match the citation to a document
            matched_doc = None
            
            # Check if citation is a numeric index
            if citation.isdigit() and citation in doc_indices:
                matched_doc = doc_indices[citation]
            
            # Check if citation is a paper ID
            elif citation in doc_ids:
                matched_doc = doc_ids[citation]
            
            # Check if citation contains a paper ID
            else:
                for paper_id, doc in doc_ids.items():
                    if paper_id in citation:
                        matched_doc = doc
                        break
            
            if matched_doc:
                # Format the citation
                formatted_citation = {
                    "paper_id": matched_doc.get("paper_id", "Unknown"),
                    "title": matched_doc.get("title", "Unknown Title"),
                    "authors": matched_doc.get("authors", []),
                    "section": matched_doc.get("section", "Unknown Section"),
                    "citation_text": citation
                }
                
                formatted_citations.append(formatted_citation)
        
        return formatted_citations
    
    def _clean_response(self, response: str) -> str:
        """
        Clean the response by removing artifacts and formatting issues.
        
        Args:
            response: The response to clean.
            
        Returns:
            Cleaned response.
        """
        # Remove repeated citations
        cleaned = re.sub(r'(\[\d+\])\s*\1+', r'\1', response)
        
        # Remove excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Remove "According to the provided context" and similar phrases
        context_phrases = [
            r'According to the (provided|given) context,?\s*',
            r'Based on the (provided|given) context,?\s*',
            r'From the (provided|given) (context|information),?\s*',
            r'The (provided|given) context (states|mentions|indicates) that\s*'
        ]
        
        for phrase in context_phrases:
            cleaned = re.sub(phrase, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def _add_missing_citations(self, response: str, documents: List[Dict[str, Any]], existing_citations: List[str]) -> str:
        """
        Add missing citations to the response.
        
        Args:
            response: The cleaned response.
            documents: The retrieved documents.
            existing_citations: Citations already in the response.
            
        Returns:
            Response with added citations where appropriate.
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP to identify statements that need citations
        
        # If no documents or already has citations, return as is
        if not documents or existing_citations:
            return response
        
        # Add a general citation at the end if none exist
        if not re.search(r'\[\d+\]', response) and not re.search(r'\([^)]+\d{4}[^)]*\)', response):
            # Get the most relevant document
            top_doc = documents[0]
            paper_id = top_doc.get("paper_id", "Unknown")
            
            # Add citation at the end
            response += f"\n\nSource: [{paper_id}]"
        
        return response
    
    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source documents for display.
        
        Args:
            documents: The retrieved documents.
            
        Returns:
            List of formatted source dictionaries.
        """
        sources = []
        
        for i, doc in enumerate(documents):
            source = {
                "index": i + 1,
                "paper_id": doc.get("paper_id", "Unknown"),
                "title": doc.get("title", "Unknown Title"),
                "authors": doc.get("authors", []),
                "section": doc.get("section", "Unknown Section"),
                "text": doc.get("text", ""),
                "score": doc.get("rerank_score", doc.get("combined_score", doc.get("score", 0)))
            }
            
            sources.append(source)
        
        return sources
    
    def generate_explanation(self, response: str, documents: List[Dict[str, Any]], query: str) -> str:
        """
        Generate an explanation of how the response was derived from the sources.
        
        Args:
            response: The formatted response.
            documents: The retrieved documents.
            query: The original query.
            
        Returns:
            Explanation text.
        """
        # Create a simple explanation
        num_sources = len(documents)
        
        explanation = [
            f"This response was generated based on {num_sources} relevant documents from arXiv.",
            "The documents were retrieved using semantic search and ranked by relevance to your query.",
            "The response synthesizes information from these sources and includes citations where appropriate."
        ]
        
        # Add information about top sources
        if documents:
            explanation.append("\nTop sources used:")
            
            for i, doc in enumerate(documents[:3]):  # Show top 3 sources
                title = doc.get("title", "Unknown Title")
                authors = doc.get("authors", [])
                author_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_text += ", et al."
                
                paper_id = doc.get("paper_id", "Unknown")
                score = doc.get("rerank_score", doc.get("combined_score", doc.get("score", 0)))
                
                explanation.append(f"[{i+1}] {title} (ID: {paper_id})")
                explanation.append(f"    Authors: {author_text}")
                explanation.append(f"    Relevance: {score:.2f}")
        
        return "\n".join(explanation)


if __name__ == "__main__":
    # Example usage
    response_generator = ResponseGenerator()
    
    # Example response and documents
    response = "The Transformer architecture, introduced in the paper 'Attention Is All You Need' [1], revolutionized natural language processing. It relies entirely on attention mechanisms without using recurrence or convolutions. BERT [2] later extended this approach with bidirectional training."
    
    documents = [
        {
            "paper_id": "1706.03762",
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "section": "Introduction",
            "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "score": 0.95
        },
        {
            "paper_id": "1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
            "section": "Method",
            "text": "BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017). In this work, we denote the number of layers (i.e., Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A.",
            "score": 0.85
        }
    ]
    
    query = "Explain the transformer architecture"
    
    # Format the response
    formatted_response = response_generator.format_response(response, documents, query)
    
    print(f"Query: {query}")
    print(f"Formatted response: {formatted_response['response']}")
    print("\nCitations:")
    for citation in formatted_response["citations"]:
        print(f"- {citation['title']} (ID: {citation['paper_id']})")
    
    # Generate explanation
    explanation = response_generator.generate_explanation(response, documents, query)
    print(f"\nExplanation:\n{explanation}")
