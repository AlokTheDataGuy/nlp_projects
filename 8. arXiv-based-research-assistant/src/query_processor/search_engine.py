"""
Search Engine Module

This module provides semantic and keyword search functionality.
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

class SearchEngine:
    """
    Class for providing semantic and keyword search functionality.
    """
    
    def __init__(self, vector_store, document_store, config_path: str = "config/app_config.yaml"):
        """
        Initialize the SearchEngine.
        
        Args:
            vector_store: Vector store for semantic search.
            document_store: Document store for keyword search.
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.vector_store = vector_store
        self.document_store = document_store
    
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
    
    def search(self, query: str, search_type: str = "hybrid", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers matching the query.
        
        Args:
            query: The search query.
            search_type: Type of search to perform ("semantic", "keyword", "hybrid").
            limit: Maximum number of results to return.
            
        Returns:
            List of search results.
        """
        if search_type == "semantic":
            return self.semantic_search(query, limit=limit)
        elif search_type == "keyword":
            return self.keyword_search(query, limit=limit)
        else:  # hybrid
            return self.hybrid_search(query, limit=limit)
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search using the vector store.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            
        Returns:
            List of search results.
        """
        # Use the vector store to search for papers
        results = self.vector_store.search(
            query,
            index_type="abstract",  # Search in abstracts for paper search
            k=limit
        )
        
        # Format results
        formatted_results = []
        
        for result in results:
            paper_id = result.get("paper_id", "")
            
            # Get full paper data from document store
            paper_data = self.document_store.find_document("papers", {"id": paper_id})
            
            if paper_data:
                formatted_result = {
                    "id": paper_id,
                    "title": paper_data.get("title", result.get("title", "")),
                    "authors": paper_data.get("authors", []),
                    "abstract": paper_data.get("abstract", result.get("text", "")),
                    "categories": paper_data.get("categories", []),
                    "published": paper_data.get("published", ""),
                    "score": result.get("score", 0),
                    "search_type": "semantic"
                }
                
                formatted_results.append(formatted_result)
        
        return formatted_results
    
    def keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform keyword search using the document store.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            
        Returns:
            List of search results.
        """
        # Parse query into keywords
        keywords = self._parse_keywords(query)
        
        # Create MongoDB query
        mongo_query = {"$or": []}
        
        for keyword in keywords:
            # Search in title, abstract, and full text
            mongo_query["$or"].extend([
                {"title": {"$regex": keyword, "$options": "i"}},
                {"abstract": {"$regex": keyword, "$options": "i"}},
                {"full_text": {"$regex": keyword, "$options": "i"}}
            ])
        
        # Search in document store
        papers = self.document_store.find_documents("papers", mongo_query, limit=limit)
        
        # Format results
        formatted_results = []
        
        for paper in papers:
            # Calculate simple relevance score based on keyword frequency
            score = self._calculate_keyword_score(paper, keywords)
            
            formatted_result = {
                "id": paper.get("id", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "abstract": paper.get("abstract", ""),
                "categories": paper.get("categories", []),
                "published": paper.get("published", ""),
                "score": score,
                "search_type": "keyword"
            }
            
            formatted_results.append(formatted_result)
        
        # Sort by score
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        
        return formatted_results
    
    def hybrid_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            
        Returns:
            List of search results.
        """
        # Get results from both search methods
        semantic_results = self.semantic_search(query, limit=limit*2)
        keyword_results = self.keyword_search(query, limit=limit*2)
        
        # Combine results
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            paper_id = result["id"]
            if paper_id not in combined_results:
                combined_results[paper_id] = result
                combined_results[paper_id]["semantic_score"] = result["score"]
                combined_results[paper_id]["keyword_score"] = 0
            else:
                combined_results[paper_id]["semantic_score"] = result["score"]
        
        # Add keyword results
        for result in keyword_results:
            paper_id = result["id"]
            if paper_id not in combined_results:
                combined_results[paper_id] = result
                combined_results[paper_id]["semantic_score"] = 0
                combined_results[paper_id]["keyword_score"] = result["score"]
            else:
                combined_results[paper_id]["keyword_score"] = result["score"]
        
        # Calculate combined score
        for paper_id, result in combined_results.items():
            semantic_score = result.get("semantic_score", 0)
            keyword_score = result.get("keyword_score", 0)
            
            # Weighted combination
            combined_score = (0.7 * semantic_score) + (0.3 * keyword_score)
            
            result["score"] = combined_score
            result["search_type"] = "hybrid"
        
        # Convert to list and sort by score
        results_list = list(combined_results.values())
        results_list.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit results
        return results_list[:limit]
    
    def _parse_keywords(self, query: str) -> List[str]:
        """
        Parse a query into keywords.
        
        Args:
            query: The query to parse.
            
        Returns:
            List of keywords.
        """
        # Remove special characters and convert to lowercase
        cleaned_query = re.sub(r'[^\w\s]', ' ', query.lower())
        
        # Split into words
        words = cleaned_query.split()
        
        # Remove stopwords
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "from", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "to", "of", "in", "on", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "having", "do", "does", "did", "doing",
            "can", "could", "should", "would", "may", "might", "must", "shall", "will"
        }
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def _calculate_keyword_score(self, paper: Dict[str, Any], keywords: List[str]) -> float:
        """
        Calculate a relevance score based on keyword frequency.
        
        Args:
            paper: Paper data.
            keywords: List of keywords.
            
        Returns:
            Relevance score.
        """
        score = 0
        
        # Check title (highest weight)
        title = paper.get("title", "").lower()
        for keyword in keywords:
            if keyword in title:
                score += 3
        
        # Check abstract (medium weight)
        abstract = paper.get("abstract", "").lower()
        for keyword in keywords:
            if keyword in abstract:
                score += 2
        
        # Check full text (lowest weight)
        full_text = paper.get("full_text", "").lower()
        for keyword in keywords:
            if keyword in full_text:
                score += 1
        
        # Normalize score
        max_possible_score = 6 * len(keywords)  # 3 + 2 + 1 per keyword
        normalized_score = score / max_possible_score if max_possible_score > 0 else 0
        
        return normalized_score
    
    def search_by_paper_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Search for a paper by its ID.
        
        Args:
            paper_id: ID of the paper to search for.
            
        Returns:
            Paper data or None if not found.
        """
        paper = self.document_store.find_document("papers", {"id": paper_id})
        
        if paper:
            return {
                "id": paper.get("id", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "abstract": paper.get("abstract", ""),
                "categories": paper.get("categories", []),
                "published": paper.get("published", ""),
                "full_text": paper.get("full_text", ""),
                "sections": paper.get("sections", {})
            }
        
        return None
    
    def search_related_papers(self, paper_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers related to a given paper.
        
        Args:
            paper_id: ID of the paper to find related papers for.
            limit: Maximum number of results to return.
            
        Returns:
            List of related papers.
        """
        # Get the paper
        paper = self.document_store.find_document("papers", {"id": paper_id})
        
        if not paper:
            return []
        
        # Use the title and abstract as the query
        query = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        
        # Perform semantic search
        results = self.semantic_search(query, limit=limit+1)
        
        # Remove the original paper from results
        results = [result for result in results if result["id"] != paper_id]
        
        # Limit results
        return results[:limit]


if __name__ == "__main__":
    # Example usage
    from src.knowledge_base.vector_store import VectorStore
    from src.knowledge_base.document_store import DocumentStore
    
    vector_store = VectorStore()
    document_store = DocumentStore()
    
    search_engine = SearchEngine(vector_store, document_store)
    
    # Test search
    query = "transformer architecture for natural language processing"
    
    print(f"Semantic search for: {query}")
    semantic_results = search_engine.semantic_search(query, limit=3)
    for i, result in enumerate(semantic_results):
        print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Authors: {', '.join(result['authors'][:3])}")
        print(f"   Abstract: {result['abstract'][:200]}...")
        print()
    
    print(f"Hybrid search for: {query}")
    hybrid_results = search_engine.hybrid_search(query, limit=3)
    for i, result in enumerate(hybrid_results):
        print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Authors: {', '.join(result['authors'][:3])}")
        print(f"   Abstract: {result['abstract'][:200]}...")
        print()
