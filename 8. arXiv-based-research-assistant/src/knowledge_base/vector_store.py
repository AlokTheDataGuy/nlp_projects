"""
Vector Store Module

This module provides an interface to the FAISS vector database.
"""

import os
import logging
import yaml
import numpy as np
import faiss
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Class for interacting with the FAISS vector database.
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml", model_config_path: str = "config/model_config.yaml"):
        """
        Initialize the VectorStore.
        
        Args:
            config_path: Path to the application configuration file.
            model_config_path: Path to the model configuration file.
        """
        self.config = self._load_config(config_path)
        self.model_config = self._load_config(model_config_path)
        
        self.embeddings_dir = Path("data/embeddings")
        self.index_path = Path(self.config["database"]["vector_db"]["index_path"])
        
        # Load the embedding model
        self.model_id = self.model_config["embedding"]["model_id"]
        self.device = self.model_config["embedding"]["device"]
        self.normalize = self.model_config["embedding"]["normalize"]
        self.model = self._load_model()
        
        # Initialize indices and metadata
        self.indices = {}
        self.metadata = {}
        self._load_indices()
    
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
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load the Sentence-Transformers model.
        
        Returns:
            Loaded model.
        """
        try:
            import torch
            
            # Check if CUDA is available
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Load the model
            model = SentenceTransformer(self.model_id, device=self.device)
            logger.info(f"Loaded embedding model: {self.model_id} on {self.device}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _load_indices(self) -> None:
        """
        Load FAISS indices and metadata.
        """
        index_types = ["chunks", "abstract", "title", "sections"]
        
        for index_type in index_types:
            index_path = self.embeddings_dir / f"faiss_index_{index_type}"
            metadata_path = self.embeddings_dir / f"faiss_index_{index_type}_metadata.yaml"
            
            if index_path.exists() and metadata_path.exists():
                try:
                    # Load the index
                    self.indices[index_type] = faiss.read_index(str(index_path))
                    
                    # Load the metadata
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata[index_type] = yaml.safe_load(f)
                    
                    logger.info(f"Loaded FAISS index for {index_type} with {self.indices[index_type].ntotal} vectors")
                except Exception as e:
                    logger.error(f"Error loading FAISS index for {index_type}: {str(e)}")
            else:
                logger.warning(f"FAISS index for {index_type} not found")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a text.
        
        Args:
            text: Text to generate embedding for.
            
        Returns:
            NumPy array of embedding.
        """
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.normalize
            )
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def search(self, query: str, index_type: str = "chunks", k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar items in the vector store.
        
        Args:
            query: Query text.
            index_type: Type of index to search ("chunks", "abstract", "title", "sections").
            k: Number of results to return.
            
        Returns:
            List of dictionaries containing search results with metadata.
        """
        if index_type not in self.indices:
            logger.error(f"Index type {index_type} not found")
            return []
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Convert to the right format
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.indices[index_type].search(query_embedding, k)
        
        # Get metadata for results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata[index_type]):
                result = {
                    **self.metadata[index_type][idx],
                    "distance": float(distances[0][i]),
                    "score": 1.0 / (1.0 + float(distances[0][i]))  # Convert distance to similarity score
                }
                results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, k: int = 5, semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Query text.
            k: Number of results to return.
            semantic_weight: Weight for semantic search results.
            keyword_weight: Weight for keyword matching.
            
        Returns:
            List of dictionaries containing search results with metadata.
        """
        # Semantic search
        semantic_results = self.search(query, index_type="chunks", k=k*2)  # Get more results for reranking
        
        # Keyword matching
        query_terms = set(query.lower().split())
        
        # Rerank results using hybrid approach
        for result in semantic_results:
            # Calculate keyword match score
            text_terms = set(result["text"].lower().split())
            matching_terms = query_terms.intersection(text_terms)
            keyword_score = len(matching_terms) / max(1, len(query_terms))
            
            # Combine scores
            semantic_score = result["score"]
            combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
            
            result["semantic_score"] = semantic_score
            result["keyword_score"] = keyword_score
            result["combined_score"] = combined_score
        
        # Sort by combined score
        reranked_results = sorted(semantic_results, key=lambda x: x["combined_score"], reverse=True)
        
        # Return top k results
        return reranked_results[:k]
    
    def add_to_index(self, texts: List[str], metadata_list: List[Dict[str, Any]], index_type: str = "chunks") -> None:
        """
        Add new items to the index.
        
        Args:
            texts: List of texts to add.
            metadata_list: List of metadata dictionaries for each text.
            index_type: Type of index to add to.
        """
        if index_type not in self.indices:
            logger.error(f"Index type {index_type} not found")
            return
        
        if len(texts) != len(metadata_list):
            logger.error("Number of texts and metadata entries must match")
            return
        
        # Generate embeddings
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to index
        self.indices[index_type].add(embeddings_array)
        
        # Add to metadata
        self.metadata[index_type].extend(metadata_list)
        
        # Save updated index and metadata
        index_path = self.embeddings_dir / f"faiss_index_{index_type}"
        metadata_path = self.embeddings_dir / f"faiss_index_{index_type}_metadata.yaml"
        
        faiss.write_index(self.indices[index_type], str(index_path))
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.metadata[index_type], f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Added {len(texts)} items to {index_type} index")


if __name__ == "__main__":
    # Example usage
    vector_store = VectorStore()
    
    # Search for a query
    query = "transformer architecture for natural language processing"
    results = vector_store.hybrid_search(query, k=5)
    
    print(f"Search results for: {query}")
    for i, result in enumerate(results):
        print(f"{i+1}. Paper: {result['title']} (ID: {result['paper_id']})")
        print(f"   Section: {result.get('section', 'N/A')}")
        print(f"   Score: {result['combined_score']:.4f} (Semantic: {result['semantic_score']:.4f}, Keyword: {result['keyword_score']:.4f})")
        print(f"   Text: {result['text'][:200]}...")
        print()
