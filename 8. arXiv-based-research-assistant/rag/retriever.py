"""
Hybrid retrieval implementation.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from data.vectordb import FAISSVectorDB
from utils.config import (
    VECTOR_DB_DIR, DATA_DIR, EMBEDDING_MODEL, 
    EMBEDDING_DEVICE, NUM_DOCUMENTS, HYBRID_ALPHA
)
from utils.logger import setup_logger

logger = setup_logger("hybrid_retriever", "retriever.log")

class HybridRetriever:
    """Hybrid retrieval combining dense and sparse retrieval."""
    
    def __init__(self, 
                 vector_db: Optional[FAISSVectorDB] = None,
                 chunks_dir: Optional[Path] = None,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device: str = EMBEDDING_DEVICE,
                 num_documents: int = NUM_DOCUMENTS,
                 hybrid_alpha: float = HYBRID_ALPHA):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_db: FAISS vector database
            chunks_dir: Directory containing document chunks
            embedding_model: Name of the embedding model
            embedding_device: Device to run the model on ('cuda' or 'cpu')
            num_documents: Number of documents to retrieve
            hybrid_alpha: Weight for hybrid search (0 = BM25 only, 1 = Vector only)
        """
        self.vector_db = vector_db or FAISSVectorDB()
        self.chunks_dir = chunks_dir or (DATA_DIR / "chunks")
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.num_documents = num_documents
        self.hybrid_alpha = hybrid_alpha
        
        # Load embedding model
        logger.info(f"Loading embedding model {embedding_model} on {embedding_device}")
        self.model = SentenceTransformer(embedding_model, device=embedding_device)
        
        # Load vector database
        try:
            self.vector_db.load_index()
        except FileNotFoundError:
            logger.warning("FAISS index not found, will be created on first use")
        
        # Initialize BM25
        self._initialize_bm25()
    
    def _initialize_bm25(self) -> None:
        """Initialize BM25 index."""
        # Check if BM25 index exists
        bm25_path = VECTOR_DB_DIR / "bm25_index.json"
        
        if bm25_path.exists():
            # Load BM25 index
            logger.info(f"Loading BM25 index from {bm25_path}")
            with open(bm25_path, "r") as f:
                bm25_data = json.load(f)
            
            # Extract data
            self.corpus = bm25_data["corpus"]
            self.chunk_ids = bm25_data["chunk_ids"]
            
            # Create BM25 index
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            logger.info(f"Loaded BM25 index with {len(self.corpus)} documents")
        
        else:
            # Create BM25 index from chunks
            logger.info("Creating BM25 index from chunks")
            self.corpus = []
            self.chunk_ids = []
            
            # Load all chunks
            for chunks_file in tqdm(list(self.chunks_dir.glob("*_chunks.json")), desc="Loading chunks for BM25"):
                try:
                    with open(chunks_file, "r") as f:
                        chunks = json.load(f)
                    
                    for chunk in chunks:
                        # Prepare text for BM25
                        text = f"Title: {chunk['title']}\nAuthors: {', '.join(chunk['authors'])}\nContent: {chunk['content']}"
                        self.corpus.append(text)
                        self.chunk_ids.append(chunk["chunk_id"])
                
                except Exception as e:
                    logger.error(f"Error loading chunks from {chunks_file}: {e}")
            
            # Create BM25 index
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            # Save BM25 index
            bm25_data = {
                "corpus": self.corpus,
                "chunk_ids": self.chunk_ids
            }
            
            with open(bm25_path, "w") as f:
                json.dump(bm25_data, f)
            
            logger.info(f"Created and saved BM25 index with {len(self.corpus)} documents")
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid retrieval.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
        
        Returns:
            List of retrieved documents
        """
        k = k or self.num_documents
        
        # Get dense retrieval results
        dense_results = self._dense_retrieval(query, k=k*2)  # Get more for reranking
        
        # Get sparse retrieval results
        sparse_results = self._sparse_retrieval(query, k=k*2)  # Get more for reranking
        
        # Combine results
        combined_results = self._combine_results(dense_results, sparse_results, k=k)
        
        return combined_results
    
    def _dense_retrieval(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform dense retrieval using FAISS."""
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, k=k)
        
        return results
    
    def _sparse_retrieval(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform sparse retrieval using BM25."""
        # Tokenize query
        tokenized_query = query.split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Get results
        results = []
        for idx in top_indices:
            # Get chunk ID
            chunk_id = self.chunk_ids[idx]
            
            # Find chunk metadata
            for chunks_file in self.chunks_dir.glob("*_chunks.json"):
                try:
                    with open(chunks_file, "r") as f:
                        chunks = json.load(f)
                    
                    # Find chunk with matching ID
                    for chunk in chunks:
                        if chunk["chunk_id"] == chunk_id:
                            # Add BM25 score
                            chunk_copy = chunk.copy()
                            chunk_copy["bm25_score"] = float(scores[idx])
                            results.append(chunk_copy)
                            break
                
                except Exception as e:
                    logger.error(f"Error loading chunks from {chunks_file}: {e}")
        
        return results
    
    def _combine_results(self, 
                        dense_results: List[Dict[str, Any]], 
                        sparse_results: List[Dict[str, Any]], 
                        k: int) -> List[Dict[str, Any]]:
        """Combine dense and sparse retrieval results."""
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Process dense results
        for i, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            
            # Normalize score (lower distance is better, so invert)
            dense_score = 1.0 / (1.0 + result.get("distance", 0.0))
            
            combined_scores[chunk_id] = {
                "chunk": result,
                "dense_score": dense_score,
                "sparse_score": 0.0,
                "combined_score": self.hybrid_alpha * dense_score
            }
        
        # Process sparse results
        for i, result in enumerate(sparse_results):
            chunk_id = result["chunk_id"]
            
            # Normalize score
            sparse_score = result.get("bm25_score", 0.0)
            
            if chunk_id in combined_scores:
                # Update existing entry
                combined_scores[chunk_id]["sparse_score"] = sparse_score
                combined_scores[chunk_id]["combined_score"] += (1.0 - self.hybrid_alpha) * sparse_score
            else:
                # Create new entry
                combined_scores[chunk_id] = {
                    "chunk": result,
                    "dense_score": 0.0,
                    "sparse_score": sparse_score,
                    "combined_score": (1.0 - self.hybrid_alpha) * sparse_score
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # Get top k results
        top_results = [item["chunk"] for item in sorted_results[:k]]
        
        return top_results


if __name__ == "__main__":
    # Example usage
    retriever = HybridRetriever()
    
    # Retrieve documents
    query = "Explain the transformer architecture in deep learning"
    results = retriever.retrieve(query, k=5)
    
    # Print results
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Title: {result['title']}")
        print(f"  Authors: {', '.join(result['authors'])}")
        print(f"  Chunk ID: {result['chunk_id']}")
        print(f"  Content: {result['content'][:100]}...")
        print()
