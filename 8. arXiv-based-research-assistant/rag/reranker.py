"""
Result re-ranking for improved retrieval quality.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import CrossEncoder

from utils.config import RERANK_TOP_N
from utils.logger import setup_logger

logger = setup_logger("reranker", "reranker.log")

class Reranker:
    """Re-ranks retrieval results for improved quality."""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 top_n: int = RERANK_TOP_N):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            top_n: Number of documents to rerank
        """
        self.model_name = model_name
        self.top_n = top_n
        
        # Load model
        logger.info(f"Loading cross-encoder model {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded successfully")
    
    def rerank(self, 
              query: str, 
              documents: List[Dict[str, Any]], 
              top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on relevance to query.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_n: Number of documents to return after reranking
        
        Returns:
            List of reranked documents
        """
        top_n = top_n or self.top_n
        
        # Limit to top_n documents if there are more
        documents_to_rerank = documents[:min(len(documents), self.top_n)]
        
        # Prepare document-query pairs
        pairs = []
        for doc in documents_to_rerank:
            # Create a text representation of the document
            doc_text = f"Title: {doc['title']}\nAuthors: {', '.join(doc['authors'])}\nContent: {doc['content']}"
            pairs.append([query, doc_text])
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for i, doc in enumerate(documents_to_rerank):
            doc["rerank_score"] = float(scores[i])
        
        # Sort by score
        reranked_documents = sorted(documents_to_rerank, key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top_n documents
        return reranked_documents[:top_n]


if __name__ == "__main__":
    # Example usage
    reranker = Reranker()
    
    # Sample documents
    documents = [
        {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "content": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "content": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers."
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "authors": ["Kaiming He", "Xiangyu Zhang"],
            "content": "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously."
        }
    ]
    
    # Rerank documents
    query = "How do transformers work in NLP?"
    reranked_docs = reranker.rerank(query, documents)
    
    # Print results
    for i, doc in enumerate(reranked_docs):
        print(f"Result {i+1} (Score: {doc['rerank_score']:.4f}):")
        print(f"  Title: {doc['title']}")
        print(f"  Authors: {', '.join(doc['authors'])}")
        print(f"  Content: {doc['content']}")
        print()
