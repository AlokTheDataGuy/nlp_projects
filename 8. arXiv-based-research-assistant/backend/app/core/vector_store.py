import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VectorIndex:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', index_path=None):
        """
        Initialize the vector index with a sentence transformer model.
        
        Args:
            embedding_model: The name of the sentence transformer model to use
            index_path: Path to load an existing index from
        """
        # Initialize the embedding model
        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize or load the index
        if index_path and os.path.exists(f"{index_path}.index"):
            self.load(index_path)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.paper_ids = []
            self.texts = []
            self.metadata = []
    
    def add_paper(self, paper_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a paper to the vector index.
        
        Args:
            paper_id: The ID of the paper
            text: The text to embed
            metadata: Additional metadata to store with the embedding
        """
        try:
            embedding = self.model.encode([text])[0]
            self.index.add(np.array([embedding], dtype=np.float32))
            self.paper_ids.append(paper_id)
            self.texts.append(text)
            self.metadata.append(metadata or {})
            return True
        except Exception as e:
            logger.error(f"Error adding paper to vector index: {e}")
            return False
    
    def add_batch(self, paper_ids: List[str], texts: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None):
        """
        Add a batch of papers to the vector index.
        
        Args:
            paper_ids: List of paper IDs
            texts: List of texts to embed
            metadata_list: List of metadata dictionaries
        """
        if metadata_list is None:
            metadata_list = [{} for _ in paper_ids]
        
        try:
            embeddings = self.model.encode(texts)
            self.index.add(np.array(embeddings, dtype=np.float32))
            self.paper_ids.extend(paper_ids)
            self.texts.extend(texts)
            self.metadata.extend(metadata_list)
            return True
        except Exception as e:
            logger.error(f"Error adding batch to vector index: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector index for similar texts.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of dictionaries containing paper_id, text, metadata, and distance
        """
        try:
            query_embedding = self.model.encode([query])[0]
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), k
            )
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.paper_ids):
                    results.append({
                        'paper_id': self.paper_ids[idx],
                        'text': self.texts[idx],
                        'metadata': self.metadata[idx],
                        'distance': float(distances[0][i])
                    })
            return results
        except Exception as e:
            logger.error(f"Error searching vector index: {e}")
            return []
    
    def save(self, filename: str):
        """
        Save the index to disk.
        
        Args:
            filename: Base filename to save to (without extension)
        """
        try:
            # Save the FAISS index
            faiss.write_index(self.index, f"{filename}.index")
            
            # Save metadata separately
            with open(f"{filename}.meta", 'wb') as f:
                pickle.dump({
                    'paper_ids': self.paper_ids,
                    'texts': self.texts,
                    'metadata': self.metadata
                }, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving vector index: {e}")
            return False
    
    def load(self, filename: str):
        """
        Load the index from disk.
        
        Args:
            filename: Base filename to load from (without extension)
        """
        try:
            # Load the FAISS index
            self.index = faiss.read_index(f"{filename}.index")
            
            # Load metadata
            with open(f"{filename}.meta", 'rb') as f:
                meta = pickle.load(f)
                self.paper_ids = meta['paper_ids']
                self.texts = meta['texts']
                self.metadata = meta.get('metadata', [{} for _ in self.paper_ids])
            
            return True
        except Exception as e:
            logger.error(f"Error loading vector index: {e}")
            return False
