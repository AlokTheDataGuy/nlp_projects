import faiss
import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path
from utils import save_vector_store, load_vector_store, filter_by_semantic_type, filter_by_entity_overlap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSRetriever:
    def __init__(self, embedding_dim=384):
        """
        Initialize the FAISS retriever

        Args:
            embedding_dim: Dimension of the embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_map = None
        self.df = None

    def build_index(self, embeddings, df, save_path='data/faiss'):
        """
        Build a FAISS index from embeddings

        Args:
            embeddings: Numpy array of embeddings
            df: DataFrame containing the dataset
            save_path: Directory to save the index and mapping

        Returns:
            Dictionary containing the index and id mapping
        """
        logger.info(f"Building FAISS index with {len(embeddings)} vectors")

        # Create a new index
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add vectors to the index
        index.add(embeddings)

        # Create a mapping from index positions to dataframe indices
        id_map = {i: int(idx) for i, idx in enumerate(df.index)}

        # Store the index and mapping
        self.index = index
        self.id_map = id_map
        self.df = df

        # Save the index and mapping
        os.makedirs(save_path, exist_ok=True)
        index_path = os.path.join(save_path, 'question_index.faiss')
        mapping_path = os.path.join(save_path, 'id_mapping.json')
        save_vector_store({'index': index, 'id_map': id_map}, index_path, mapping_path)

        logger.info(f"FAISS index built and saved to {save_path}")

        return {'index': index, 'id_map': id_map}

    def load_index(self, index_path='data/faiss/question_index.faiss',
                  mapping_path='data/faiss/id_mapping.json', df=None):
        """
        Load a FAISS index from disk

        Args:
            index_path: Path to the FAISS index
            mapping_path: Path to the id mapping
            df: DataFrame containing the dataset

        Returns:
            True if successful
        """
        logger.info(f"Loading FAISS index from {index_path}")

        vector_store = load_vector_store(index_path, mapping_path)
        self.index = vector_store['index']
        self.id_map = vector_store['id_map']
        self.df = df

        logger.info("FAISS index loaded successfully")
        return True

    def search(self, query_embedding, top_k=5):
        """
        Search the index for similar vectors

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (score, index) tuples
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call build_index or load_index first.")

        # Normalize query vector for cosine similarity
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)

        # Map indices to dataframe indices
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no result
                # Handle both string and integer keys in id_map
                try:
                    df_idx = self.id_map[str(idx)]
                except KeyError:
                    try:
                        df_idx = self.id_map[idx]
                    except KeyError:
                        logger.warning(f"Index {idx} not found in id_map")
                        continue
                results.append((float(scores[0][i]), df_idx))

        return results

    def retrieve_answers(self, query_embedding, ner_results, top_k=5):
        """
        Retrieve and rank answers based on query embedding and NER results

        Args:
            query_embedding: Query embedding vector
            ner_results: NER results from the query
            top_k: Number of results to return

        Returns:
            List of answer dictionaries
        """
        if self.df is None:
            raise ValueError("DataFrame not initialized. Call build_index or load_index with a DataFrame.")

        # Search for similar questions
        search_results = self.search(query_embedding, top_k=top_k*2)  # Get more results for filtering

        # Extract candidate answers
        candidates = []
        for score, idx in search_results:
            row = self.df.iloc[idx]
            candidate = {
                'question': row['question'],
                'answer': row['answer'],
                'score': score,
                'source': row['source'],
                'semantic_types': row.get('semantic_types', []),
                'entities': row.get('entities', [])
            }
            candidates.append(candidate)

        # Filter by semantic type
        if 'semantic_types' in ner_results and ner_results['semantic_types']:
            candidates = filter_by_semantic_type(candidates, ner_results['semantic_types'])

        # Filter by entity overlap
        if 'entities' in ner_results and ner_results['entities']:
            candidates = filter_by_entity_overlap(candidates, ner_results['entities'])

        # Sort by score and return top_k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]
