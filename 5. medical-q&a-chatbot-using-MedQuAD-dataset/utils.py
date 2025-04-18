import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def load_dataset(file_path='data/processed/medquad_complete.csv'):
    """Load the processed MedQuAD dataset"""
    return pd.read_csv(file_path)

def filter_by_semantic_type(candidates, query_semantic_types, threshold=0.3):
    """
    Filter candidate answers based on semantic type overlap

    Args:
        candidates: List of candidate QA pairs
        query_semantic_types: List of semantic types from the query
        threshold: Minimum overlap ratio required

    Returns:
        Filtered list of candidates
    """
    if not query_semantic_types:
        return candidates

    filtered = []
    for candidate in candidates:
        candidate_types = candidate.get('semantic_types', [])
        if not candidate_types:
            filtered.append(candidate)
            continue

        # Calculate overlap
        common_types = set(query_semantic_types).intersection(set(candidate_types))
        if len(common_types) / max(len(query_semantic_types), 1) >= threshold:
            filtered.append(candidate)

    return filtered if filtered else candidates  # Return original if all filtered out

def filter_by_entity_overlap(candidates, query_entities, threshold=0.2):
    """
    Filter candidate answers based on entity overlap

    Args:
        candidates: List of candidate QA pairs
        query_entities: List of entities from the query
        threshold: Minimum overlap ratio required

    Returns:
        Filtered list of candidates
    """
    if not query_entities:
        return candidates

    filtered = []
    for candidate in candidates:
        candidate_entities = candidate.get('entities', [])
        if not candidate_entities:
            filtered.append(candidate)
            continue

        # Calculate overlap
        common_entities = set(query_entities).intersection(set(candidate_entities))
        if len(common_entities) / max(len(query_entities), 1) >= threshold:
            filtered.append(candidate)

    return filtered if filtered else candidates  # Return original if all filtered out

def save_vector_store(vector_store, index_path, mapping_path):
    """Save FAISS index and id mapping"""
    import faiss

    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Use faiss.write_index instead of the save method
    faiss.write_index(vector_store['index'], index_path)

    with open(mapping_path, 'w') as f:
        json.dump(vector_store['id_map'], f)

def load_vector_store(index_path, mapping_path):
    """Load FAISS index and id mapping"""
    import faiss
    import logging

    logger = logging.getLogger(__name__)

    try:
        index = faiss.read_index(index_path)

        with open(mapping_path, 'r') as f:
            id_map = json.load(f)

        logger.info(f"Successfully loaded FAISS index from {index_path}")
        return {'index': index, 'id_map': id_map}
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise
