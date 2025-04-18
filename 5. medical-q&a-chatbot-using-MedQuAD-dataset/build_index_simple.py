import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from embedding import TextEmbedder
from retrieval import FAISSRetriever
from utils import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Build embeddings and FAISS index for the dataset"""
    try:
        # Create necessary directories
        os.makedirs('data/embeddings', exist_ok=True)
        os.makedirs('data/faiss', exist_ok=True)
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset()
        logger.info(f"Dataset loaded with {len(dataset)} QA pairs")
        
        # Clean dataset
        logger.info("Cleaning dataset...")
        dataset = dataset.dropna(subset=['question', 'answer'])
        logger.info(f"Dataset now has {len(dataset)} QA pairs")
        
        # Initialize embedder
        logger.info("Initializing text embedder...")
        embedder = TextEmbedder()
        
        # Build embeddings
        logger.info("Building embeddings...")
        embeddings = embedder.build_question_embeddings(dataset)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Initialize retriever
        logger.info("Initializing FAISS retriever...")
        retriever = FAISSRetriever()
        
        # Build index
        logger.info("Building FAISS index...")
        retriever.build_index(embeddings, dataset)
        
        logger.info("Index building completed successfully")
    except Exception as e:
        logger.error(f"Error building index: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
