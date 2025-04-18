import pandas as pd
import numpy as np
import os
import logging
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
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset()
        logger.info(f"Dataset loaded with {len(dataset)} QA pairs")
        
        # Initialize embedder
        logger.info("Initializing text embedder...")
        embedder = TextEmbedder()
        
        # Build embeddings
        embeddings_path = 'data/embeddings/question_embeddings.npy'
        if os.path.exists(embeddings_path):
            logger.info("Loading existing embeddings...")
            embeddings = np.load(embeddings_path)
        else:
            logger.info("Building new embeddings...")
            embeddings = embedder.build_question_embeddings(dataset)
        
        # Initialize retriever
        logger.info("Initializing FAISS retriever...")
        retriever = FAISSRetriever()
        
        # Build index
        logger.info("Building FAISS index...")
        retriever.build_index(embeddings, dataset)
        
        logger.info("Index building completed successfully")
    except Exception as e:
        logger.error(f"Error building index: {e}")

if __name__ == "__main__":
    main()
