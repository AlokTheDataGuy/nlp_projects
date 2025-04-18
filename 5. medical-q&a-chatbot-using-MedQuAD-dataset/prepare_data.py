import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from embedding import TextEmbedder
from retrieval import FAISSRetriever
from utils import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/embeddings',
        'data/faiss'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def prepare_dataset():
    """Prepare the dataset for use"""
    try:
        # Load the dataset
        logger.info("Loading dataset...")
        dataset = load_dataset()
        logger.info(f"Dataset loaded with {len(dataset)} QA pairs")
        
        # Check if we need to clean or preprocess the data
        logger.info("Checking dataset for missing values...")
        missing_questions = dataset['question'].isna().sum()
        missing_answers = dataset['answer'].isna().sum()
        
        if missing_questions > 0 or missing_answers > 0:
            logger.warning(f"Found {missing_questions} missing questions and {missing_answers} missing answers")
            logger.info("Removing rows with missing values...")
            dataset = dataset.dropna(subset=['question', 'answer'])
            logger.info(f"Dataset now has {len(dataset)} QA pairs")
        
        # Save the cleaned dataset
        output_path = 'data/processed/medquad_cleaned.csv'
        dataset.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned dataset to {output_path}")
        
        return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise

def build_embeddings(dataset):
    """Build embeddings for the dataset"""
    try:
        # Initialize the embedder
        logger.info("Initializing text embedder...")
        embedder = TextEmbedder()
        
        # Check if embeddings already exist
        embeddings_path = 'data/embeddings/question_embeddings.npy'
        if os.path.exists(embeddings_path):
            logger.info(f"Embeddings already exist at {embeddings_path}")
            logger.info("Loading existing embeddings...")
            embeddings = np.load(embeddings_path)
            
            # Check if the number of embeddings matches the dataset
            if len(embeddings) != len(dataset):
                logger.warning(f"Number of embeddings ({len(embeddings)}) doesn't match dataset size ({len(dataset)})")
                logger.info("Rebuilding embeddings...")
                embeddings = embedder.build_question_embeddings(dataset, embeddings_path)
        else:
            logger.info("Building new embeddings...")
            embeddings = embedder.build_question_embeddings(dataset, embeddings_path)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error building embeddings: {e}")
        raise

def build_faiss_index(dataset, embeddings):
    """Build FAISS index for the dataset"""
    try:
        # Initialize the retriever
        logger.info("Initializing FAISS retriever...")
        retriever = FAISSRetriever()
        
        # Check if index already exists
        index_path = 'data/faiss/question_index.faiss'
        mapping_path = 'data/faiss/id_mapping.json'
        
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            logger.info(f"FAISS index already exists at {index_path}")
            overwrite = input("Do you want to rebuild the index? (y/n): ").lower() == 'y'
            
            if not overwrite:
                logger.info("Using existing index")
                return
        
        # Build the index
        logger.info("Building FAISS index...")
        retriever.build_index(embeddings, dataset, 'data/faiss')
        logger.info("FAISS index built successfully")
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise

def main():
    """Main function to prepare data and build index"""
    parser = argparse.ArgumentParser(description='Prepare data and build index for the Medical Q&A Chatbot')
    parser.add_argument('--force', action='store_true', help='Force rebuild of embeddings and index')
    args = parser.parse_args()
    
    try:
        # Create necessary directories
        create_directories()
        
        # Prepare the dataset
        dataset = prepare_dataset()
        
        # Build embeddings
        embeddings = build_embeddings(dataset)
        
        # Build FAISS index
        build_faiss_index(dataset, embeddings)
        
        logger.info("Data preparation completed successfully")
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
