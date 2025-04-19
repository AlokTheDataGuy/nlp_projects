"""
Script to generate embeddings in smaller batches to avoid memory issues.
"""
import os
import argparse
import time
from pathlib import Path
from typing import List, Optional

from data.embedder import DocumentEmbedder
from utils.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("batch_embedder", "batch_embedder.log")

def embed_in_batches(batch_size: int = 10, delay: int = 10):
    """
    Generate embeddings in smaller batches to avoid memory issues.
    
    Args:
        batch_size: Number of papers to process in each batch
        delay: Delay in seconds between batches to allow memory to be freed
    """
    # Initialize embedder
    embedder = DocumentEmbedder()
    
    # Get list of all chunked papers
    chunks_dir = DATA_DIR / "chunks"
    all_paper_ids = [f.stem.replace("_chunks", "") for f in chunks_dir.glob("*_chunks.json")]
    
    # Filter out already embedded papers
    embedded_papers = embedder.get_processed_papers()
    papers_to_embed = [pid for pid in all_paper_ids if pid not in embedded_papers]
    
    logger.info(f"Found {len(papers_to_embed)} papers to embed")
    
    # Process in batches
    total_batches = (len(papers_to_embed) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(papers_to_embed))
        
        batch = papers_to_embed[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_num+1}/{total_batches} with {len(batch)} papers")
        
        # Process batch
        successful = embedder.generate_embeddings(batch)
        
        logger.info(f"Batch {batch_num+1}/{total_batches} completed. Processed {len(successful)}/{len(batch)} papers successfully")
        
        # Delay between batches to allow memory to be freed
        if batch_num < total_batches - 1:
            logger.info(f"Waiting {delay} seconds before next batch...")
            time.sleep(delay)
    
    logger.info("All batches processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings in batches")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of papers to process in each batch")
    parser.add_argument("--delay", type=int, default=10, help="Delay in seconds between batches")
    
    args = parser.parse_args()
    
    embed_in_batches(args.batch_size, args.delay)
