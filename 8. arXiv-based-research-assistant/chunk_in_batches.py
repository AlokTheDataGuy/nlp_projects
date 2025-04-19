"""
Script to chunk documents in smaller batches to avoid memory issues.
"""
import os
import argparse
import time
from pathlib import Path
from typing import List, Optional

from data.chunker import DocumentChunker
from utils.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("batch_chunker", "batch_chunker.log")

def chunk_in_batches(batch_size: int = 20, delay: int = 5):
    """
    Chunk documents in smaller batches to avoid memory issues.
    
    Args:
        batch_size: Number of papers to process in each batch
        delay: Delay in seconds between batches to allow memory to be freed
    """
    # Initialize chunker
    chunker = DocumentChunker()
    
    # Get list of all processed papers
    input_dir = DATA_DIR / "processed" / "text"
    all_paper_ids = [f.stem for f in input_dir.glob("*.json")]
    
    # Filter out already chunked papers
    chunked_papers = chunker.get_chunked_papers()
    papers_to_chunk = [pid for pid in all_paper_ids if pid not in chunked_papers]
    
    logger.info(f"Found {len(papers_to_chunk)} papers to chunk")
    
    # Process in batches
    total_batches = (len(papers_to_chunk) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(papers_to_chunk))
        
        batch = papers_to_chunk[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_num+1}/{total_batches} with {len(batch)} papers")
        
        # Process batch
        successful = chunker.chunk_documents(batch)
        
        logger.info(f"Batch {batch_num+1}/{total_batches} completed. Processed {len(successful)}/{len(batch)} papers successfully")
        
        # Delay between batches to allow memory to be freed
        if batch_num < total_batches - 1:
            logger.info(f"Waiting {delay} seconds before next batch...")
            time.sleep(delay)
    
    logger.info("All batches processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk documents in batches")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of papers to process in each batch")
    parser.add_argument("--delay", type=int, default=5, help="Delay in seconds between batches")
    
    args = parser.parse_args()
    
    chunk_in_batches(args.batch_size, args.delay)
