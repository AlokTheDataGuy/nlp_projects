# scripts/download_papers.py

"""
Download Papers Script

This script downloads papers from arXiv based on the configured categories.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.paper_processor import ArxivPaperProcessor
from src.knowledge_base.document_store import DocumentStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to download papers from arXiv.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download papers from arXiv")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of papers to download")
    parser.add_argument("--config", type=str, default="config/app_config.yaml", help="Path to configuration file")
    parser.add_argument("--store", action="store_true", help="Store papers in MongoDB")
    args = parser.parse_args()

    try:
        # Initialize paper processor
        processor = ArxivPaperProcessor(config_path=args.config)

        # Download papers
        logger.info(f"Downloading up to {args.limit} papers...")
        papers = processor.download_papers(limit=args.limit)

        logger.info(f"Downloaded {len(papers)} papers")

        # Store papers in MongoDB if requested
        if args.store:
            logger.info("Storing papers in MongoDB...")
            document_store = DocumentStore()

            # Insert papers
            for paper in papers:
                # Check if paper already exists - fixed comparison method
                existing_paper = document_store.find_document("papers", {"id": paper["id"]})
                if existing_paper is not None:  # Compare with None instead of truth test
                    logger.info(f"Paper {paper['id']} already exists in database, skipping")
                    continue

                # Insert paper
                document_store.insert_document("papers", paper)
                logger.info(f"Inserted paper {paper['id']} into database")

            logger.info("Finished storing papers in MongoDB")

        logger.info("Done")

    except Exception as e:
        logger.error(f"Error downloading papers: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
