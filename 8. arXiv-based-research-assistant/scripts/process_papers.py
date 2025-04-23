"""
Process Papers Script

This script processes downloaded papers, extracts text, generates features, and creates embeddings.
"""

import os
import sys
import logging
import argparse
import glob
import yaml
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.text_extractor import TextExtractor
from src.data_pipeline.feature_generator import FeatureGenerator
from src.data_pipeline.embedding_creator import EmbeddingCreator
from src.knowledge_base.document_store import DocumentStore
from src.knowledge_base.knowledge_graph import KnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to process papers.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process papers")
    parser.add_argument("--config", type=str, default="config/app_config.yaml", help="Path to configuration file")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml", help="Path to model configuration file")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of papers to process")
    parser.add_argument("--store", action="store_true", help="Store processed papers in MongoDB")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip text extraction")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature generation")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding creation")
    parser.add_argument("--skip-graph", action="store_true", help="Skip knowledge graph creation")
    args = parser.parse_args()
    
    try:
        # Initialize document store
        document_store = DocumentStore()
        
        # Get papers from MongoDB
        papers = document_store.find_documents("papers", {})
        
        if args.limit:
            papers = papers[:args.limit]
        
        logger.info(f"Processing {len(papers)} papers")
        
        # Process papers
        processed_papers = papers
        
        # Extract text
        if not args.skip_extraction:
            logger.info("Extracting text from papers...")
            text_extractor = TextExtractor(config_path=args.config)
            processed_papers = text_extractor.process_papers(processed_papers)
        
        # Generate features
        if not args.skip_features:
            logger.info("Generating features from papers...")
            feature_generator = FeatureGenerator(config_path=args.config)
            processed_papers = feature_generator.process_papers(processed_papers)
        
        # Create embeddings
        if not args.skip_embeddings:
            logger.info("Creating embeddings for papers...")
            embedding_creator = EmbeddingCreator(config_path=args.model_config)
            processed_papers = embedding_creator.create_papers_embeddings(processed_papers)
            
            # Build FAISS indices
            logger.info("Building FAISS indices...")
            for index_type in ["chunks", "abstract", "title", "sections"]:
                embedding_creator.build_faiss_index(processed_papers, index_type=index_type)
        
        # Create knowledge graph
        if not args.skip_graph:
            logger.info("Building knowledge graph...")
            knowledge_graph = KnowledgeGraph(config_path=args.config)
            knowledge_graph.build_graph_from_papers(processed_papers)
        
        # Store processed papers in MongoDB
        if args.store:
            logger.info("Storing processed papers in MongoDB...")
            for paper in processed_papers:
                # Update paper in database
                document_store.update_document("papers", {"id": paper["id"]}, paper)
                logger.info(f"Updated paper {paper['id']} in database")
        
        logger.info("Done")
    
    except Exception as e:
        logger.error(f"Error processing papers: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
