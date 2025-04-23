"""
Setup Database Script

This script sets up the MongoDB database for the arXiv research assistant.
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_base.document_store import DocumentStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to set up the MongoDB database.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Set up the MongoDB database")
    parser.add_argument("--config", type=str, default="config/app_config.yaml", help="Path to configuration file")
    parser.add_argument("--drop", action="store_true", help="Drop existing collections")
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get database configuration
        db_config = config["database"]["mongodb"]
        
        # Initialize document store
        document_store = DocumentStore(config_path=args.config)
        
        # Get MongoDB client and database
        client = document_store.client
        db = document_store.db
        
        # Check connection
        try:
            client.admin.command('ping')
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            sys.exit(1)
        
        # Set up collections
        collections = db_config["collections"]
        
        for collection_name in collections.values():
            # Check if collection exists
            if collection_name in db.list_collection_names():
                if args.drop:
                    logger.info(f"Dropping collection: {collection_name}")
                    db.drop_collection(collection_name)
                else:
                    logger.info(f"Collection already exists: {collection_name}")
                    continue
            
            # Create collection
            logger.info(f"Creating collection: {collection_name}")
            db.create_collection(collection_name)
            
            # Create indexes
            if collection_name == "papers":
                logger.info("Creating indexes for papers collection")
                db[collection_name].create_index("id", unique=True)
                db[collection_name].create_index("title")
                db[collection_name].create_index("authors")
                db[collection_name].create_index("categories")
                db[collection_name].create_index("published")
            
            elif collection_name == "embeddings":
                logger.info("Creating indexes for embeddings collection")
                db[collection_name].create_index("paper_id")
                db[collection_name].create_index("type")
            
            elif collection_name == "conversations":
                logger.info("Creating indexes for conversations collection")
                db[collection_name].create_index("id", unique=True)
                db[collection_name].create_index("user_id")
                db[collection_name].create_index("created_at")
                db[collection_name].create_index("updated_at")
        
        logger.info("Database setup complete")
    
    except Exception as e:
        logger.error(f"Error setting up database: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
