import logging
import os
from app.db.init_db import init_db
from app.core.vector_store import VectorIndex
from app.core.processing_queue import processing_queue

logger = logging.getLogger(__name__)

def init_app():
    """
    Initialize the application.
    """
    try:
        # Initialize database
        init_db()
        
        # Initialize vector index
        vector_index = VectorIndex()
        
        # Check if vector index exists
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        os.makedirs(data_dir, exist_ok=True)
        vector_index_path = os.path.join(data_dir, "vector_index")
        
        if os.path.exists(f"{vector_index_path}.index"):
            # Load existing vector index
            vector_index.load(vector_index_path)
            logger.info("Vector index loaded successfully")
        else:
            # Create new vector index
            logger.info("Creating new vector index")
        
        # Start processing queue
        processing_queue.start()
        
        logger.info("Application initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        return False
