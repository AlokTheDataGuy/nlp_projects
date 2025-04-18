import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the Medical Q&A Chatbot"""
    try:
        # Check if the dataset exists
        dataset_path = Path('data/processed/medquad_complete.csv')
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            logger.info("Please make sure the dataset is available")
            return 1

        # Check if the FAISS index exists
        index_path = Path('data/faiss/question_index.faiss')
        if not index_path.exists():
            logger.warning(f"FAISS index not found at {index_path}")
            logger.info("Building the index first...")

            # Run the build_index_simple.py script
            import build_index_simple
            build_index_simple.main()

        # Import and run the Flask app
        logger.info("Starting the Flask application...")
        from app import app, initialize_models

        # Initialize the models
        logger.info("Initializing models...")
        if initialize_models():
            # Run the app
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            logger.error("Failed to initialize models. Exiting.")
            return 1

        return 0
    except Exception as e:
        logger.error(f"Error running the application: {e}")
        return 1

if __name__ == "__main__":
    main()
