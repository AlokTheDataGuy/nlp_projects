import os
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import flask_cors
        import pandas
        import numpy
        import sentence_transformers
        import faiss
        import spacy

        # Check if scispacy is installed
        try:
            import scispacy
            logger.info("SciSpacy is installed")
        except ImportError:
            logger.warning("SciSpacy is not installed. Entity linking will be limited.")
            logger.warning("To install SciSpacy, follow the instructions in requirements.txt")

        # Check if the scientific model is installed
        try:
            spacy.load("en_core_sci_sm")
            logger.info("Scientific spaCy model is installed")
        except OSError:
            logger.error("Scientific spaCy model (en_core_sci_sm) is not installed")
            logger.info("Please install it with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz")
            return False

        logger.info("All core dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Please install all dependencies with: pip install -r requirements.txt")
        return False

def check_data():
    """Check if the dataset is available"""
    dataset_path = Path('data/processed/medquad_complete.csv')
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.info("Please run processing.py first to generate the dataset")
        return False

    logger.info(f"Dataset found at {dataset_path}")
    return True

def build_index_if_needed():
    """Build the FAISS index if it doesn't exist"""
    index_path = Path('data/faiss/question_index.faiss')
    if not index_path.exists():
        logger.info("FAISS index not found, building now...")
        try:
            from build_index import main as build_index_main
            build_index_main()
            logger.info("FAISS index built successfully")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return False
    else:
        logger.info("FAISS index found")

    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run the Medical Q&A Chatbot')
    parser.add_argument('--build-index', action='store_true', help='Force rebuild the FAISS index')
    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        return

    # Check data
    if not check_data():
        return

    # Build index if needed or requested
    if args.build_index:
        logger.info("Rebuilding FAISS index...")
        try:
            from build_index import main as build_index_main
            build_index_main()
            logger.info("FAISS index rebuilt successfully")
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {e}")
            return
    else:
        if not build_index_if_needed():
            return

    # Run the app
    logger.info("Starting the Flask application...")
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error starting the Flask application: {e}")

if __name__ == "__main__":
    main()
