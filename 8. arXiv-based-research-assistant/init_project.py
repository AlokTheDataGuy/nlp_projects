"""
Script to initialize the arXiv Research Assistant project.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path

from utils.config import DATA_DIR, MODEL_DIR, VECTOR_DB_DIR
from utils.logger import setup_logger

logger = setup_logger("init_project", "init_project.log")

def init_project(download_model=True, setup_frontend=True):
    """
    Initialize the project.
    
    Args:
        download_model: Whether to download the LLaMA model
        setup_frontend: Whether to set up the React frontend
    """
    # Create directories
    logger.info("Creating directories...")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    os.makedirs(DATA_DIR / "papers" / "pdfs", exist_ok=True)
    os.makedirs(DATA_DIR / "papers" / "metadata", exist_ok=True)
    os.makedirs(DATA_DIR / "processed" / "text", exist_ok=True)
    os.makedirs(DATA_DIR / "chunks", exist_ok=True)
    os.makedirs(VECTOR_DB_DIR / "embeddings", exist_ok=True)
    os.makedirs(BASE_DIR / "logs", exist_ok=True)
    os.makedirs(BASE_DIR / "conversations", exist_ok=True)
    
    # Download model
    if download_model:
        logger.info("Downloading LLaMA model...")
        subprocess.run([sys.executable, "download_model.py"])
    
    # Set up frontend
    if setup_frontend:
        logger.info("Setting up React frontend...")
        subprocess.run([sys.executable, "setup_frontend.py"])
    
    logger.info("Project initialization complete!")

if __name__ == "__main__":
    # Import here to avoid circular import
    from utils.config import BASE_DIR
    
    parser = argparse.ArgumentParser(description="Initialize arXiv Research Assistant project")
    parser.add_argument("--no-model", action="store_true", help="Skip downloading the LLaMA model")
    parser.add_argument("--no-frontend", action="store_true", help="Skip setting up the React frontend")
    
    args = parser.parse_args()
    
    init_project(
        download_model=not args.no_model,
        setup_frontend=not args.no_frontend
    )
