"""
Main script to run the arXiv Research Assistant.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path

from utils.config import DATA_DIR, MODEL_DIR, VECTOR_DB_DIR
from utils.logger import setup_logger

logger = setup_logger("main", "main.log")

def check_dependencies():
    """Check if all dependencies are installed."""
    try:
        import torch
        import transformers
        import langchain
        import faiss
        import sentence_transformers
        import arxiv
        import PyPDF2
        import fitz
        import nltk
        import spacy
        import rank_bm25
        import llama_cpp
        import fastapi
        import uvicorn
        import pydantic
        import dotenv
        import tqdm
        import matplotlib
        import plotly
        import numpy
        import pandas
        
        logger.info("All dependencies are installed")
        return True
    
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install all dependencies with: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the LLaMA model is downloaded."""
    model_path = MODEL_DIR / "llama-3.1-8b-instruct.Q4_K_M.gguf"
    
    if not model_path.exists():
        logger.warning(f"LLaMA model not found at {model_path}")
        logger.info("Downloading LLaMA model...")
        
        # Run download script
        subprocess.run([sys.executable, "download_model.py"])
        
        # Check again
        if not model_path.exists():
            logger.error("Failed to download LLaMA model")
            return False
    
    logger.info(f"LLaMA model found at {model_path}")
    return True

def check_data():
    """Check if data is downloaded and processed."""
    # Check if vector database exists
    index_path = VECTOR_DB_DIR / "faiss_index.bin"
    metadata_path = VECTOR_DB_DIR / "faiss_metadata.json"
    
    if not index_path.exists() or not metadata_path.exists():
        logger.warning("Vector database not found")
        logger.info("You need to run the data pipeline to download and process papers")
        logger.info("Run: python -m data.pipeline")
        return False
    
    logger.info("Vector database found")
    return True

def run_api():
    """Run the API server."""
    logger.info("Starting API server...")
    
    # Run API server
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])

def run_frontend():
    """Run the frontend development server."""
    logger.info("Starting frontend development server...")
    
    # Check if frontend directory exists
    if not os.path.exists("frontend"):
        logger.error("Frontend directory not found")
        logger.error("Please set up the frontend first")
        return
    
    # Run frontend server
    os.chdir("frontend")
    subprocess.run(["npm", "start"])

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="arXiv Research Assistant")
    parser.add_argument("--api", action="store_true", help="Run API server")
    parser.add_argument("--frontend", action="store_true", help="Run frontend development server")
    parser.add_argument("--check", action="store_true", help="Check dependencies and data")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model
    if not check_model():
        return
    
    # Check data
    data_ready = check_data()
    
    if args.check:
        # Just checking, exit
        return
    
    if not data_ready:
        logger.warning("Data not ready, some features may not work")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != "y":
            return
    
    if args.api:
        # Run API server
        run_api()
    
    elif args.frontend:
        # Run frontend server
        run_frontend()
    
    else:
        # Run both
        logger.info("Running both API and frontend")
        
        # Run API server in a separate process
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
        
        try:
            # Run frontend server
            run_frontend()
        
        finally:
            # Terminate API server
            api_process.terminate()
            api_process.wait()

if __name__ == "__main__":
    main()
