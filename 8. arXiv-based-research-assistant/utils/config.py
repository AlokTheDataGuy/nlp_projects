"""
Configuration management for the arXiv Research Assistant.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_storage"
MODEL_DIR = BASE_DIR / "models"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# arXiv configuration
ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE"]  # Computer Science categories
MAX_PAPERS = int(os.getenv("MAX_PAPERS", 1000))  # Maximum number of papers to download
PAPERS_PER_CATEGORY = MAX_PAPERS // len(ARXIV_CATEGORIES)

# Text processing configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))  # Token size for text chunks
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))  # Overlap between chunks

# Embedding configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", str(MODEL_DIR / "bge-small-en-v1.5"))  # Local path to cloned model
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))

# LLM configuration
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", str(MODEL_DIR / "phi-2.Q4_K_M.gguf"))
LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", 2048))  # Phi-2 has a 2048 token context window
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.9))

# RAG configuration
NUM_DOCUMENTS = int(os.getenv("NUM_DOCUMENTS", 5))  # Number of documents to retrieve
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 10))  # Number of documents to rerank
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))  # Weight for hybrid search (0 = BM25 only, 1 = Vector only)

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 1))

# Frontend configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
