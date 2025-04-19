"""
Tests for the configuration module.
"""
import os
import pytest
from pathlib import Path

from utils.config import (
    BASE_DIR, DATA_DIR, MODEL_DIR, VECTOR_DB_DIR,
    ARXIV_CATEGORIES, MAX_PAPERS, PAPERS_PER_CATEGORY,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE,
    LLM_MODEL_PATH, LLM_CONTEXT_SIZE, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P,
    NUM_DOCUMENTS, RERANK_TOP_N, HYBRID_ALPHA,
    API_HOST, API_PORT, API_WORKERS,
    FRONTEND_URL, LOG_LEVEL
)

def test_base_directories():
    """Test base directories."""
    assert isinstance(BASE_DIR, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(MODEL_DIR, Path)
    assert isinstance(VECTOR_DB_DIR, Path)

def test_arxiv_config():
    """Test arXiv configuration."""
    assert isinstance(ARXIV_CATEGORIES, list)
    assert all(isinstance(category, str) for category in ARXIV_CATEGORIES)
    assert isinstance(MAX_PAPERS, int)
    assert isinstance(PAPERS_PER_CATEGORY, int)

def test_text_processing_config():
    """Test text processing configuration."""
    assert isinstance(CHUNK_SIZE, int)
    assert isinstance(CHUNK_OVERLAP, int)
    assert CHUNK_OVERLAP < CHUNK_SIZE

def test_embedding_config():
    """Test embedding configuration."""
    assert isinstance(EMBEDDING_MODEL, str)
    assert isinstance(EMBEDDING_DEVICE, str)
    assert EMBEDDING_DEVICE in ["cuda", "cpu"]
    assert isinstance(EMBEDDING_BATCH_SIZE, int)

def test_llm_config():
    """Test LLM configuration."""
    assert isinstance(LLM_MODEL_PATH, str)
    assert isinstance(LLM_CONTEXT_SIZE, int)
    assert isinstance(LLM_MAX_TOKENS, int)
    assert isinstance(LLM_TEMPERATURE, float)
    assert 0.0 <= LLM_TEMPERATURE <= 1.0
    assert isinstance(LLM_TOP_P, float)
    assert 0.0 <= LLM_TOP_P <= 1.0

def test_rag_config():
    """Test RAG configuration."""
    assert isinstance(NUM_DOCUMENTS, int)
    assert isinstance(RERANK_TOP_N, int)
    assert isinstance(HYBRID_ALPHA, float)
    assert 0.0 <= HYBRID_ALPHA <= 1.0

def test_api_config():
    """Test API configuration."""
    assert isinstance(API_HOST, str)
    assert isinstance(API_PORT, int)
    assert isinstance(API_WORKERS, int)

def test_frontend_config():
    """Test frontend configuration."""
    assert isinstance(FRONTEND_URL, str)

def test_logging_config():
    """Test logging configuration."""
    assert isinstance(LOG_LEVEL, str)
    assert LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
