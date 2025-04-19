"""
Test script to verify the arXiv Research Assistant system.
"""
import os
import argparse
from pathlib import Path

from utils.config import MODEL_DIR
from utils.logger import setup_logger
from rag.llm import LlamaModel
from rag.retriever import HybridRetriever
from rag.reranker import Reranker
from rag.generator import ResponseGenerator

logger = setup_logger("test_system", "test_system.log")

def test_llm():
    """Test the LLaMA model."""
    logger.info("Testing LLaMA model...")
    
    # Check if model exists
    model_path = MODEL_DIR / "llama-3.1-8b-instruct.Q4_K_M.gguf"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please download the model first with: python download_model.py")
        return False
    
    try:
        # Initialize model
        llm = LlamaModel()
        
        # Test generation
        prompt = "Explain the concept of attention in transformer models:"
        logger.info(f"Testing generation with prompt: {prompt}")
        
        response = llm.generate(prompt, max_tokens=100)
        logger.info(f"Response: {response}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing LLaMA model: {e}")
        return False

def test_retriever():
    """Test the hybrid retriever."""
    logger.info("Testing hybrid retriever...")
    
    try:
        # Initialize retriever
        retriever = HybridRetriever()
        
        # Test retrieval
        query = "Explain the transformer architecture in deep learning"
        logger.info(f"Testing retrieval with query: {query}")
        
        results = retriever.retrieve(query, k=3)
        
        if results:
            logger.info(f"Retrieved {len(results)} documents")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}:")
                logger.info(f"  Title: {result['title']}")
                logger.info(f"  Authors: {', '.join(result['authors'])}")
                logger.info(f"  Content: {result['content'][:100]}...")
            
            return True
        else:
            logger.warning("No documents retrieved")
            logger.warning("This may be normal if you haven't run the data pipeline yet")
            return False
    
    except Exception as e:
        logger.error(f"Error testing retriever: {e}")
        return False

def test_generator():
    """Test the response generator."""
    logger.info("Testing response generator...")
    
    try:
        # Initialize generator
        generator = ResponseGenerator()
        
        # Test generation
        query = "Explain the transformer architecture in deep learning"
        logger.info(f"Testing generation with query: {query}")
        
        response = generator.generate_response(query)
        
        logger.info(f"Response: {response['response']}")
        logger.info(f"Context: {len(response['context'])} documents")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing generator: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test arXiv Research Assistant system")
    parser.add_argument("--llm", action="store_true", help="Test LLaMA model")
    parser.add_argument("--retriever", action="store_true", help="Test hybrid retriever")
    parser.add_argument("--generator", action="store_true", help="Test response generator")
    parser.add_argument("--all", action="store_true", help="Test all components")
    
    args = parser.parse_args()
    
    # Default to testing all if no specific test is specified
    if not (args.llm or args.retriever or args.generator):
        args.all = True
    
    # Test components
    if args.all or args.llm:
        test_llm()
    
    if args.all or args.retriever:
        test_retriever()
    
    if args.all or args.generator:
        test_generator()

if __name__ == "__main__":
    main()
