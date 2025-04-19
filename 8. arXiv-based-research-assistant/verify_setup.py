"""
Script to verify the setup of the arXiv Research Assistant.
"""
import os
import sys
import subprocess
from pathlib import Path

from utils.config import MODEL_DIR, LLM_MODEL_PATH
from utils.logger import setup_logger

logger = setup_logger("verify_setup", "verify_setup.log")

def check_python_version():
    """Check Python version."""
    logger.info(f"Python version: {sys.version}")
    if sys.version_info < (3, 9):
        logger.error("Python 3.9 or higher is required")
        return False
    return True

def check_dependencies():
    """Check if all dependencies are installed."""
    try:
        # Check core dependencies
        import torch
        import transformers
        import langchain
        import faiss
        import sentence_transformers
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        return True
    
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install all dependencies with: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the LLaMA model is downloaded."""
    model_path = Path(LLM_MODEL_PATH)
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please download the model with: python download_model.py")
        return False
    
    # Check if the model directory contains the expected files
    if not (model_path / "config.json").exists():
        logger.error(f"Model configuration file not found at {model_path / 'config.json'}")
        logger.error("Please download the model with: python download_model.py")
        return False
    
    if not ((model_path / "model.safetensors").exists() or any((model_path / "pytorch_model").glob("*.bin"))):
        logger.error(f"Model weights not found at {model_path}")
        logger.error("Please download the model with: python download_model.py")
        return False
    
    logger.info(f"Model found at {model_path}")
    return True

def test_model():
    """Test the LLaMA model."""
    try:
        logger.info("Testing model...")
        result = subprocess.run(
            [sys.executable, "test_model.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Model test successful")
            print(result.stdout)
            return True
        else:
            logger.error(f"Model test failed: {result.stderr}")
            print(result.stderr)
            return False
    
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

def main():
    """Main function."""
    print("Verifying arXiv Research Assistant setup...")
    
    # Check Python version
    if not check_python_version():
        print("❌ Python version check failed")
        return
    print("✅ Python version check passed")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependencies check failed")
        return
    print("✅ Dependencies check passed")
    
    # Check model
    if not check_model():
        print("❌ Model check failed")
        return
    print("✅ Model check passed")
    
    # Test model
    if not test_model():
        print("❌ Model test failed")
        return
    print("✅ Model test passed")
    
    print("\n✅ All checks passed! The arXiv Research Assistant is ready to use.")
    print("\nYou can now run the system with:")
    print("  - CLI: python cli.py")
    print("  - API: python run_api.py")
    print("  - Frontend: cd frontend && npm run dev")

if __name__ == "__main__":
    main()
