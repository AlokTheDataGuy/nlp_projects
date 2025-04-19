"""
Script to download the embedding model.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path

from utils.config import MODEL_DIR
from utils.logger import setup_logger

logger = setup_logger("embedding_model_downloader", "embedding_model_downloader.log")

def check_model_exists(model_name="bge-small-en-v1.5"):
    """
    Check if the model exists in the expected location.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        bool: True if the model exists, False otherwise
    """
    model_path = MODEL_DIR / model_name
    
    if model_path.exists():
        # Check if the model directory contains the expected files
        if (model_path / "config.json").exists() and \
           ((model_path / "model.safetensors").exists() or \
            any((model_path / "pytorch_model").glob("*.bin"))):
            logger.info(f"Model found at {model_path}")
            return True
    
    logger.warning(f"Model not found at {model_path}")
    return False

def clone_model_from_huggingface(model_name="bge-small-en-v1.5", repo="BAAI/bge-small-en-v1.5"):
    """
    Clone the model from Hugging Face using git.
    
    Args:
        model_name: Name of the model directory
        repo: Hugging Face repository name
    
    Returns:
        bool: True if cloning was successful, False otherwise
    """
    try:
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Check if git is installed
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Git is not installed or not in PATH. Please install Git and try again.")
            return False
        
        # Clone the repository
        logger.info(f"Cloning {repo} from Hugging Face...")
        
        # Change to the model directory
        os.chdir(MODEL_DIR)
        
        # Clone the repository
        result = subprocess.run(
            ["git", "clone", f"https://huggingface.co/{repo}", model_name],
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(result.stdout)
        
        # Check if the model was cloned successfully
        if check_model_exists(model_name):
            logger.info("Model cloned successfully")
            return True
        else:
            logger.error("Model cloning failed")
            return False
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error cloning model: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    
    except Exception as e:
        logger.error(f"Error cloning model: {e}")
        return False

def print_manual_instructions(model_name="bge-small-en-v1.5", repo="BAAI/bge-small-en-v1.5"):
    """Print instructions for manually downloading the model."""
    print("\n" + "=" * 80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    print(f"\nTo manually download the {model_name} model:")
    print(f"\n1. Go to https://huggingface.co/{repo}")
    print("2. Sign in to your Hugging Face account (or create one if needed)")
    print("3. Accept the model license agreement")
    print("4. Clone the repository using Git:\n")
    print(f"   git clone https://huggingface.co/{repo} {MODEL_DIR / model_name}\n")
    print("   OR download the files manually and place them in:")
    print(f"   {MODEL_DIR / model_name}\n")
    print("5. Ensure the following files are present in the model directory:")
    print("   - config.json")
    print("   - tokenizer.json")
    print("   - tokenizer_config.json")
    print("   - model.safetensors (or pytorch_model.bin files)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download embedding model")
    parser.add_argument("--model", type=str, default="bge-small-en-v1.5", help="Model name")
    parser.add_argument("--repo", type=str, default="BAAI/bge-small-en-v1.5", help="Hugging Face repository")
    parser.add_argument("--manual", action="store_true", help="Show manual download instructions")
    
    args = parser.parse_args()
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if args.manual:
        print_manual_instructions(args.model, args.repo)
        sys.exit(0)
    
    # Check if model already exists
    if check_model_exists(args.model):
        logger.info("Model already exists, skipping download")
        sys.exit(0)
    
    # Try to clone the model
    success = clone_model_from_huggingface(args.model, args.repo)
    
    if not success:
        logger.warning("Automatic download failed. Please download the model manually.")
        print_manual_instructions(args.model, args.repo)
