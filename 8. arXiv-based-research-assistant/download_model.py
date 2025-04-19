"""
Script to download the LLaMA model in GGUF format.
"""
import os
import argparse
import requests
import sys
from pathlib import Path
from tqdm import tqdm

from utils.config import MODEL_DIR
from utils.logger import setup_logger

logger = setup_logger("model_downloader", "model_downloader.log")

# Model information
# Using alternative models that don't require authentication
MODELS = {
    "mistral-7b-instruct-v0.2-q4_k_m": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size": 4_100_000_000,  # Approximate size in bytes
    },
    "mistral-7b-instruct-v0.2-q5_k_m": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "filename": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "size": 5_000_000_000,  # Approximate size in bytes
    },
    "phi-2-q4_k_m": {
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi-2.Q4_K_M.gguf",
        "size": 1_600_000_000,  # Approximate size in bytes
    },
    "tinyllama-1.1b-chat-v1.0-q4_k_m": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size": 700_000_000,  # Approximate size in bytes
    }
}

def check_model_exists(model_name="phi-2-q4_k_m"):
    """
    Check if the model exists in the expected location.

    Args:
        model_name: Name of the model to check

    Returns:
        bool: True if the model exists, False otherwise
    """
    if model_name not in MODELS:
        logger.error(f"Model {model_name} not found in available models")
        return False

    model_info = MODELS[model_name]
    model_path = MODEL_DIR / model_info["filename"]

    if model_path.exists():
        logger.info(f"Model found at {model_path}")
        return True

    logger.warning(f"Model not found at {model_path}")
    return False

def download_file(url: str, destination: Path, expected_size: int = None) -> bool:
    """
    Download a file with progress bar.

    Args:
        url: URL to download
        destination: Destination path
        expected_size: Expected file size in bytes

    Returns:
        True if download was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(destination.parent, exist_ok=True)

        # Check if file already exists
        if destination.exists():
            logger.info(f"File {destination} already exists, checking size...")

            # Check file size
            file_size = destination.stat().st_size
            if expected_size and file_size == expected_size:
                logger.info(f"File size matches expected size ({file_size} bytes), skipping download")
                return True
            else:
                logger.warning(f"File size ({file_size} bytes) doesn't match expected size ({expected_size} bytes), re-downloading")

        # Download file
        logger.info(f"Downloading {url} to {destination}")

        # Stream download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total size from headers or use expected size
        total_size = int(response.headers.get("content-length", 0)) or expected_size or 0

        # Create progress bar
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

        # Download file
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()

        # Check file size
        file_size = destination.stat().st_size
        if expected_size and file_size != expected_size:
            logger.warning(f"Downloaded file size ({file_size} bytes) doesn't match expected size ({expected_size} bytes)")
            return False

        logger.info(f"Download complete: {destination}")
        return True

    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def download_model(model_name: str) -> bool:
    """
    Download a model.

    Args:
        model_name: Name of the model to download

    Returns:
        True if download was successful, False otherwise
    """
    if model_name not in MODELS:
        logger.error(f"Model {model_name} not found in available models")
        return False

    model_info = MODELS[model_name]
    url = model_info["url"]
    filename = model_info["filename"]
    size = model_info.get("size")

    # Download model
    destination = MODEL_DIR / filename
    success = download_file(url, destination, size)

    return success

def print_available_models():
    """Print available models."""
    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)
    print("\nThe following quantized models are available:")

    for model_name, model_info in MODELS.items():
        size_gb = model_info.get("size", 0) / 1_000_000_000
        print(f"\n- {model_name}:")
        print(f"  Filename: {model_info['filename']}")
        print(f"  Size: {size_gb:.2f} GB")

    print("\nUse the --model parameter to specify which model to download:")
    print("  python download_model.py --model phi-2-q4_k_m")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LLaMA model in GGUF format")
    parser.add_argument("--model", type=str, default="phi-2-q4_k_m",
                        help="Model to download (use --list to see available models)")
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.list:
        print_available_models()
        sys.exit(0)

    # Check if model already exists
    if check_model_exists(args.model):
        logger.info(f"Model {args.model} already exists, skipping download")
        sys.exit(0)

    # Download model
    success = download_model(args.model)

    if success:
        logger.info(f"Model {args.model} downloaded successfully")
    else:
        logger.error(f"Failed to download model {args.model}")
        print_available_models()
