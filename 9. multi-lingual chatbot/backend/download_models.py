"""
Script to download required models for the multilingual chatbot.
"""
import os
import sys
import logging
import urllib.request
import subprocess
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_PATH = "models/lid.176.bin"
OLLAMA_MODEL = "llama3.1:8b"
TRANSLATION_MODEL = "ai4bharat/IndicBART"


def download_fasttext_model():
    """Download the fastText language detection model."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(FASTTEXT_MODEL_PATH), exist_ok=True)

        # Check if model already exists
        if os.path.exists(FASTTEXT_MODEL_PATH):
            logger.info(f"FastText model already exists at {FASTTEXT_MODEL_PATH}")
            return

        # Download model
        logger.info(f"Downloading FastText model from {FASTTEXT_MODEL_URL}...")
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)
        logger.info(f"FastText model downloaded to {FASTTEXT_MODEL_PATH}")

    except Exception as e:
        logger.error(f"Error downloading FastText model: {e}")
        sys.exit(1)


def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_ollama_model():
    """Download the LLaMA3.1 model via Ollama."""
    try:
        # Check if Ollama is installed
        if not check_ollama_installed():
            logger.error("Ollama is not installed. Please install it from https://ollama.ai/")
            logger.info(f"After installing Ollama, run: ollama pull {OLLAMA_MODEL}")
            return

        # Download model
        logger.info(f"Downloading LLaMA3.1 model via Ollama...")
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        logger.info(f"LLaMA3.1 model downloaded successfully")

    except Exception as e:
        logger.error(f"Error downloading LLaMA3.1 model: {e}")


def download_indicbart_model():
    """Download the IndicBART translation model."""
    try:
        logger.info(f"Downloading IndicBART translation model ({TRANSLATION_MODEL})...")
        logger.info("This may take a few minutes as the model is downloaded...")

        # Create a temporary script to download the model
        # This ensures the model is cached properly by Hugging Face
        AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
        AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)

        logger.info(f"IndicBART model downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading IndicBART model: {e}")
        logger.error("Make sure you have an internet connection to download the model.")


def check_indicxlit_models():
    """Check if IndicXlit models are available."""
    try:
        # Import the module to trigger model download
        logger.info("Checking IndicXlit transliteration models...")
        from indicxlit_transliteration import load_engines
        load_engines()
        logger.info("IndicXlit transliteration models are available")
    except Exception as e:
        logger.error(f"Error checking IndicXlit models: {e}")
        logger.info("IndicXlit models will be downloaded when first used")


def main():
    """Main function to download all required models."""
    logger.info("Starting download of required models...")

    # Download fastText model
    download_fasttext_model()

    # Download LLaMA3.1 model via Ollama
    download_ollama_model()

    # Download IndicBART translation model
    download_indicbart_model()

    # Check IndicXlit transliteration models
    check_indicxlit_models()

    logger.info("All models downloaded successfully")


if __name__ == "__main__":
    main()
