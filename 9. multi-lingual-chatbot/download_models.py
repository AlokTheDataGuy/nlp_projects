"""
Script to download required models.
"""

import os
import logging
import argparse
import fasttext
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import LANGUAGE_DETECTION, TRANSLATION

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def download_fasttext_model():
    """Download fastText language identification model."""
    model_path = LANGUAGE_DETECTION["model_path"]
    model_dir = os.path.dirname(model_path)

    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path):
        logger.info(f"fastText model already exists at {model_path}")
        return

    logger.info("Downloading fastText language identification model...")
    try:
        # Download the compressed model (lid.176.ftz)
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        urllib.request.urlretrieve(url, model_path)

        # Test the model
        model = fasttext.load_model(model_path)
        test_result = model.predict("Hello, how are you?")
        logger.info(f"fastText model test: {test_result}")

        logger.info(f"fastText model downloaded to {model_path}")
    except Exception as e:
        logger.error(f"Failed to download fastText model: {e}")

def download_nllb_model(quantize=True):
    """
    Download NLLB translation model.

    Args:
        quantize: Whether to quantize the model to 8-bit
    """
    model_name = TRANSLATION["model_name"]
    local_model_path = TRANSLATION.get("local_model_path")

    # Check if we have local tokenizer files
    has_local_tokenizer = False
    if local_model_path and os.path.exists(local_model_path):
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model"]
        has_local_tokenizer = all(os.path.exists(os.path.join(local_model_path, f)) for f in tokenizer_files)

    # Check if we have local model weights
    has_local_weights = False
    if local_model_path and os.path.exists(local_model_path):
        weight_files = ["pytorch_model.bin"]
        shard_files = [f for f in os.listdir(local_model_path) if f.startswith("pytorch_model-") and f.endswith(".bin")]
        has_local_weights = any(os.path.exists(os.path.join(local_model_path, f)) for f in weight_files) or len(shard_files) > 0

    logger.info(f"Local tokenizer available: {has_local_tokenizer}")
    logger.info(f"Local model weights available: {has_local_weights}")

    # Download what's needed
    logger.info(f"Processing NLLB model {model_name}...")
    try:
        # Load/download tokenizer
        if has_local_tokenizer:
            logger.info(f"Loading tokenizer from local path: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        else:
            logger.info(f"Downloading tokenizer from Hugging Face: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Save tokenizer to local path if specified
            if local_model_path:
                os.makedirs(local_model_path, exist_ok=True)
                tokenizer.save_pretrained(local_model_path)
                logger.info(f"Saved tokenizer to {local_model_path}")

        # Load/download model
        if has_local_weights:
            logger.info(f"Local model weights found at {local_model_path}")
            if quantize and torch.cuda.is_available():
                logger.info("Loading model with 8-bit quantization...")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    local_model_path,
                    device_map="auto",
                    load_in_8bit=True
                )
            else:
                logger.info("Loading model without quantization...")
                model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
        else:
            logger.info(f"Downloading model weights from Hugging Face: {model_name}")
            if quantize and torch.cuda.is_available():
                logger.info("Downloading model with 8-bit quantization...")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    load_in_8bit=True
                )
            else:
                logger.info("Downloading model without quantization...")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Save model to local path if specified
            if local_model_path:
                os.makedirs(local_model_path, exist_ok=True)
                model.save_pretrained(local_model_path)
                logger.info(f"Saved model to {local_model_path}")

        logger.info(f"NLLB model processing complete")

        # Test the model
        logger.info("Testing model with a sample translation...")
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")
        if torch.cuda.is_available():
            test_input = test_input.to("cuda")

        with torch.no_grad():
            output_tokens = model.generate(
                **test_input,
                forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"],
                max_length=128
            )

        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        logger.info(f"NLLB model test: {output_text}")

    except Exception as e:
        logger.error(f"Failed to process NLLB model: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download required models")
    parser.add_argument("--fasttext", action="store_true", help="Download fastText model")
    parser.add_argument("--nllb", action="store_true", help="Download NLLB model")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--no-quantize", action="store_true", help="Don't quantize NLLB model")

    args = parser.parse_args()

    # If no specific model is specified, download all
    if not (args.fasttext or args.nllb) or args.all:
        args.fasttext = True
        args.nllb = True

    # Download models
    if args.fasttext:
        download_fasttext_model()

    if args.nllb:
        download_nllb_model(not args.no_quantize)

    logger.info("Model download complete")

if __name__ == "__main__":
    main()
