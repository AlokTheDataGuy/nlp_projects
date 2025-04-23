"""
Model Loader Module

This module handles loading and managing the LLM model.
"""

import os
import logging
import yaml
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Class for loading and managing the LLM model.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the ModelLoader.

        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.base_model_id = self.config["model"]["base_model_id"]
        self.fine_tuned_model_path = self.config["model"]["fine_tuned_model_path"]
        self.quantization = self.config["model"]["quantization"]
        self.generation_config = self.config["model"]["generation"]

        # Check if fine-tuned model exists
        self.use_fine_tuned_model = os.path.exists(self.fine_tuned_model_path)

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Dict containing configuration.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_model(self) -> None:
        """
        Load the LLM model and tokenizer.
        """
        try:
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Configure quantization
            quantization_config = None
            if self.quantization["method"] == "bitsandbytes" and self.quantization["bits"] in [4, 8] and torch.cuda.is_available():
                logger.info(f"Using {self.quantization['bits']}-bit quantization with bitsandbytes")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.quantization["bits"] == 4,
                    load_in_8bit=self.quantization["bits"] == 8,
                    bnb_4bit_compute_dtype=torch.float16 if self.quantization["bits"] == 4 else None,
                    bnb_4bit_quant_type="nf4" if self.quantization["bits"] == 4 else None,
                )

            # Determine which model to load
            if self.use_fine_tuned_model:
                # Load fine-tuned model
                model_path = self.fine_tuned_model_path
                logger.info(f"Loading fine-tuned model from: {model_path}")

                try:
                    # Load tokenizer and model
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )

                    # Create text generation pipeline
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device_map="auto" if torch.cuda.is_available() else None
                    )

                    logger.info("Successfully loaded fine-tuned model")
                except Exception as e:
                    logger.error(f"Error loading fine-tuned model: {str(e)}")
                    raise
            else:
                # Fine-tuned model not found, try to load base model
                logger.info(f"Fine-tuned model not found. Trying to load base model: {self.base_model_id}")

                try:
                    # Try to load the base model
                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_id,
                        quantization_config=quantization_config,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )

                    # Create text generation pipeline
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device_map="auto" if torch.cuda.is_available() else None
                    )

                    logger.info("Successfully loaded base model")
                except Exception as e:
                    logger.warning(f"Could not load base model: {str(e)}")

                    # If base model fails, try to load a smaller model for testing
                    logger.info("Trying to load GPT-2 model for testing")

                    try:
                        # Try to load a small model for testing
                        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                        self.model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto" if torch.cuda.is_available() else None)

                        # Create text generation pipeline
                        self.pipeline = pipeline(
                            "text-generation",
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device_map="auto" if torch.cuda.is_available() else None
                        )

                        logger.info("Loaded GPT-2 model for testing")
                    except Exception as e:
                        logger.warning(f"Could not load GPT-2 model: {str(e)}")

                        # If all models fail, create a dummy pipeline
                        logger.info("Creating a mock pipeline that returns fixed responses")

                        # Create a dummy tokenizer
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=False)
                        except Exception:
                            self.tokenizer = None

                        self.model = None

                        # Create a dummy pipeline that returns fixed responses
                        class DummyPipeline:
                            def __call__(self, prompt, **kwargs):
                                # Ignore unused parameters
                                _ = prompt, kwargs
                                return [{"generated_text": "I'm a research assistant for arXiv papers. The LLM model is not loaded yet. Please run the fine-tuning script to create a model or specify a different model to use."}]

                        self.pipeline = DummyPipeline()

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_text(self, prompt: str, max_new_tokens: Optional[int] = None, **kwargs) -> str:
        """
        Generate text using the loaded model.

        Args:
            prompt: The prompt to generate text from.
            max_new_tokens: Maximum number of tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text.
        """
        if self.tokenizer is None:
            logger.error("Tokenizer not loaded. Call load_model() first.")
            return ""

        # If we're using a dummy model, return a fixed response
        if self.model is None:
            logger.warning("Using dummy model - returning fixed response")
            return "I'm a research assistant for arXiv papers. The LLM model is not loaded yet. Please quantize and load the Meta-Llama-3.1-8B model to enable full functionality."

        try:
            # Set default generation parameters
            generation_params = {
                "max_new_tokens": max_new_tokens or self.generation_config["max_new_tokens"],
                "temperature": kwargs.get("temperature", self.generation_config["temperature"]),
                "top_p": kwargs.get("top_p", self.generation_config["top_p"]),
                "top_k": kwargs.get("top_k", self.generation_config.get("top_k", 40)),
                "repetition_penalty": kwargs.get("repetition_penalty", self.generation_config.get("repetition_penalty", 1.1)),
                "do_sample": kwargs.get("do_sample", self.generation_config.get("do_sample", True)),
            }

            # Generate text
            outputs = self.pipeline(
                prompt,
                **generation_params,
                return_full_text=False
            )

            # Extract generated text
            generated_text = outputs[0]["generated_text"]

            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return ""

    def unload_model(self) -> None:
        """
        Unload the model to free up memory.
        """
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")


if __name__ == "__main__":
    # Example usage
    loader = ModelLoader()
    loader.load_model()

    # Generate text
    prompt = "Explain the concept of transformers in deep learning:"
    generated_text = loader.generate_text(prompt, max_new_tokens=200)

    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

    # Unload model
    loader.unload_model()
