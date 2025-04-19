"""
Phi-2 integration via llama.cpp.
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json

from llama_cpp import Llama

from utils.config import (
    MODEL_DIR, LLM_MODEL_PATH, LLM_CONTEXT_SIZE,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P
)
from utils.logger import setup_logger

logger = setup_logger("llm_integration", "llm.log")

class PhiModel:
    """Phi-2 model integration via llama.cpp."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 context_size: int = LLM_CONTEXT_SIZE,
                 max_tokens: int = LLM_MAX_TOKENS,
                 temperature: float = LLM_TEMPERATURE,
                 top_p: float = LLM_TOP_P,
                 n_gpu_layers: int = -1,  # -1 means use all available layers
                 n_threads: Optional[int] = None):  # None means use all available threads
        """
        Initialize the Phi-2 model.

        Args:
            model_path: Path to the model file
            context_size: Context size in tokens
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_threads: Number of threads to use (None for all)
        """
        self.model_path = model_path or LLM_MODEL_PATH
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads

        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load model
        logger.info(f"Loading Phi-2 model from {self.model_path}")
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            verbose=False
        )
        logger.info("Phi-2 model loaded successfully")

    def generate(self,
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                stop: Optional[List[str]] = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            stop: List of strings to stop generation

        Returns:
            Generated text
        """
        # Use instance defaults if not provided
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p

        try:
            # Generate text
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False  # Don't include prompt in output
            )

            # Extract generated text
            generated_text = output["choices"][0]["text"]
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"

    def generate_chat(self,
                     messages: List[Dict[str, str]],
                     max_tokens: Optional[int] = None,
                     temperature: Optional[float] = None,
                     top_p: Optional[float] = None,
                     stop: Optional[List[str]] = None) -> str:
        """
        Generate text from chat messages.

        Args:
            messages: List of chat messages in the format [{"role": "user", "content": "..."}]
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            stop: List of strings to stop generation

        Returns:
            Generated text
        """
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Generate text
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to prompt.

        Args:
            messages: List of chat messages in the format [{"role": "user", "content": "..."}]

        Returns:
            Formatted prompt
        """
        # Format for Phi-2
        prompt = ""

        # Add system message if present
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        if system_message:
            prompt += f"<|system|>\n{system_message}\n\n"

        # Add conversation history
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                # Already handled above
                continue
            elif role == "user":
                prompt += f"<|user|>\n{content}\n\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n\n"
            else:
                logger.warning(f"Unknown role: {role}")

        # Add final assistant prompt
        prompt += "<|assistant|>\n"

        return prompt


if __name__ == "__main__":
    # Example usage
    llm = PhiModel()

    # Simple generation
    prompt = "Explain the concept of attention in transformer models:"
    response = llm.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    # Chat generation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specializing in computer science research."},
        {"role": "user", "content": "What is the difference between CNN and RNN?"}
    ]
    response = llm.generate_chat(messages)
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print(f"Response: {response}")
