"""
Test script to verify the Phi-2 model integration.
"""
import os
import argparse
from pathlib import Path

from utils.config import MODEL_DIR, LLM_MODEL_PATH
from utils.logger import setup_logger
from rag.llm import PhiModel

logger = setup_logger("test_model", "test_model.log")

def test_model(model_path=None, prompt=None):
    """
    Test the Phi-2 model.

    Args:
        model_path: Path to the model directory
        prompt: Prompt to test with
    """
    model_path = model_path or LLM_MODEL_PATH
    prompt = prompt or "Explain the concept of attention in transformer models in 3 sentences:"

    logger.info(f"Testing model at {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.error("Please download the model first with: python download_model.py")
        return

    try:
        # Initialize model
        logger.info("Initializing model...")
        llm = PhiModel(model_path=model_path)

        # Test generation
        logger.info(f"Testing generation with prompt: {prompt}")

        response = llm.generate(prompt, max_tokens=100)

        logger.info("Generation successful!")
        print("\nPrompt:")
        print(prompt)
        print("\nResponse:")
        print(response)

        # Test chat
        logger.info("Testing chat generation...")

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specializing in computer science research."},
            {"role": "user", "content": "What is the difference between CNN and RNN?"}
        ]

        chat_response = llm.generate_chat(messages, max_tokens=100)

        logger.info("Chat generation successful!")
        print("\nChat Messages:")
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")
        print("\nChat Response:")
        print(chat_response)

    except Exception as e:
        logger.error(f"Error testing model: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Phi-2 model")
    parser.add_argument("--model-path", type=str, help="Path to the model directory")
    parser.add_argument("--prompt", type=str, help="Prompt to test with")

    args = parser.parse_args()

    test_model(args.model_path, args.prompt)
