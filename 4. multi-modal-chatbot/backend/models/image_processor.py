import base64
from typing import Dict, Any, List, Optional
import os
from PIL import Image
import ollama
import sys

async def process_image(image_data: Dict[str, Any], prompt: str, history: Optional[List] = None) -> Dict[str, Any]:
    """
    Process an image using llama3.2-vision:11b model via Ollama.

    Args:
        image_data: Dictionary containing path and base64 data of the image
        prompt: Text prompt to guide image analysis
        history: Optional conversation history

    Returns:
        Dictionary with the model's response
    """
    try:
        print(f"Processing image with prompt: {prompt}")

        # Get the image path from the image_data
        img_path = image_data["path"]

        # Verify the image exists
        if not os.path.exists(img_path):
            raise Exception(f"Image file not found: {img_path}")

        # Read the image file directly
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        print(f"Image loaded and converted to base64 (length: {len(image_base64)})")

        # Create the messages array with base64 image data
        messages = [{
            'role': 'user',
            'content': prompt,
            'images': [image_base64]
        }]

        # Add history if provided
        if history and len(history) > 0:
            # Insert history before the current message
            messages = history + messages

        print("Sending request to Ollama...")

        # Call the Ollama API
        response = ollama.chat(
            model='llama3.2-vision',
            messages=messages
        )

        print("Received response from Ollama")

        # Return the response
        return {
            "text": response["message"]["content"],
            "model": "llama3.2-vision"
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        return {
            "text": f"Error analyzing image: {str(e)}\n\nI'm unable to analyze the image at the moment. Please try again later or provide a description of the image so I can assist you better.",
            "model": "llama3.2-vision",
            "error": True
        }
