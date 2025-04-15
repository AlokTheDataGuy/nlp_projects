import ollama
from typing import Dict, Any, List, Optional

async def process_text(message: str, history: Optional[List] = None) -> Dict[str, Any]:
    """
    Process text using qwen2.5:7b model via Ollama.

    Args:
        message: Text message to process
        history: Optional conversation history

    Returns:
        Dictionary with the model's response
    """
    try:
        # Prepare messages
        messages = []

        # Add history if provided
        if history:
            messages.extend(history)

        # Add the user message
        messages.append({
            "role": "user",
            "content": message
        })

        # Call the Ollama API
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=messages,
            stream=False
        )

        # Extract the response text
        response_text = response["message"]["content"]

        return {
            "text": response_text,
            "model": "qwen2.5:7b"
        }

    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "text": f"Error processing text: {str(e)}",
            "model": "qwen2.5:7b",
            "error": True
        }
