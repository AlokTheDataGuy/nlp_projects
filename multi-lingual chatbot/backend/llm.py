"""
LLaMA3 integration via Ollama.
"""
import logging
import json
import time
import subprocess
import platform
from typing import List, Dict, Any, Optional
import httpx

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, SYSTEM_PROMPTS, CULTURAL_CONTEXT

# Initialize logger
logger = logging.getLogger(__name__)


async def generate_response(
    messages: List[Dict[str, str]],
    language_code: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 2,
    retry_delay: float = 2.0
) -> Optional[str]:
    """
    Generate a response using LLaMA3 via Ollama.

    Args:
        messages: List of conversation messages
        language_code: Language code for the response
        temperature: Temperature for response generation
        max_tokens: Maximum number of tokens to generate
        max_retries: Maximum number of retries on failure
        retry_delay: Delay between retries in seconds

    Returns:
        Generated response or None if generation fails
    """
    # Prepare system message with appropriate language and cultural context
    system_message = {
        "role": "system",
        "content": f"{SYSTEM_PROMPTS.get(language_code, SYSTEM_PROMPTS['eng_Latn'])} {CULTURAL_CONTEXT.get(language_code, '')}"
    }

    # Format messages for Ollama
    formatted_messages = [system_message] + messages

    # Prepare request payload
    payload = {
        "model": OLLAMA_MODEL,
        "messages": formatted_messages,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False
    }

    # Try with retries
    for attempt in range(max_retries + 1):
        try:
            # Log the request for debugging
            logger.info(f"Sending request to Ollama API with model: {OLLAMA_MODEL} (attempt {attempt + 1}/{max_retries + 1})")

            # Make request to Ollama API with increased timeout
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload
                )

                # Check if request was successful
                response.raise_for_status()

                # Log response status
                logger.info(f"Ollama API response status: {response.status_code}")

                # Parse response
                result = response.json()

                # Extract generated text
                if "message" in result and "content" in result["message"]:
                    return result["message"]["content"]
                else:
                    logger.error(f"Unexpected response format from Ollama: {result}")
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    return None

        except httpx.ReadTimeout:
            logger.error(f"Timeout while waiting for Ollama response (attempt {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                # Increase retry delay for subsequent attempts
                adjusted_delay = retry_delay * (attempt + 1)
                logger.info(f"Retrying in {adjusted_delay} seconds...")
                time.sleep(adjusted_delay)
                continue
            logger.error("All attempts timed out. The model may be overloaded or the request too complex.")
            return None

        except Exception as e:
            logger.error(f"Error generating response with Ollama (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return None

    # If we get here, all attempts failed
    logger.error(f"All {max_retries + 1} attempts to generate response failed")
    return None


async def check_model_availability(max_retries: int = 2, retry_delay: float = 2.0) -> bool:
    """
    Check if the LLaMA3.1 model is available in Ollama.

    Args:
        max_retries: Maximum number of retries on failure
        retry_delay: Delay between retries in seconds

    Returns:
        True if the model is available, False otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Checking if model {OLLAMA_MODEL} is available in Ollama... (attempt {attempt + 1}/{max_retries + 1})")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")

                # Check if request was successful
                response.raise_for_status()

                # Parse response
                result = response.json()

                # Check if model is in the list of available models
                if "models" in result:
                    available = any(model["name"] == OLLAMA_MODEL for model in result["models"])
                    if available:
                        logger.info(f"Model {OLLAMA_MODEL} is available in Ollama.")
                    else:
                        logger.warning(f"Model {OLLAMA_MODEL} is not available in Ollama. Available models: {[model['name'] for model in result['models']]}")
                    return available
                else:
                    logger.error(f"Unexpected response format from Ollama: {result}")
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    return False
        except httpx.ReadTimeout:
            logger.error(f"Timeout while checking model availability (attempt {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return False
        except Exception as e:
            logger.error(f"Error checking model availability (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return False

    # If we get here, all attempts failed
    logger.error(f"All {max_retries + 1} attempts to check model availability failed")
    return False


def restart_ollama_service() -> bool:
    """
    Attempt to restart the Ollama service if it's not responding.

    Returns:
        True if restart was successful, False otherwise
    """
    try:
        logger.warning("Attempting to restart Ollama service...")

        # Different commands based on operating system
        if platform.system() == "Windows":
            # On Windows, we need to kill and restart the Ollama process
            # First try to kill any running Ollama processes
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Then start Ollama again (this assumes Ollama is in the PATH)
            subprocess.Popen(["ollama", "serve"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           creationflags=subprocess.CREATE_NO_WINDOW)

        elif platform.system() == "Linux":
            # On Linux, we can use systemctl if Ollama is installed as a service
            subprocess.run(["systemctl", "restart", "ollama"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        elif platform.system() == "Darwin":  # macOS
            # On macOS, we can use launchctl if Ollama is installed as a service
            subprocess.run(["killall", "ollama"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.Popen(["ollama", "serve"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for Ollama to start up
        logger.info("Waiting for Ollama service to start...")
        time.sleep(10)  # Give it more time to start

        # Check if it's running now
        for attempt in range(5):  # Try more times
            try:
                with httpx.Client(timeout=10.0) as client:  # Increase timeout
                    response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
                    if response.status_code == 200:
                        logger.info("Ollama service restarted successfully")
                        return True
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/5 to connect to restarted Ollama failed: {e}")
            # Increase wait time between checks
            time.sleep(3 + attempt * 2)  # 3s, 5s, 7s, 9s, 11s

        logger.error("Failed to restart Ollama service")
        return False

    except Exception as e:
        logger.error(f"Error restarting Ollama service: {e}")
        return False
