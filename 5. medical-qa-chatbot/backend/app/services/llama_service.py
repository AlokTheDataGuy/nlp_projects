import requests
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class LlamaService:
    """Service for interacting with Llama 3.1 model via Ollama"""

    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        self.api_base = "http://localhost:11434/api"

    def curate_response(self, question: str, original_answer: str) -> str:
        """
        Use Llama 3.1 to curate and enhance the medical response for better understanding

        Args:
            question: The original user question
            original_answer: The answer from the retrieval system or Meditron

        Returns:
            Enhanced and curated response
        """
        try:
            prompt = f"""You are a helpful medical assistant that provides extremely concise and clear medical information.

Original Question: {question}

Original Answer: {original_answer}

FIRST, evaluate if the original answer is relevant to the question:
- If the original answer is NOT about the topic in the question, respond ONLY with: "I don't have specific information about [topic] at the moment. Would you like to ask about something else?"
- If the original answer contains irrelevant information or seems to be about a different topic, respond ONLY with: "I don't have specific information about [topic] at the moment. Would you like to ask about something else?"

If the original answer IS relevant to the question, create an EXTREMELY CONCISE response that:

1. Is NO MORE THAN 100 WORDS TOTAL
2. Extracts ONLY the 2-3 most important points directly answering the question
3. Uses a simple structure with clear headings (What is it, Causes, Symptoms, Treatment)
4. Uses bullet points for lists (never more than 3-4 bullet points)
5. Explains medical terms simply in parentheses
6. Uses a friendly tone

Format your answer like this:
[Main heading]
[1-2 sentence introduction]

[Subheading (if needed)]:
• [Key point 1]
• [Key point 2]
• [Key point 3]

[Brief conclusion or advice (optional)]

Be ruthlessly brief. Cut any information not directly answering the question.

Enhanced Answer:"""

            # Call Ollama API
            response = requests.post(
                f"{self.api_base}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 500,  # Limit token generation
                        "stop": ["\n\n\n"]  # Stop on triple newline to prevent rambling
                    }
                }
            )

            if response.status_code == 200:
                result = response.json()
                enhanced_answer = result.get("response", "").strip()
                return enhanced_answer
            else:
                logger.error(f"Error from Ollama API: {response.text}")
                return original_answer

        except Exception as e:
            logger.error(f"Error in Llama curation: {str(e)}")
            # Return the original answer if there's an error
            return original_answer

    def answer_simple_question(self, question: str) -> str:
        """
        Use Llama 3.1 to directly answer a simple medical question

        Args:
            question: The user's question

        Returns:
            A direct answer from Llama
        """
        try:
            prompt = f"""You are a helpful medical assistant that provides extremely concise and clear medical information.

Question: {question}

Provide a brief, accurate answer about this medical topic. Your answer should:

1. Be NO MORE THAN 100 WORDS TOTAL
2. Start with a clear definition or explanation
3. Include 2-3 key facts or points using bullet points
4. Use simple language, explaining medical terms in parentheses
5. Be factual and evidence-based

Answer:"""

            # Call Ollama API
            response = requests.post(
                f"{self.api_base}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 500,  # Limit token generation
                        "stop": ["\n\n\n"]  # Stop on triple newline to prevent rambling
                    }
                }
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                return answer
            else:
                logger.error(f"Error from Ollama API: {response.text}")
                raise Exception("Failed to get response from Llama model")

        except Exception as e:
            logger.error(f"Error in Llama direct answer: {str(e)}")
            raise e

    def health_check(self) -> bool:
        """Check if Ollama is running and the model is available"""
        try:
            response = requests.get(f"{self.api_base}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model.get("name", "").startswith(self.model_name.split(":")[0]) for model in models)
            return False
        except Exception:
            return False
