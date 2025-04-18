import requests
import json
from typing import Dict, Any, List, Optional
import os

class Chatbot:
    def __init__(self):
        self.model = "llama3.1:8b"
        self.ollama_url = "http://localhost:11434/api/chat"
        
        # Define sentiment-based prompt templates
        self.prompt_templates = {
            "positive": "The user seems to be in a positive mood. Respond in a friendly and enthusiastic manner while addressing their query: {message}",
            "neutral": "The user has a neutral tone. Respond professionally and directly to their query: {message}",
            "negative": "The user seems to be frustrated or upset. Respond with empathy and understanding while addressing their concern: {message}",
            "default": "Respond to the following user message: {message}"
        }
    
    def generate_response(self, message: str, sentiment: Dict[str, Any], conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response based on the user message and sentiment"""
        # Select appropriate prompt template based on sentiment
        sentiment_label = sentiment.get("label", "neutral")
        prompt_template = self.prompt_templates.get(sentiment_label, self.prompt_templates["default"])
        
        # Format the prompt with the user message
        formatted_prompt = prompt_template.format(message=message)
        
        # Prepare conversation history for context
        messages = []
        
        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-5:]:  # Use last 5 messages for context
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        
        # Add the current message
        messages.append({"role": "user", "content": formatted_prompt})
        
        # Call Ollama API
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            else:
                return f"Sorry, I'm having trouble generating a response. Error: {response.status_code}"
        
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
