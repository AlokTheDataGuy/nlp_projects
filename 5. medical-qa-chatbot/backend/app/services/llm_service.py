import httpx
from typing import Dict, Any, Optional, List

class MeditronService:
    """
    Service for interacting with the Meditron LLM via Ollama
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "meditron:7b"
        self.client = httpx.Client(timeout=60.0)  # Longer timeout for LLM responses
    
    async def generate_response(self, 
                          query: str, 
                          context: Optional[str] = None,
                          entities: Optional[List[Dict[str, Any]]] = None,
                          temperature: float = 0.7,
                          max_tokens: int = 1024) -> str:
        """
        Generate a response from Meditron based on the query and context
        """
        # Format entities if available
        entities_text = ""
        if entities and len(entities) > 0:
            entities_text = "Identified medical entities:\n"
            for entity in entities:
                entities_text += f"- {entity['text']} (Type: {entity['type']})\n"
        
        # Construct the prompt
        system_prompt = """You are a helpful medical assistant providing information based on reliable medical sources.
        Provide accurate, evidence-based information. If you're unsure, acknowledge the limitations.
        Always include a disclaimer that you're not providing medical advice and users should consult healthcare professionals.
        Keep responses concise, clear, and focused on the medical question."""
        
        # Add context if available
        context_text = f"\nRelevant medical information:\n{context}\n" if context else ""
        
        # Construct the full prompt
        full_prompt = f"{system_prompt}\n\n{context_text}{entities_text}\nUser question: {query}\n\nResponse:"
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            # Make the API call to Ollama
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            return result.get("response", "I couldn't generate a response. Please try again.")
            
        except Exception as e:
            print(f"Error calling Meditron LLM: {str(e)}")
            return "I encountered an error while processing your question. Please try again later."
    
    def health_check(self) -> bool:
        """
        Check if Ollama is running and Meditron model is available
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model.get("name") == self.model for model in models)
            return False
        except Exception:
            return False
