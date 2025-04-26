import aiohttp
from typing import Optional, Dict, Any

class OllamaService:
    def __init__(self, model_name: str = "llama3.1:8b", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
    
    async def generate(self, prompt: str, system_message: Optional[str] = None, max_tokens: int = 2048) -> str:
        """Generate a response using the Ollama API"""
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        if system_message:
            payload["system"] = system_message
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/generate", 
                    headers=headers, 
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["response"]
                    else:
                        error_text = await response.text()
                        return f"Error generating response: {response.status} - {error_text}"
        except Exception as e:
            return f"Error connecting to Ollama service: {str(e)}"