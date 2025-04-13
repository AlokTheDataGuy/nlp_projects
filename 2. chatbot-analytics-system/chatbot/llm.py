# File: chatbot/llm.py
# Interface with Ollama and Cogito model

import requests
import json
from typing import Optional

class OllamaClient:
    def __init__(self, model="cogito:8b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"
        
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Get a completion from the Ollama model"""
        try:
            # Add instructions to avoid thinking process in the system prompt
            enhanced_system_prompt = """You are a helpful AI assistant. Be concise and clear in your responses.
            DO NOT show your thinking process or reasoning steps.
            DO NOT use phrases like 'let me think', 'I'll analyze', or similar thinking indicators.
            Provide ONLY the final, direct answer to the user's question."""
            
            if system_prompt:
                enhanced_system_prompt = f"{system_prompt}\n{enhanced_system_prompt}"
                
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,  # Ensure we're not using streaming mode
                "system": enhanced_system_prompt
            }
                
            response = requests.post(self.api_generate, json=payload)
            
            if response.status_code == 200:
                # Try to parse the response as JSON
                try:
                    # Get only the first line of the response text to avoid parsing issues
                    first_line = response.text.strip().split('\n')[0]
                    json_response = json.loads(first_line)
                    raw_response = json_response.get("response", "")
                    
                    # Clean up the response to remove thinking patterns
                    cleaned_response = self._clean_response(raw_response)
                    return cleaned_response
                except json.JSONDecodeError:
                    # If JSON parsing fails, return a fallback message
                    return "I'm having trouble processing your request right now. Please try again."
            else:
                print(f"Error: API returned status code {response.status_code}")
                return f"Error communicating with LLM: {response.status_code}"
        except Exception as e:
            print(f"Exception when calling Ollama API: {e}")
            return f"Error: {str(e)}"
            
    def _clean_response(self, response: str) -> str:
        """Clean up the response to remove thinking patterns"""
        # Remove common thinking indicators
        thinking_patterns = [
            "Let me think", "I'll analyze", "Let's analyze", "Let's think",
            "Let me analyze", "I'm thinking", "Thinking about", "Analyzing",
            "First, I'll", "Step 1:", "Step 2:", "Step 3:", "First step",
            "To answer this", "To solve this", "To address this",
            "My reasoning", "My thought process", "My analysis"
        ]
        
        # Check if the response contains any thinking patterns
        response_lower = response.lower()
        contains_thinking = any(pattern.lower() in response_lower for pattern in thinking_patterns)
        
        if contains_thinking:
            # Try to extract just the conclusion/final answer
            conclusion_indicators = [
                "In conclusion", "Therefore", "So,", "Thus,", "To summarize",
                "In summary", "Finally,", "The answer is", "The solution is",
                "To conclude", "In the end", "Ultimately", "In short"
            ]
            
            for indicator in conclusion_indicators:
                if indicator.lower() in response_lower:
                    # Find the indicator and return everything after it
                    idx = response_lower.find(indicator.lower())
                    conclusion = response[idx:].strip()
                    return conclusion
        
        # If no conclusion indicators found or no thinking patterns detected, return the original response
        return response

    def analyze_topic(self, user_query: str) -> str:
        """Analyze the topic of a user query using the Cogito LLM"""
        # Define the allowed topic categories
        allowed_topics = [
            "General Question",
            "Technical Query",
            "Informational Request",
            "Feedback",
            "Opinion or Suggestion",
            "Complaint",
            "Task or Command",
            "Code or Programming",
            "Help or Guidance",
            "Miscellaneous"
        ]
        
        # Create a detailed prompt with examples for better classification
        prompt = f"""You are a query classification expert. Your task is to categorize the following user query into exactly one of the predefined categories.

        CATEGORIES:
        - General Question: Basic factual questions about the world, like "What is the capital of France?" or "How many continents are there?"
        - Technical Query: Questions about technical issues, errors, or technical concepts, like "How do I fix a 404 error?" or "Why is my Python code so slow?"
        - Informational Request: Requests for detailed information or explanations, like "Tell me about the history of the Eiffel Tower" or "What are the effects of global warming?"
        - Feedback: Comments about the quality or performance of a product/service, like "The chatbot's speed needs improvement" or "It's good, but could be more accurate."
        - Opinion or Suggestion: Expressing a viewpoint or recommendation, like "AI should be more regulated" or "You should add more personality to your replies."
        - Complaint: Expressing dissatisfaction or reporting problems, like "This tool keeps crashing!" or "None of my queries are being answered!"
        - Task or Command: Asking the system to perform a specific action, like "Summarize this text for me" or "Set a reminder for 5 PM."
        - Code or Programming: Questions about programming, coding, or algorithms, like "Write a Python function to reverse a string" or "What does 'Big O' mean in algorithms?"
        - Help or Guidance: Asking for assistance or instructions, like "Can you guide me on how to create a portfolio website?" or "How do I start freelancing online?"
        - Miscellaneous: Greetings, jokes, or queries that don't fit other categories, like "Tell me a joke" or "Hi there!"

        USER QUERY: "{user_query}"
        
        INSTRUCTIONS:
        1. Analyze the query carefully
        2. Try to Select EXACTLY ONE category from the list above and if doesn't fit into any category, choose Miscellaneous
        3. Return ONLY the category name, nothing else
        4. Do not add any explanations, punctuation, or additional text
        
        CATEGORY:"""

        # Get the raw topic from the LLM with low temperature for consistent results
        raw_topic = self.get_completion(prompt, temperature=0.1, max_tokens=20).strip()
        
        # Clean up the response to ensure it's one of our allowed topics
        for topic in allowed_topics:
            if topic.lower() in raw_topic.lower():
                return topic
        
        # If no match is found, return Miscellaneous as a fallback
        return "Miscellaneous"
