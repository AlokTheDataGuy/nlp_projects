from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import Dict, Any, List, Optional
import psutil
import gc
import os

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, model_name="microsoft/phi-3-mini-4k-instruct"):
        """
        Initialize the LLM manager.
        
        Args:
            model_name: The name of the model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self):
        """
        Load the model and tokenizer.
        """
        if self.is_loaded:
            return True
        
        try:
            # Check available resources
            resources = self._check_resources()
            if resources["is_constrained"]:
                logger.warning("System resources are constrained. Model loading may fail.")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.is_loaded = True
            logger.info(f"Successfully loaded model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def unload_model(self):
        """
        Unload the model to free up resources.
        """
        if not self.is_loaded:
            return True
        
        try:
            # Delete model and tokenizer
            del self.model
            del self.tokenizer
            
            # Run garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            
            logger.info("Model unloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False
    
    def generate_response(self, context: str, query: str, max_new_tokens: int = 512, 
                         temperature: float = 0.7, do_sample: bool = True) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            context: Context information from papers
            query: User query
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to use sampling
            
        Returns:
            Generated response
        """
        if not self.is_loaded:
            if not self.load_model():
                return "Error: Failed to load the language model."
        
        try:
            prompt = f"""
            You are an expert AI assistant specializing in computer science research. 
            Your task is to provide clear, accurate, and helpful information based on the context provided.
            
            Context from scientific papers:
            {context}
            
            User query: {query}
            
            Please provide a comprehensive explanation based on the scientific literature above. 
            Include relevant concepts, methodologies, and findings. If the context doesn't contain 
            enough information to answer the query, acknowledge this and provide general information 
            about the topic based on your knowledge.
            """
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _check_resources(self) -> Dict[str, Any]:
        """
        Check system resources.
        
        Returns:
            Dictionary with resource information
        """
        # Check system RAM
        available_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        
        # Check GPU memory if available
        gpu_usage = None
        total_gpu = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
            total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
        
        return {
            "available_ram_gb": available_ram,
            "gpu_usage_gb": gpu_usage,
            "total_gpu_gb": total_gpu,
            "is_constrained": available_ram < 2.0
        }

# Create a singleton instance
llm_manager = LLMManager()
