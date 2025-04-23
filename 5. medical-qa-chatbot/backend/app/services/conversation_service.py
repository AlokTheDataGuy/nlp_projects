import re
import random
from typing import Dict, List, Tuple, Optional

class ConversationService:
    """Service for handling conversational interactions"""
    
    def __init__(self):
        # Define patterns for common conversational queries
        self.patterns: Dict[str, List[str]] = {
            "greeting": [
                r"^hi+\s*$",
                r"^hello+\s*$",
                r"^hey+\s*$",
                r"^greetings\s*$",
                r"^good\s+(morning|afternoon|evening)\s*$"
            ],
            "how_are_you": [
                r"^how\s+are\s+you\s*\??$",
                r"^how\s+are\s+you\s+doing\s*\??$",
                r"^how\s+is\s+it\s+going\s*\??$"
            ],
            "what_are_you": [
                r"^what\s+are\s+you\s*\??$",
                r"^who\s+are\s+you\s*\??$",
                r"^tell\s+me\s+about\s+yourself\s*$",
                r"^what\s+is\s+this\s*\??$",
                r"^what\s+can\s+you\s+do\s*\??$",
                r"^what\s+do\s+you\s+do\s*\??$"
            ],
            "thanks": [
                r"^thanks\s*$",
                r"^thank\s+you\s*$",
                r"^thx\s*$",
                r"^ty\s*$"
            ],
            "bye": [
                r"^bye\s*$",
                r"^goodbye\s*$",
                r"^see\s+you\s*$",
                r"^farewell\s*$"
            ],
            "help": [
                r"^help\s*$",
                r"^help\s+me\s*$",
                r"^i\s+need\s+help\s*$",
                r"^what\s+can\s+i\s+ask\s*\??$"
            ]
        }
        
        # Define responses for each pattern type
        self.responses: Dict[str, List[str]] = {
            "greeting": [
                "Hello! How can I help with your medical questions today?",
                "Hi there! I'm here to provide medical information. What would you like to know?",
                "Greetings! I'm your medical assistant. What questions do you have?"
            ],
            "how_are_you": [
                "I'm functioning well, thank you! How can I assist with your medical questions?",
                "I'm here and ready to help with your health questions. What would you like to know?",
                "I'm operational and ready to provide medical information. What can I help you with?"
            ],
            "what_are_you": [
                "I'm a medical Q&A chatbot designed to answer health-related questions using information from the MedQuAD dataset and enhanced by AI models like Meditron and Llama 3.1. I can provide information about medical conditions, symptoms, treatments, and more. What would you like to know?",
                "I'm your medical assistant, built to provide reliable health information from trusted medical sources. I use the MedQuAD dataset and AI models to give you accurate and understandable answers. How can I help you today?",
                "I'm a specialized medical chatbot that can answer questions about health conditions, symptoms, treatments, and more. I use information from medical databases and enhance responses with AI to make them easier to understand. What medical information are you looking for?"
            ],
            "thanks": [
                "You're welcome! Feel free to ask if you have more questions.",
                "Happy to help! Let me know if you need anything else.",
                "Glad I could assist. Don't hesitate to ask more questions if needed."
            ],
            "bye": [
                "Goodbye! Take care of your health.",
                "Farewell! Remember I'm here if you have more medical questions later.",
                "See you later! Stay healthy!"
            ],
            "help": [
                "I can answer questions about medical conditions, symptoms, treatments, medications, and general health information. Just ask something like 'What is diabetes?' or 'What are the symptoms of a cold?' or 'How is high blood pressure treated?'",
                "You can ask me about various health topics like diseases, symptoms, treatments, or medications. For example, try asking 'What causes asthma?' or 'What are the symptoms of a heart attack?' or 'How is pneumonia treated?'",
                "I'm here to help with medical information. You can ask questions about health conditions, symptoms, treatments, or preventive care. For instance, try 'What is arthritis?' or 'What are common symptoms of the flu?' or 'How is depression treated?'"
            ]
        }
    
    def detect_conversation_type(self, text: str) -> Optional[str]:
        """
        Detect if the input is a conversational query
        
        Args:
            text: The user's input text
            
        Returns:
            The type of conversation or None if not conversational
        """
        # Normalize text: lowercase and strip
        normalized_text = text.lower().strip()
        
        # Check against each pattern type
        for conv_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.match(pattern, normalized_text):
                    return conv_type
        
        return None
    
    def get_response(self, conv_type: str) -> str:
        """
        Get a response for the given conversation type
        
        Args:
            conv_type: The type of conversation
            
        Returns:
            A response string
        """
        if conv_type in self.responses:
            return random.choice(self.responses[conv_type])
        
        # Fallback response
        return "I'm here to help with medical questions. What would you like to know?"
