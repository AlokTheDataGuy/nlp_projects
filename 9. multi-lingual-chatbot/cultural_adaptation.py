"""
Cultural adaptation module for the chatbot.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from config import CULTURAL_ADAPTATION, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

class CulturalAdapter:
    """
    Adapts responses to be culturally appropriate for the target language.
    """
    
    def __init__(self):
        """Initialize the cultural adapter."""
        self.honorifics = CULTURAL_ADAPTATION["honorifics"]
        self.greetings = CULTURAL_ADAPTATION["greetings"]
        logger.info("Cultural adapter initialized")
    
    def adapt_response(self, response: str, target_lang: str) -> str:
        """
        Adapt a response to be culturally appropriate for the target language.
        
        Args:
            response: Response in the target language
            target_lang: Target language code
            
        Returns:
            Culturally adapted response
        """
        if target_lang not in SUPPORTED_LANGUAGES:
            return response
        
        # Apply language-specific adaptations
        if target_lang == "hin_Deva":
            response = self._adapt_hindi(response)
        elif target_lang == "ben_Beng":
            response = self._adapt_bengali(response)
        elif target_lang == "mar_Deva":
            response = self._adapt_marathi(response)
        
        return response
    
    def _adapt_hindi(self, text: str) -> str:
        """
        Apply Hindi-specific cultural adaptations.
        
        Args:
            text: Text to adapt
            
        Returns:
            Adapted text
        """
        # Replace formal pronouns if needed
        # In Hindi, "आप" is formal, "तुम" is informal, "तू" is very informal
        
        # Format dates according to Indian convention (DD-MM-YYYY)
        text = self._format_dates(text)
        
        # Format numbers with Indian numbering system
        text = self._format_numbers_indian(text)
        
        return text
    
    def _adapt_bengali(self, text: str) -> str:
        """
        Apply Bengali-specific cultural adaptations.
        
        Args:
            text: Text to adapt
            
        Returns:
            Adapted text
        """
        # Replace formal pronouns if needed
        # In Bengali, "আপনি" is formal, "তুমি" is informal, "তুই" is very informal
        
        # Format dates according to Bengali convention
        text = self._format_dates(text)
        
        # Format numbers with Bengali numbering system
        text = self._format_numbers_indian(text)
        
        return text
    
    def _adapt_marathi(self, text: str) -> str:
        """
        Apply Marathi-specific cultural adaptations.
        
        Args:
            text: Text to adapt
            
        Returns:
            Adapted text
        """
        # Replace formal pronouns if needed
        # In Marathi, "आपण" is formal, "तुम्ही" is semi-formal, "तू" is informal
        
        # Format dates according to Indian convention
        text = self._format_dates(text)
        
        # Format numbers with Indian numbering system
        text = self._format_numbers_indian(text)
        
        return text
    
    def _format_dates(self, text: str) -> str:
        """
        Format dates according to Indian convention (DD-MM-YYYY).
        
        Args:
            text: Text containing dates
            
        Returns:
            Text with formatted dates
        """
        # Find dates in MM/DD/YYYY format and convert to DD-MM-YYYY
        mm_dd_yyyy_pattern = r'\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/(\d{4})\b'
        text = re.sub(mm_dd_yyyy_pattern, r'\2-\1-\3', text)
        
        return text
    
    def _format_numbers_indian(self, text: str) -> str:
        """
        Format numbers according to Indian numbering system (with lakhs and crores).
        
        Args:
            text: Text containing numbers
            
        Returns:
            Text with formatted numbers
        """
        # Find numbers and format them with commas according to Indian system
        # e.g., 1000000 -> 10,00,000 (10 lakh)
        def indian_format(match):
            number = int(match.group(0))
            if number < 1000:
                return str(number)
            
            s = str(number)
            result = s[-3:]
            s = s[:-3]
            
            while s:
                result = s[-2:] + ',' + result if len(s) >= 2 else s + ',' + result
                s = s[:-2]
            
            return result
        
        # Find numbers and format them
        number_pattern = r'\b\d{4,}\b'
        text = re.sub(number_pattern, indian_format, text)
        
        return text
    
    def detect_formality(self, text: str, lang: str) -> str:
        """
        Detect the formality level in the input text.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            Formality level: "formal", "informal", or "neutral"
        """
        if lang not in self.honorifics:
            return "neutral"
        
        honorifics = self.honorifics[lang]
        
        # Check for formal pronouns
        if honorifics[0] in text:
            return "formal"
        
        # Check for informal pronouns
        if honorifics[1] in text or honorifics[2] in text:
            return "informal"
        
        return "neutral"
    
    def adapt_formality(self, text: str, lang: str, formality: str) -> str:
        """
        Adapt the response to match the detected formality level.
        
        Args:
            text: Response text
            lang: Language code
            formality: Formality level ("formal", "informal", or "neutral")
            
        Returns:
            Adapted response with matching formality
        """
        if lang not in self.honorifics or formality == "neutral":
            return text
        
        # Apply formality adaptations based on language
        if lang == "hin_Deva":
            if formality == "formal":
                # Replace informal pronouns with formal ones
                text = text.replace("तुम", "आप").replace("तू", "आप")
            elif formality == "informal":
                # Replace formal pronouns with informal ones
                text = text.replace("आप", "तुम")
        
        elif lang == "ben_Beng":
            if formality == "formal":
                # Replace informal pronouns with formal ones
                text = text.replace("তুমি", "আপনি").replace("তুই", "আপনি")
            elif formality == "informal":
                # Replace formal pronouns with informal ones
                text = text.replace("আপনি", "তুমি")
        
        elif lang == "mar_Deva":
            if formality == "formal":
                # Replace informal pronouns with formal ones
                text = text.replace("तुम्ही", "आपण").replace("तू", "आपण")
            elif formality == "informal":
                # Replace formal pronouns with informal ones
                text = text.replace("आपण", "तुम्ही")
        
        return text
    
    def adapt_greeting(self, lang: str, time_of_day: Optional[str] = None) -> str:
        """
        Get a culturally appropriate greeting for the given language.
        
        Args:
            lang: Language code
            time_of_day: Optional time of day ("morning", "afternoon", "evening")
            
        Returns:
            Appropriate greeting
        """
        if lang not in self.greetings:
            # Default to English
            lang = "eng_Latn"
        
        greetings = self.greetings[lang]
        
        # Select greeting based on time of day if specified
        if time_of_day:
            if time_of_day == "morning" and len(greetings) >= 3:
                return greetings[2]  # Good morning
            elif time_of_day == "afternoon" and len(greetings) >= 4:
                return greetings[3]  # Good afternoon
            elif time_of_day == "evening" and len(greetings) >= 5:
                return greetings[4]  # Good evening
        
        # Default to general greeting
        return greetings[0]
