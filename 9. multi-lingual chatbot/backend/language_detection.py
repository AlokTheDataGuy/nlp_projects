"""
Language detection module using fastText and langdetect as fallback.
"""
import os
import logging
from typing import Optional
import fasttext
import langdetect
from langdetect import DetectorFactory

from config import FASTTEXT_MODEL_PATH, DETECT_TO_CODE_MAP, DEFAULT_LANGUAGE

# Set seed for langdetect for reproducibility
DetectorFactory.seed = 0

# Initialize logger
logger = logging.getLogger(__name__)

# Global variable to hold the fastText model
_model = None


def load_model() -> None:
    """Load the fastText language detection model."""
    global _model
    try:
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            logger.warning(f"FastText model not found at {FASTTEXT_MODEL_PATH}. "
                          "Will download it automatically.")
            os.makedirs(os.path.dirname(FASTTEXT_MODEL_PATH), exist_ok=True)
            
        _model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info("FastText language detection model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading FastText model: {e}")
        _model = None


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    
    Args:
        text: The text to detect language for
        
    Returns:
        Language code in the format used by the system
    """
    if not text.strip():
        return DEFAULT_LANGUAGE
    
    # Try fastText first
    if _model is not None:
        try:
            predictions = _model.predict(text, k=1)
            lang_code = predictions[0][0].replace("__label__", "")
            
            # Map to our language codes
            if lang_code in DETECT_TO_CODE_MAP:
                return DETECT_TO_CODE_MAP[lang_code]
        except Exception as e:
            logger.error(f"Error detecting language with fastText: {e}")
    
    # Fallback to langdetect
    try:
        lang_code = langdetect.detect(text)
        if lang_code in DETECT_TO_CODE_MAP:
            return DETECT_TO_CODE_MAP[lang_code]
    except Exception as e:
        logger.error(f"Error detecting language with langdetect: {e}")
    
    # Default to English if detection fails
    return DEFAULT_LANGUAGE


# Load the model when the module is imported
load_model()
