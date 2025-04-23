"""
Translation module using ai4bharat/IndicBART.
"""
import logging
from typing import Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import TRANSLATION_MODEL, TRANSLATION_DEVICE

# Initialize logger
logger = logging.getLogger(__name__)

# Global variables to hold the translation model and tokenizer
_model = None
_tokenizer = None


def load_model() -> None:
    """Load the IndicBART translation model and tokenizer."""
    global _model, _tokenizer
    try:
        logger.info(f"Loading translation model {TRANSLATION_MODEL} on {TRANSLATION_DEVICE}...")
        logger.info("This may take a few minutes the first time as the model is downloaded...")
        _model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)

        # Move model to the specified device
        _model.to(TRANSLATION_DEVICE)

        logger.info("Translation model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading translation model: {e}")
        logger.error("Make sure you have an internet connection to download the model.")
        _model, _tokenizer = None, None


def translate(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Translate text from source language to target language.

    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Translated text or None if translation fails
    """
    if not text.strip():
        return text

    # Skip translation if source and target languages are the same
    if source_lang == target_lang:
        return text

    # Skip translation if model or tokenizer is not loaded
    if _model is None or _tokenizer is None:
        logger.warning("Translation model not loaded. Skipping translation.")
        return text

    try:
        # Map target language code to the format expected by the model
        # IndicBART only needs the target language code
        tgt_lang_code = _map_to_model_lang_code(target_lang)

        # For IndicBART, we need to format the input with language tags
        # Format: <2xx> text </s>, where xx is the target language code
        prefix = f"<2{tgt_lang_code}>"

        # Tokenize the input text
        inputs = _tokenizer(prefix + text, return_tensors="pt").to(TRANSLATION_DEVICE)

        # Generate translation
        with torch.no_grad():
            generated_tokens = _model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

        # Decode the generated tokens
        translation = _tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        logger.info(f"Translated from {source_lang} to {target_lang}")
        logger.debug(f"Original: {text[:50]}...")
        logger.debug(f"Translation: {translation[:50]}...")

        return translation
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return text


def _map_to_model_lang_code(lang_code: str) -> str:
    """
    Map our language codes to the format expected by the IndicBART model.

    Args:
        lang_code: Our internal language code

    Returns:
        Language code in the format expected by the model
    """
    # Mapping from our language codes to IndicBART language codes
    # IndicBART uses 2-letter language codes: en, hi, bn, mr, etc.
    mapping = {
        "eng_Latn": "en",
        "hin_Deva": "hi",
        "ben_Beng": "bn",
        "mar_Deva": "mr"
    }

    model_code = mapping.get(lang_code, "en")
    logger.debug(f"Mapped language code {lang_code} to IndicBART code {model_code}")
    return model_code


# Load the model when the module is imported
load_model()
