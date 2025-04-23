"""
Transliteration module using IndicXlit for the multilingual chatbot.
This module provides functions to transliterate text between different scripts.
"""
import logging
import os
import sys
from typing import List, Optional

# Add the IndicXlit directory to the Python path
INDICXLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IndicXlit-v1.0", "app")
sys.path.append(INDICXLIT_DIR)

# Initialize logger
logger = logging.getLogger(__name__)

# Global variables to hold the transliteration engines
_en2indic_engine = None
_indic2en_engine = None

# Language code mappings
LANG_CODE_MAP = {
    "eng_Latn": "en",
    "hin_Deva": "hi",
    "ben_Beng": "bn",
    "mar_Deva": "mr"
}

def load_engines() -> None:
    """Load the IndicXlit transliteration engines."""
    global _en2indic_engine, _indic2en_engine
    try:
        logger.info("Loading IndicXlit transliteration engines...")

        # Check for required dependencies
        missing_deps = []
        try:
            import ujson
            # Just to avoid unused import warning
            ujson.dumps({})
        except ImportError:
            missing_deps.append("ujson")
        try:
            import pydload
            # Just to avoid unused import warning
            pydload.__version__
        except ImportError:
            missing_deps.append("pydload")
        try:
            import tqdm
            # Just to avoid unused import warning
            tqdm.__version__
        except ImportError:
            missing_deps.append("tqdm")

        if missing_deps:
            logger.warning(f"Missing dependencies for IndicXlit: {', '.join(missing_deps)}")
            logger.warning("Install them with: pip install " + " ".join(missing_deps))
            return

        # Import the XlitEngine from the IndicXlit package
        from ai4bharat.transliteration import XlitEngine

        # Create engines for both directions
        _en2indic_engine = XlitEngine(
            beam_width=4,
            rescore=True,
            model_type="transformer",
            src_script_type="roman"
        )

        _indic2en_engine = XlitEngine(
            beam_width=4,
            rescore=False,
            model_type="transformer",
            src_script_type="indic"
        )

        logger.info("IndicXlit transliteration engines loaded successfully.")

        # Log supported languages
        logger.info(f"Supported languages for en2indic: {_en2indic_engine.all_supported_langs}")
        logger.info(f"Supported languages for indic2en: {_indic2en_engine.all_supported_langs}")

    except Exception as e:
        logger.error(f"Error loading IndicXlit transliteration engines: {e}")
        _en2indic_engine, _indic2en_engine = None, None

def transliterate_word(text: str, source_lang: str, target_lang: str, num_suggestions: int = 1) -> Optional[List[str]]:
    """
    Transliterate a word from source language to target language.

    Args:
        text: Text to transliterate
        source_lang: Source language code in our format (e.g., 'eng_Latn')
        target_lang: Target language code in our format (e.g., 'hin_Deva')
        num_suggestions: Number of transliteration suggestions to return

    Returns:
        List of transliterated text suggestions or None if transliteration fails
    """
    if not text.strip():
        return [text]

    # Skip transliteration if source and target languages are the same
    if source_lang == target_lang:
        return [text]

    # Map our language codes to IndicXlit language codes
    src_lang_code = LANG_CODE_MAP.get(source_lang)
    tgt_lang_code = LANG_CODE_MAP.get(target_lang)

    if not src_lang_code or not tgt_lang_code:
        logger.warning(f"Unsupported language code: {source_lang} or {target_lang}")
        return [text]

    try:
        # English to Indic language transliteration
        if src_lang_code == "en" and tgt_lang_code in ["hi", "bn", "mr"]:
            if _en2indic_engine is None:
                logger.warning("en2indic engine not loaded. Loading now...")
                load_engines()
                if _en2indic_engine is None:
                    return [text]

            result = _en2indic_engine.translit_word(text, tgt_lang_code, topk=num_suggestions)
            logger.info(f"Transliterated from {source_lang} to {target_lang}: {result[0] if result else text}")
            return result

        # Indic language to English transliteration
        elif src_lang_code in ["hi", "bn", "mr"] and tgt_lang_code == "en":
            if _indic2en_engine is None:
                logger.warning("indic2en engine not loaded. Loading now...")
                load_engines()
                if _indic2en_engine is None:
                    return [text]

            result = _indic2en_engine.translit_word(text, src_lang_code, topk=num_suggestions)
            logger.info(f"Transliterated from {source_lang} to {target_lang}: {result[0] if result else text}")
            return result

        else:
            logger.warning(f"Transliteration from {source_lang} to {target_lang} not supported")
            return [text]

    except Exception as e:
        logger.error(f"Error transliterating text: {e}")
        return [text]

def transliterate_sentence(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Transliterate a sentence from source language to target language.

    Args:
        text: Text to transliterate
        source_lang: Source language code in our format (e.g., 'eng_Latn')
        target_lang: Target language code in our format (e.g., 'hin_Deva')

    Returns:
        Transliterated text or None if transliteration fails
    """
    if not text.strip():
        return text

    # Skip transliteration if source and target languages are the same
    if source_lang == target_lang:
        return text

    # Map our language codes to IndicXlit language codes
    src_lang_code = LANG_CODE_MAP.get(source_lang)
    tgt_lang_code = LANG_CODE_MAP.get(target_lang)

    if not src_lang_code or not tgt_lang_code:
        logger.warning(f"Unsupported language code: {source_lang} or {target_lang}")
        return text

    try:
        # English to Indic language transliteration
        if src_lang_code == "en" and tgt_lang_code in ["hi", "bn", "mr"]:
            if _en2indic_engine is None:
                logger.warning("en2indic engine not loaded. Loading now...")
                load_engines()
                if _en2indic_engine is None:
                    return text

            result = _en2indic_engine.translit_sentence(text, tgt_lang_code)
            logger.info(f"Transliterated sentence from {source_lang} to {target_lang}")
            return result

        # Indic language to English transliteration
        elif src_lang_code in ["hi", "bn", "mr"] and tgt_lang_code == "en":
            if _indic2en_engine is None:
                logger.warning("indic2en engine not loaded. Loading now...")
                load_engines()
                if _indic2en_engine is None:
                    return text

            result = _indic2en_engine.translit_sentence(text, src_lang_code)
            logger.info(f"Transliterated sentence from {source_lang} to {target_lang}")
            return result

        else:
            logger.warning(f"Transliteration from {source_lang} to {target_lang} not supported")
            return text

    except Exception as e:
        logger.error(f"Error transliterating sentence: {e}")
        return text

# Try to load the engines when the module is imported
load_engines()
