"""
Language detection module using fastText.
"""

import os
import fasttext
import langid
import logging
from typing import Tuple, Dict, List, Optional
from config import LANGUAGE_DETECTION, SUPPORTED_LANGUAGES
from debug_utils import debug_checkpoint

logger = logging.getLogger(__name__)

class LanguageDetector:
    """
    Language detection using fastText model optimized for the supported languages.
    Falls back to langid for robustness.
    """

    def __init__(self):
        """Initialize the language detector."""
        self.supported_langs = list(SUPPORTED_LANGUAGES.keys())
        self.confidence_threshold = LANGUAGE_DETECTION["confidence_threshold"]
        self.default_language = LANGUAGE_DETECTION["default_language"]

        # Initialize fastText model if available
        self.ft_model = None
        model_path = LANGUAGE_DETECTION["model_path"]

        if os.path.exists(model_path):
            try:
                # Load the model with minimal memory footprint
                self.ft_model = fasttext.load_model(model_path)
                logger.info(f"Loaded fastText model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load fastText model: {e}")
        else:
            logger.warning(f"fastText model not found at {model_path}. Will download on first use.")

    def _ensure_model_loaded(self):
        """Ensure the fastText model is loaded."""
        if self.ft_model is None:
            try:
                # Download and load the compressed model
                os.makedirs(os.path.dirname(LANGUAGE_DETECTION["model_path"]), exist_ok=True)
                self.ft_model = fasttext.load_model(LANGUAGE_DETECTION["model_path"])
                logger.info("Downloaded and loaded fastText model")
            except Exception as e:
                logger.error(f"Failed to download fastText model: {e}")
                logger.info("Using langid as fallback")

    def _map_to_supported_language(self, lang_code: str) -> str:
        """Map language code to supported language code."""
        # Simple mapping from ISO codes to NLLB codes
        mapping = {
            "en": "eng_Latn",
            "hi": "hin_Deva",
            "bn": "ben_Beng",
            "mr": "mar_Deva"
        }

        # Additional script detection for Devanagari and Bengali scripts
        # This helps with detecting Hindi and Marathi which use Devanagari script
        if any(0x0900 <= ord(c) <= 0x097F for c in lang_code):  # Devanagari range
            # Check if it's more likely to be Marathi or Hindi
            if lang_code in ["mr", "mai", "kok"]:
                return "mar_Deva"  # Marathi
            return "hin_Deva"  # Default to Hindi for Devanagari

        # Bengali script detection
        if any(0x0980 <= ord(c) <= 0x09FF for c in lang_code):  # Bengali range
            return "ben_Beng"

        return mapping.get(lang_code, self.default_language)

    def _detect_script(self, text: str) -> Optional[str]:
        """
        Detect script based on Unicode character ranges.

        Args:
            text: Text to analyze

        Returns:
            Detected language code or None if no script is dominant
        """
        # Count characters in different script ranges
        devanagari_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        bengali_count = sum(1 for c in text if 0x0980 <= ord(c) <= 0x09FF)
        latin_count = sum(1 for c in text if (0x0041 <= ord(c) <= 0x005A) or (0x0061 <= ord(c) <= 0x007A))

        # Get the dominant script
        total_chars = len(text.strip())
        if total_chars == 0:
            return None

        # Calculate percentages
        devanagari_percent = devanagari_count / total_chars
        bengali_percent = bengali_count / total_chars
        latin_percent = latin_count / total_chars

        logger.info(f"Script detection - Devanagari: {devanagari_percent:.2f}, Bengali: {bengali_percent:.2f}, Latin: {latin_percent:.2f}")

        # Determine dominant script (if >40% of characters are from that script)
        if devanagari_percent > 0.4:
            # Further distinguish between Hindi and Marathi
            # Marathi-specific characters and word patterns
            marathi_markers = ['ळ', 'ऴ', 'ऱ']
            marathi_words = ['आहे', 'आहात', 'काय', 'तुझे', 'नाव', 'मराठी', 'कसे', 'तुम्ही']

            # Check for Marathi-specific characters
            if any(marker in text for marker in marathi_markers):
                logger.info(f"Detected Marathi-specific character in: '{text}'")
                return "mar_Deva"  # Marathi

            # Check for Marathi-specific words
            if any(word in text for word in marathi_words):
                logger.info(f"Detected Marathi-specific word in: '{text}'")
                return "mar_Deva"  # Marathi

            # Hindi-specific words
            hindi_words = ['है', 'हैं', 'क्या', 'तुम', 'नाम', 'हिंदी', 'कैसे', 'आप']
            if any(word in text for word in hindi_words):
                logger.info(f"Detected Hindi-specific word in: '{text}'")
                return "hin_Deva"  # Hindi

            # Default to Hindi for Devanagari if no specific markers found
            return "hin_Deva"  # Hindi

        if bengali_percent > 0.4:
            return "ben_Beng"  # Bengali

        if latin_percent > 0.4:
            return "eng_Latn"  # English

        return None

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the given text.

        Args:
            text: The text to detect language for

        Returns:
            Tuple of (language_code, confidence)
        """
        debug_checkpoint("Starting language detection", {"text": text})
        logger.info(f"Detecting language for text: '{text}'")

        if not text or text.strip() == "":
            logger.info("Empty text, returning default language")
            debug_checkpoint("Empty text, returning default", {"default_language": self.default_language})
            return self.default_language, 1.0

        # Clean the text - remove newlines and extra whitespace
        cleaned_text = text.replace('\n', ' ').strip()
        logger.info(f"Cleaned text for detection: '{cleaned_text}'")
        debug_checkpoint("Cleaned text for detection", {"cleaned_text": cleaned_text})

        # Try direct script detection first
        debug_checkpoint("Before script detection", {"cleaned_text": cleaned_text})
        script_lang = self._detect_script(cleaned_text)
        debug_checkpoint("After script detection", {"script_lang": script_lang})

        if script_lang:
            logger.info(f"Script detection identified: {script_lang}")
            debug_checkpoint("Using script-based detection result", {
                "language": script_lang,
                "confidence": 0.9
            })
            return script_lang, 0.9  # High confidence for script-based detection

        # Try fastText first if available
        if self.ft_model is not None or os.path.exists(LANGUAGE_DETECTION["model_path"]):
            self._ensure_model_loaded()

            if self.ft_model is not None:
                try:
                    # Get prediction from fastText
                    logger.info("Using fastText for language detection")
                    predictions = self.ft_model.predict(cleaned_text, k=len(self.supported_langs))

                    # Extract language and confidence
                    lang_codes = [label.replace("__label__", "") for label in predictions[0]]
                    confidences = predictions[1]

                    logger.info(f"fastText predictions: {list(zip(lang_codes, confidences))}")

                    # Filter to only supported languages
                    for i, lang in enumerate(lang_codes):
                        mapped_lang = self._map_to_supported_language(lang)
                        logger.info(f"Mapped '{lang}' to '{mapped_lang}' with confidence {confidences[i]}")
                        if mapped_lang in self.supported_langs and confidences[i] >= self.confidence_threshold:
                            logger.info(f"Selected language: {mapped_lang} with confidence {confidences[i]}")
                            return mapped_lang, confidences[i]
                except Exception as e:
                    logger.error(f"fastText detection failed: {e}")
                    logger.info("Falling back to langid")

        # Fallback to langid
        try:
            logger.info("Falling back to langid for language detection")
            lang_code, confidence = langid.classify(cleaned_text)
            mapped_lang = self._map_to_supported_language(lang_code)

            logger.info(f"langid detected '{lang_code}' (mapped to '{mapped_lang}') with confidence {confidence}")

            if mapped_lang in self.supported_langs and confidence >= self.confidence_threshold:
                logger.info(f"Selected language: {mapped_lang} with confidence {confidence}")
                return mapped_lang, confidence
        except Exception as e:
            logger.error(f"langid detection failed: {e}")

        # Default to English if detection fails or confidence is low
        logger.info(f"Detection failed or confidence too low, defaulting to {self.default_language}")
        return self.default_language, 1.0

    @staticmethod
    def test_detection():
        """
        Test the language detection with various examples.

        This function demonstrates how the language detection works with
        examples in English, Hindi, Bengali, and Marathi.
        """
        detector = LanguageDetector()

        # Test examples
        examples = {
            "English": [
                "Hello, how are you today?",
                "What is your name?",
                "I would like to learn more about this."
            ],
            "Hindi": [
                "नमस्ते, आप कैसे हैं?",  # Hello, how are you?
                "आपका नाम क्या है?",  # What is your name?
                "मुझे इसके बारे में और जानना है।"  # I want to know more about this.
            ],
            "Bengali": [
                "হ্যালো, আপনি কেমন আছেন?",  # Hello, how are you?
                "আপনার নাম কি?",  # What is your name?
                "আমি এ সম্পর্কে আরও জানতে চাই।"  # I want to know more about this.
            ],
            "Marathi": [
                "नमस्कार, तुम्ही कसे आहात?",  # Hello, how are you?
                "तुझे नाव काय आहे?",  # What is your name?
                "मला याबद्दल अधिक जाणून घ्यायचे आहे."  # I want to know more about this.
            ],
            "Mixed": [
                "Hello नमस्ते",  # English + Hindi
                "What is your नाम?",  # English + Hindi
                "আমি want to learn मराठी"  # Bengali + English + Hindi
            ]
        }

        print("\n===== LANGUAGE DETECTION TEST =====\n")

        for language, sentences in examples.items():
            print(f"\n----- {language} Examples -----")
            for sentence in sentences:
                # Clean the text to avoid newline issues
                cleaned_text = sentence.replace('\n', ' ').strip()

                # Try direct script detection
                script_lang = detector._detect_script(cleaned_text)
                script_result = f"Script detection: {script_lang}" if script_lang else "Script detection: None"

                # Get full detection result
                detected_lang, confidence = detector.detect(cleaned_text)

                print(f"\nText: {cleaned_text}")
                print(script_result)
                print(f"Detected: {detected_lang} (confidence: {confidence:.2f})")

        print("\n===== TEST COMPLETE =====\n")

        return "Test completed successfully"

    def is_code_mixed(self, text: str) -> bool:
        """
        Check if the text contains multiple languages (code-mixed).

        Args:
            text: The text to check

        Returns:
            True if code-mixed, False otherwise
        """
        # Split text into sentences/chunks
        chunks = [s.strip() for s in text.split('.') if s.strip()]
        if not chunks:
            chunks = [text]

        # Detect language for each chunk
        languages = set()
        for chunk in chunks:
            lang, conf = self.detect(chunk)
            if conf >= self.confidence_threshold:
                languages.add(lang)

            # If we already found multiple languages, return early
            if len(languages) > 1:
                return True

        return len(languages) > 1
