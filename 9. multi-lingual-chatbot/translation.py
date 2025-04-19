"""
Translation module using NLLB-200 model.
"""

import os
import logging
import torch
from typing import List, Dict, Optional, Union
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import TRANSLATION, SUPPORTED_LANGUAGES
from cache import TranslationCache
from debug_utils import debug_checkpoint

logger = logging.getLogger(__name__)

class Translator:
    """
    Translation service using NLLB-200 model with optimizations for limited hardware.
    """

    def __init__(self):
        """Initialize the translator with the NLLB model."""
        self.model_name = TRANSLATION["model_name"]
        self.device = TRANSLATION["device"]
        self.max_length = TRANSLATION["max_length"]
        self.batch_size = TRANSLATION["batch_size"]

        # Check if CUDA is available when device is set to cuda
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        self.model = None
        self.tokenizer = None
        self.cache = TranslationCache()

        logger.info(f"Translator initialized with model {self.model_name} on {self.device}")

    def _load_model(self):
        """Load the translation model and tokenizer with optimizations."""
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info(f"Loading NLLB model {self.model_name}...")

        try:
            # Check if local model path is available
            local_model_path = TRANSLATION.get("local_model_path")

            # Use local path if available, otherwise use model_name
            model_path = local_model_path if local_model_path and os.path.exists(local_model_path) else self.model_name

            # Load tokenizer
            logger.info(f"Loading tokenizer from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=(model_path != self.model_name)  # Only use local files if using local path
            )
            logger.info("Tokenizer loaded successfully")

            # Load model with optimizations
            logger.info(f"Loading model from: {model_path}")
            if TRANSLATION["quantization"] and self.device == "cuda":
                # 8-bit quantization for GPU
                logger.info("Loading model with 8-bit quantization")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    load_in_8bit=True,
                    local_files_only=(model_path != self.model_name)  # Only use local files if using local path
                )
                logger.info("Loaded model with 8-bit quantization")
            else:
                # Standard loading
                logger.info(f"Loading model on {self.device}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    local_files_only=(model_path != self.model_name)  # Only use local files if using local path
                )
                self.model.to(self.device)
                logger.info(f"Loaded model on {self.device}")

            # Apply additional memory optimizations
            if self.device == "cuda":
                # Free up memory
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def translate(self,
                  texts: Union[str, List[str]],
                  source_lang: str,
                  target_lang: str) -> Union[str, List[str]]:
        """
        Translate text from source language to target language.

        Args:
            texts: Text or list of texts to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text or list of translated texts
        """
        logger.info(f"Translating from {source_lang} to {target_lang}")

        # Handle single text input
        is_single_text = isinstance(texts, str)
        if is_single_text:
            logger.info(f"Single text input: '{texts}'")
            texts = [texts]
        else:
            logger.info(f"Multiple texts input: {len(texts)} items")

        # Check if languages are supported
        if source_lang not in SUPPORTED_LANGUAGES:
            logger.error(f"Source language {source_lang} not supported")
            raise ValueError(f"Source language {source_lang} not supported")
        if target_lang not in SUPPORTED_LANGUAGES:
            logger.error(f"Target language {target_lang} not supported")
            raise ValueError(f"Target language {target_lang} not supported")

        # Skip translation if source and target languages are the same
        if source_lang == target_lang:
            logger.info(f"Source and target languages are the same ({source_lang}), skipping translation")
            return texts[0] if is_single_text else texts

        # Check cache for each text
        results = []
        texts_to_translate = []
        cache_indices = []

        for i, text in enumerate(texts):
            cached_translation = self.cache.get(text, source_lang, target_lang)
            if cached_translation is not None:
                logger.info(f"Cache hit for text: '{text}'")
                results.append(cached_translation)
            else:
                logger.info(f"Cache miss for text: '{text}'")
                results.append(None)  # Placeholder
                texts_to_translate.append(text)
                cache_indices.append(i)

        # If all texts were in cache, return results
        if not texts_to_translate:
            logger.info("All translations found in cache")
            return results[0] if is_single_text else results

        # Load model if not already loaded
        try:
            logger.info("Loading translation model")
            self._load_model()
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            # Return original texts if model loading fails
            return texts[0] if is_single_text else texts

        # Process in batches to save memory
        all_translations = []
        for i in range(0, len(texts_to_translate), self.batch_size):
            batch_texts = texts_to_translate[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1} with {len(batch_texts)} texts")

            try:
                # Tokenize
                logger.info("Tokenizing input texts")
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)

                # Translate
                logger.info(f"Generating translations to {target_lang}")
                with torch.no_grad():
                    translated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
                        max_length=self.max_length,
                    )

                # Decode
                logger.info("Decoding translated tokens")
                batch_translations = self.tokenizer.batch_decode(
                    translated_tokens,
                    skip_special_tokens=True
                )

                logger.info(f"Batch translations: {batch_translations}")
                all_translations.extend(batch_translations)
            except Exception as e:
                logger.error(f"Translation error: {e}")
                # Add original texts as fallback
                all_translations.extend(batch_texts)

        # Update cache and results
        for i, translation in enumerate(all_translations):
            original_idx = cache_indices[i]
            original_text = texts_to_translate[i]

            logger.info(f"Original: '{original_text}' → Translated: '{translation}'")

            # Add to cache
            self.cache.add(original_text, source_lang, target_lang, translation)

            # Update results
            results[original_idx] = translation

        # Free up memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        final_result = results[0] if is_single_text else results
        logger.info(f"Translation complete: {final_result}")
        return final_result

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """
        Translate text to English.

        Args:
            text: Text to translate
            source_lang: Source language code

        Returns:
            Translated text in English
        """
        debug_checkpoint("Starting translation to English", {
            "text": text,
            "source_lang": source_lang
        })

        try:
            result = self.translate(text, source_lang, "eng_Latn")
            debug_checkpoint("Successful translation to English", {"result": result})
            return result
        except Exception as e:
            logger.error(f"Translation to English failed: {e}")
            debug_checkpoint("Translation to English failed", {"error": str(e)})

            # For Hindi and Marathi, provide some basic translations for common phrases
            if source_lang == "hin_Deva":
                debug_checkpoint("Checking Hindi phrases", {"text": text})
                if "नमस्ते" in text or "नमस्कार" in text:
                    debug_checkpoint("Matched Hindi greeting", {"translation": "Hello"})
                    return "Hello"
                if "कैसे हैं" in text or "कैसे हो" in text:
                    debug_checkpoint("Matched Hindi 'how are you'", {"translation": "How are you"})
                    return "How are you"
                if "आप का नाम" in text or "तुम्हारा नाम" in text:
                    debug_checkpoint("Matched Hindi 'what is your name'", {"translation": "What is your name"})
                    return "What is your name"
            elif source_lang == "mar_Deva":
                debug_checkpoint("Checking Marathi phrases", {"text": text})
                if "नमस्कार" in text:
                    debug_checkpoint("Matched Marathi greeting", {"translation": "Hello"})
                    return "Hello"
                if "कसे आहात" in text or "कसे आहेस" in text:
                    debug_checkpoint("Matched Marathi 'how are you'", {"translation": "How are you"})
                    return "How are you"
                if "तुझे नाव" in text or "तुमचे नाव" in text:
                    debug_checkpoint("Matched Marathi 'what is your name'", {"translation": "What is your name"})
                    return "What is your name"
            elif source_lang == "ben_Beng":
                debug_checkpoint("Checking Bengali phrases", {"text": text})
                if "নমস্কার" in text:
                    debug_checkpoint("Matched Bengali greeting", {"translation": "Hello"})
                    return "Hello"
                if "কেমন আছেন" in text or "কেমন আছো" in text:
                    debug_checkpoint("Matched Bengali 'how are you'", {"translation": "How are you"})
                    return "How are you"
                if "আপনার নাম" in text or "তোমার নাম" in text:
                    debug_checkpoint("Matched Bengali 'what is your name'", {"translation": "What is your name"})
                    return "What is your name"

            # If no matches, return the original text
            debug_checkpoint("No phrase matches, returning original text", {"text": text})
            return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate text from English to target language.

        Args:
            text: Text in English to translate
            target_lang: Target language code

        Returns:
            Translated text in target language
        """
        debug_checkpoint("Starting translation from English", {
            "text": text,
            "target_lang": target_lang
        })

        try:
            result = self.translate(text, "eng_Latn", target_lang)
            debug_checkpoint("Successful translation from English", {"result": result})
            return result
        except Exception as e:
            logger.error(f"Translation from English failed: {e}")
            debug_checkpoint("Translation from English failed", {"error": str(e)})

            # For Hindi and Marathi, provide some basic translations for common phrases
            if target_lang == "hin_Deva":
                debug_checkpoint("Checking English phrases for Hindi translation", {"text": text.lower()})
                if "hello" in text.lower() or "hi" in text.lower():
                    debug_checkpoint("Matched English greeting for Hindi", {"translation": "नमस्ते"})
                    return "नमस्ते"
                if "how are you" in text.lower():
                    debug_checkpoint("Matched 'how are you' for Hindi", {"translation": "आप कैसे हैं"})
                    return "आप कैसे हैं"
                if "what is your name" in text.lower():
                    debug_checkpoint("Matched 'what is your name' for Hindi", {"translation": "आपका नाम क्या है"})
                    return "आपका नाम क्या है"
                if "sorry" in text.lower() or "understand" in text.lower():
                    debug_checkpoint("Matched 'sorry/understand' for Hindi", {"translation": "मुझे माफ़ करें..."})
                    return "मुझे माफ़ करें, मैं समझ नहीं पाया। कृपया दोबारा कहें।"
            elif target_lang == "mar_Deva":
                debug_checkpoint("Checking English phrases for Marathi translation", {"text": text.lower()})
                if "hello" in text.lower() or "hi" in text.lower():
                    debug_checkpoint("Matched English greeting for Marathi", {"translation": "नमस्कार"})
                    return "नमस्कार"
                if "how are you" in text.lower():
                    debug_checkpoint("Matched 'how are you' for Marathi", {"translation": "तुम्ही कसे आहात"})
                    return "तुम्ही कसे आहात"
                if "what is your name" in text.lower():
                    debug_checkpoint("Matched 'what is your name' for Marathi", {"translation": "तुझे नाव काय आहे"})
                    return "तुझे नाव काय आहे"
                if "sorry" in text.lower() or "understand" in text.lower():
                    debug_checkpoint("Matched 'sorry/understand' for Marathi", {"translation": "मला माफ करा..."})
                    return "मला माफ करा, मला समजले नाही. कृपया पुन्हा सांगा."
            elif target_lang == "ben_Beng":
                debug_checkpoint("Checking English phrases for Bengali translation", {"text": text.lower()})
                if "hello" in text.lower() or "hi" in text.lower():
                    debug_checkpoint("Matched English greeting for Bengali", {"translation": "নমস্কার"})
                    return "নমস্কার"
                if "how are you" in text.lower():
                    debug_checkpoint("Matched 'how are you' for Bengali", {"translation": "আপনি কেমন আছেন"})
                    return "আপনি কেমন আছেন"
                if "what is your name" in text.lower():
                    debug_checkpoint("Matched 'what is your name' for Bengali", {"translation": "আপনার নাম কি"})
                    return "আপনার নাম কি"
                if "sorry" in text.lower() or "understand" in text.lower():
                    debug_checkpoint("Matched 'sorry/understand' for Bengali", {"translation": "দুঃখিত..."})
                    return "দুঃখিত, আমি বুঝতে পারিনি। অনুগ্রহ করে আবার বলুন।"

            # If no matches, return the original text
            debug_checkpoint("No phrase matches for translation, returning original text", {"text": text})
            return text
