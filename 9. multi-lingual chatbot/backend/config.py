"""
Configuration settings for the multilingual chatbot.
"""
import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Language settings
SUPPORTED_LANGUAGES = [
    {"code": "eng_Latn", "name": "English"},
    {"code": "hin_Deva", "name": "Hindi"},
    {"code": "ben_Beng", "name": "Bengali"},
    {"code": "mar_Deva", "name": "Marathi"},
]

# Language code mappings
LANGUAGE_CODE_MAP: Dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
}

# Reverse mapping for language detection
DETECT_TO_CODE_MAP: Dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
}

# Default language
DEFAULT_LANGUAGE = "eng_Latn"

# Translation model settings
TRANSLATION_MODEL = "ai4bharat/IndicBART"
TRANSLATION_DEVICE = "cuda" if os.getenv("USE_GPU", "False").lower() == "true" else "cpu"

# Language detection settings
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "models/lid.176.bin")

# Session settings
SESSION_EXPIRY_MINUTES = int(os.getenv("SESSION_EXPIRY_MINUTES", "60"))

# System prompts for different languages
SYSTEM_PROMPTS: Dict[str, str] = {
    "eng_Latn": "You are a helpful multilingual assistant that can communicate in English, Hindi, Bengali, and Marathi. Provide concise and accurate responses.",
    "hin_Deva": "आप एक सहायक बहुभाषी सहायक हैं जो अंग्रेजी, हिंदी, बंगाली और मराठी में संवाद कर सकते हैं। संक्षिप्त और सटीक प्रतिक्रियाएँ प्रदान करें।",
    "ben_Beng": "আপনি একজন সহায়ক বহুভাষিক সহকারী যিনি ইংরেজি, হিন্দি, বাংলা এবং মারাঠি ভাষায় যোগাযোগ করতে পারেন। সংক্ষিপ্ত এবং সঠিক প্রতিক্রিয়া প্রদান করুন।",
    "mar_Deva": "तुम्ही एक मदतगार बहुभाषिक सहाय्यक आहात जो इंग्रजी, हिंदी, बंगाली आणि मराठी मध्ये संवाद साधू शकता. संक्षिप्त आणि अचूक प्रतिसाद द्या.",
}

# Cultural context for each language
CULTURAL_CONTEXT: Dict[str, str] = {
    "eng_Latn": "Respond in a friendly, professional manner.",
    "hin_Deva": "हिंदी संस्कृति के संदर्भ में उत्तर दें, जैसे 'नमस्ते' का उपयोग अभिवादन के लिए करें।",
    "ben_Beng": "বাঙালি সংস্কৃতির প্রসঙ্গে উত্তর দিন, যেমন অভিবাদনের জন্য 'নমস্কার' ব্যবহার করুন।",
    "mar_Deva": "मराठी संस्कृतीच्या संदर्भात उत्तर द्या, जसे अभिवादनासाठी 'नमस्कार' वापरा.",
}
