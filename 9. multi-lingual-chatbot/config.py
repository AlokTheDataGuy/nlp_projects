"""
Configuration settings for the multilingual chatbot.
"""

# Supported languages
SUPPORTED_LANGUAGES = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "mar_Deva": "Marathi"
}

# Language detection settings
LANGUAGE_DETECTION = {
    "model_path": "models/lid.176.ftz",  # Path to fastText model
    "confidence_threshold": 0.7,
    "default_language": "eng_Latn"
}

# Translation model settings
TRANSLATION = {
    "model_name": "facebook/nllb-200-distilled-600M",
    "local_model_path": "./nllb-200-distilled-600M",  # Path to local model files
    "device": "cpu",  # Using CPU due to CUDA issues
    "quantization": False,  # Disable quantization for CPU
    "max_length": 128,
    "batch_size": 4  # Reduced batch size for CPU
}

# Cache settings
CACHE = {
    "enabled": True,
    "max_size": 1000,  # Maximum number of entries in cache
    "ttl": 3600  # Time to live in seconds
}

# Cultural adaptation settings
CULTURAL_ADAPTATION = {
    "honorifics": {
        "hin_Deva": ["आप", "तुम", "तू"],
        "ben_Beng": ["আপনি", "তুমি", "তুই"],
        "mar_Deva": ["आपण", "तुम्ही", "तू"]
    },
    "greetings": {
        "eng_Latn": ["Hello", "Hi", "Good morning", "Good afternoon", "Good evening"],
        "hin_Deva": ["नमस्ते", "नमस्कार", "सुप्रभात", "शुभ दोपहर", "शुभ संध्या"],
        "ben_Beng": ["নমস্কার", "হ্যালো", "শুভ সকাল", "শুভ দুপুর", "শুভ সন্ধ্যা"],
        "mar_Deva": ["नमस्कार", "नमस्ते", "सुप्रभात", "शुभ दुपार", "शुभ संध्याकाळ"]
    }
}

# Chatbot settings
CHATBOT = {
    "max_context_length": 10,  # Maximum number of conversation turns to remember
    "default_response": "I'm sorry, I didn't understand that. Could you please rephrase?",
    "temperature": 0.7
}

# API settings
API = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True
}
