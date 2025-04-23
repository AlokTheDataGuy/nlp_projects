import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any

class QueryProcessor:
    """
    Processes user queries for better retrieval performance
    """
    
    def __init__(self):
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        # Add medical stopwords that shouldn't be removed
        self.medical_terms = {
            "pain", "fever", "cough", "headache", "nausea", "vomiting", 
            "diarrhea", "fatigue", "rash", "swelling", "blood", "heart", 
            "lung", "kidney", "liver", "brain", "cancer", "diabetes", 
            "hypertension", "asthma", "arthritis", "allergy"
        }
        # Remove medical terms from stopwords
        self.stop_words = self.stop_words - self.medical_terms
        
        # Common medical abbreviations and their expansions
        self.medical_abbreviations = {
            "dr": "doctor",
            "bp": "blood pressure",
            "hr": "heart rate",
            "temp": "temperature",
            "rx": "prescription",
            "dx": "diagnosis",
            "tx": "treatment",
            "hx": "history",
            "fx": "fracture",
            "sx": "symptoms",
            "labs": "laboratory tests",
            "meds": "medications",
            "uti": "urinary tract infection",
            "uri": "upper respiratory infection",
            "copd": "chronic obstructive pulmonary disease",
            "chf": "congestive heart failure",
            "mi": "myocardial infarction",
            "cva": "cerebrovascular accident",
            "htn": "hypertension",
            "dm": "diabetes mellitus"
        }
    
    def process_query(self, query: str) -> str:
        """
        Process the query by:
        1. Converting to lowercase
        2. Expanding medical abbreviations
        3. Removing punctuation
        4. Removing stopwords (except medical terms)
        5. Normalizing whitespace
        """
        # Convert to lowercase
        query = query.lower()
        
        # Expand medical abbreviations
        tokens = word_tokenize(query)
        expanded_tokens = []
        for token in tokens:
            if token in self.medical_abbreviations:
                expanded_tokens.append(self.medical_abbreviations[token])
            else:
                expanded_tokens.append(token)
        
        query = " ".join(expanded_tokens)
        
        # Remove punctuation
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Remove stopwords
        tokens = word_tokenize(query)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Normalize whitespace
        processed_query = " ".join(filtered_tokens)
        processed_query = re.sub(r'\s+', ' ', processed_query).strip()
        
        return processed_query
    
    def extract_question_type(self, query: str) -> str:
        """
        Attempt to classify the question type based on patterns
        """
        query_lower = query.lower()
        
        if re.search(r'what is|what are|define|meaning of', query_lower):
            return "information"
        elif re.search(r'symptoms|signs|feel like|experiencing', query_lower):
            return "symptoms"
        elif re.search(r'cause|reason|why do|why does', query_lower):
            return "causes"
        elif re.search(r'treat|cure|therapy|medication|drug|medicine', query_lower):
            return "treatment"
        elif re.search(r'diagnose|test|exam|check|detect', query_lower):
            return "exams and tests"
        elif re.search(r'prevent|avoid|reduce risk|lower chance', query_lower):
            return "prevention"
        elif re.search(r'outlook|prognosis|chance|survival|recover', query_lower):
            return "outlook"
        elif re.search(r'risk|chance of getting|susceptible|prone to', query_lower):
            return "susceptibility"
        else:
            return "general"
