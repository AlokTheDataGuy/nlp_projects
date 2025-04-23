import spacy
import re
from typing import List, Dict, Any

class EntityRecognizer:
    """
    Recognizes medical entities in text using spaCy and custom rules
    """
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model is not installed, download it
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Add custom medical entity patterns
        self.medical_entities = {
            "diseases": [
                "cancer", "diabetes", "hypertension", "asthma", "arthritis", 
                "alzheimer", "parkinson", "depression", "anxiety", "schizophrenia",
                "pneumonia", "bronchitis", "influenza", "covid", "hepatitis",
                "cirrhosis", "nephritis", "leukemia", "anemia", "hemophilia"
            ],
            "symptoms": [
                "pain", "fever", "cough", "headache", "nausea", "vomiting", 
                "diarrhea", "fatigue", "rash", "swelling", "dizziness", "weakness",
                "shortness of breath", "chest pain", "back pain", "sore throat",
                "runny nose", "congestion", "itching", "burning"
            ],
            "body_parts": [
                "head", "neck", "chest", "abdomen", "back", "arm", "leg", "foot",
                "hand", "eye", "ear", "nose", "throat", "heart", "lung", "liver",
                "kidney", "brain", "stomach", "intestine", "skin", "muscle", "bone"
            ],
            "treatments": [
                "surgery", "medication", "therapy", "antibiotic", "vaccine",
                "chemotherapy", "radiation", "transplant", "dialysis", "injection",
                "pill", "tablet", "capsule", "ointment", "cream", "inhaler"
            ]
        }
        
        # Compile regex patterns for each entity type
        self.entity_patterns = {}
        for entity_type, terms in self.medical_entities.items():
            pattern = r'\b(' + '|'.join(terms) + r')s?\b'
            self.entity_patterns[entity_type] = re.compile(pattern, re.IGNORECASE)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text using spaCy and custom rules
        """
        entities = []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract named entities from spaCy
        for ent in doc.ents:
            if ent.label_ in ["DISEASE", "ORG", "PRODUCT", "GPE", "PERSON"]:
                entities.append({
                    "text": ent.text,
                    "type": "disease" if ent.label_ == "DISEASE" else "other",
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy"
                })
        
        # Apply custom regex patterns
        for entity_type, pattern in self.entity_patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    "text": match.group(),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "custom"
                })
        
        # Remove duplicates
        unique_entities = []
        seen_spans = set()
        
        for entity in sorted(entities, key=lambda x: (x["start"], -x["end"])):
            span = (entity["start"], entity["end"])
            if span not in seen_spans:
                unique_entities.append(entity)
                seen_spans.add(span)
        
        return unique_entities
    
    def get_entity_descriptions(self, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Get descriptions for recognized entities (placeholder for future integration)
        """
        # This would connect to a medical knowledge base in a real implementation
        descriptions = {}
        
        for entity in entities:
            entity_text = entity["text"].lower()
            entity_type = entity["type"]
            
            # Placeholder descriptions
            if entity_type == "diseases":
                descriptions[entity_text] = f"{entity_text.capitalize()} is a medical condition that affects health."
            elif entity_type == "symptoms":
                descriptions[entity_text] = f"{entity_text.capitalize()} is a symptom that may indicate various conditions."
            elif entity_type == "body_parts":
                descriptions[entity_text] = f"{entity_text.capitalize()} is a part of the human body."
            elif entity_type == "treatments":
                descriptions[entity_text] = f"{entity_text.capitalize()} is a medical treatment or intervention."
        
        return descriptions
