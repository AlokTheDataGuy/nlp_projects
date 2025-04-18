from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import os
from typing import Dict, Any, List

class SentimentAnalyzer:
    def __init__(self):
        # Load model and tokenizer
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # Define sentiment labels
        self.labels = ['negative', 'neutral', 'positive']

    def preprocess(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Basic preprocessing
        text = text.lower()
        text = text.replace('\n', ' ')
        return text

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the given text"""
        # Preprocess text
        preprocessed_text = self.preprocess(text)

        # Check if text is a short greeting
        if self._is_greeting(preprocessed_text):
            # For greetings, default to neutral sentiment
            result = {
                "label": "neutral",
                "scores": {
                    "negative": 0.1,
                    "neutral": 0.8,
                    "positive": 0.1
                },
                "confidence": 0.8
            }
            return result

        # For very short texts (less than 5 words), bias toward neutral
        if len(preprocessed_text.split()) < 5:
            # Proceed with analysis but with caution
            encoded_input = self.tokenizer(preprocessed_text, return_tensors='pt')
            output = self.model(**encoded_input)
            scores = output.logits[0].detach().numpy()
            scores = softmax(scores)

            # Bias toward neutral for very short texts
            scores[1] = scores[1] * 1.2  # Boost neutral score
            scores = scores / np.sum(scores)  # Renormalize

            result = {
                "label": self.labels[np.argmax(scores)],
                "scores": {
                    label: float(score) for label, score in zip(self.labels, scores)
                },
                "confidence": float(np.max(scores))
            }
            return result

        # For normal length texts, proceed with standard analysis
        encoded_input = self.tokenizer(preprocessed_text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)

        # Format results
        result = {
            "label": self.labels[np.argmax(scores)],
            "scores": {
                label: float(score) for label, score in zip(self.labels, scores)
            },
            "confidence": float(np.max(scores))
        }

        return result

    def _is_greeting(self, text: str) -> bool:
        """Check if the text is a simple greeting"""
        common_greetings = [
            'hi', 'hello', 'hey', 'hii', 'hiii', 'hiiii', 'hiiiii',
            'hiya', 'howdy', 'greetings', 'sup', 'whats up', "what's up",
            'yo', 'good morning', 'good afternoon', 'good evening',
            'morning', 'afternoon', 'evening', 'hola', 'namaste'
        ]

        # Clean the text for comparison
        text = text.strip().lower()

        # Check if the text is just a greeting
        if text in common_greetings:
            return True

        # Check if the text starts with a greeting and is short
        if len(text.split()) <= 3:
            for greeting in common_greetings:
                if text.startswith(greeting):
                    return True

        return False

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts"""
        return [self.analyze(text) for text in texts]
