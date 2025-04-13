"""
Text Analysis Utilities

Provides functions for analyzing text quality, readability, and other metrics.
"""

import re
import math
from typing import Dict, Any, List, Tuple


def count_words(text: str) -> int:
    """Count the number of words in a text."""
    return len(re.findall(r'\b\w+\b', text))


def count_sentences(text: str) -> int:
    """Count the number of sentences in a text."""
    # Simple sentence counting based on punctuation
    return len(re.findall(r'[.!?]+', text)) or 1  # At least 1 sentence


def count_paragraphs(text: str) -> int:
    """Count the number of paragraphs in a text."""
    # Count non-empty lines as paragraphs
    paragraphs = [p for p in text.split('\n') if p.strip()]
    return len(paragraphs) or 1  # At least 1 paragraph


def has_title(text: str) -> bool:
    """Check if the text has a title."""
    # Look for patterns like "Title:" or a line ending with a newline at the beginning
    return bool(re.match(r'^.*(?:title|headline).*:.*\n', text.lower(), re.IGNORECASE)) or \
           bool(re.match(r'^#+ .*\n', text)) or \
           bool(re.match(r'^[A-Z].*\n\n', text))


def has_sections(text: str) -> bool:
    """Check if the text has sections."""
    # Look for patterns like "Section:" or markdown headings
    return bool(re.search(r'\n\s*(?:##+ |[A-Z][^a-z\n]+:)\s*', text))


def flesch_kincaid_grade(text: str) -> float:
    """Calculate the Flesch-Kincaid Grade Level for the text."""
    word_count = count_words(text)
    sentence_count = count_sentences(text)
    
    if word_count == 0 or sentence_count == 0:
        return 0.0
    
    # Count syllables (simplified approach)
    syllable_count = 0
    for word in re.findall(r'\b\w+\b', text.lower()):
        word = word.strip(".,;:!?")
        if not word:
            continue
        # Count vowel groups as syllables
        syllables = len(re.findall(r'[aeiouy]+', word))
        # Words with no vowels get 1 syllable
        syllable_count += max(1, syllables)
    
    # Calculate Flesch-Kincaid Grade Level
    return 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59


def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text and return various metrics."""
    word_count = count_words(text)
    sentence_count = count_sentences(text)
    paragraph_count = count_paragraphs(text)
    
    # Calculate words per sentence
    words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    
    # Calculate readability
    fk_grade = flesch_kincaid_grade(text)
    
    # Check for structure
    has_title_bool = has_title(text)
    has_sections_bool = has_sections(text)
    
    # Calculate structure score (0-100)
    structure_score = 0
    if has_title_bool:
        structure_score += 30
    if has_sections_bool:
        structure_score += 30
    if paragraph_count >= 3:
        structure_score += 20
    if words_per_sentence >= 10 and words_per_sentence <= 25:
        structure_score += 20
    
    # Calculate overall quality score (0-100)
    quality_score = min(100, (
        structure_score * 0.5 +
        min(100, word_count / 10) * 0.3 +
        (100 - min(100, abs(fk_grade - 10) * 5)) * 0.2
    ))
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "words_per_sentence": round(words_per_sentence, 2),
        "readability_grade": round(fk_grade, 2),
        "has_title": has_title_bool,
        "has_sections": has_sections_bool,
        "structure_score": round(structure_score, 2),
        "quality_score": round(quality_score, 2),
    }


def compare_articles(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare multiple articles and rank them."""
    if not articles:
        return {"error": "No articles to compare"}
    
    # Calculate scores for each article
    for article in articles:
        if "article" not in article or not article["article"]:
            article["analysis"] = {"error": "No article content"}
            article["score"] = 0
            continue
        
        analysis = analyze_text(article["article"])
        article["analysis"] = analysis
        article["score"] = analysis["quality_score"]
    
    # Sort articles by score
    sorted_articles = sorted(articles, key=lambda x: x.get("score", 0), reverse=True)
    
    # Create rankings
    rankings = []
    for i, article in enumerate(sorted_articles):
        rankings.append({
            "rank": i + 1,
            "model": article.get("model", "unknown"),
            "topic": article.get("topic", "unknown"),
            "score": article.get("score", 0),
            "generation_time": article.get("generation_time", 0),
            "tokens": article.get("tokens", 0),
        })
    
    return {
        "rankings": rankings,
        "best_model": rankings[0]["model"] if rankings else None,
        "score_range": (
            min(a.get("score", 0) for a in articles),
            max(a.get("score", 0) for a in articles)
        ) if articles else (0, 0),
    }
