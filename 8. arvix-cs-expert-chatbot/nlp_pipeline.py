import nltk
import spacy
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from collections import Counter, defaultdict
import yaml
from gensim import corpora, models
import networkx as nx

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

class AdvancedNLPPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.nlp_config = self.config['nlp']
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.setup_advanced_pipelines()
        
    def setup_advanced_pipelines(self):
        """Setup advanced NLP pipelines"""
        try:
            # Enhanced summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.nlp_config['summarization_model'],
                max_length=200,
                min_length=50,
                do_sample=False,
                truncation=True
            )
            
            # Question answering pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.nlp_config['qa_model'],
                return_multiple_answers=True
            )
            
            print("Advanced NLP pipelines loaded successfully")
            
        except Exception as e:
            print(f"Error loading advanced NLP pipelines: {e}")
            self.summarizer = None
            self.qa_pipeline = None
    
    def extract_enhanced_entities(self, text):
        """
        Extract enhanced named entities with CS-specific recognition
        """
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Add CS-specific entity recognition
        cs_entities = self.extract_cs_entities(text)
        entities.extend(cs_entities)
        
        return entities
    
    def extract_cs_entities(self, text):
        """Extract CS-specific entities using pattern matching"""
        cs_patterns = {
            'ALGORITHM': [
                r'\b(?:algorithm|method|approach|technique)\b.*?(?:for|to)\b.*?\b(?:learning|training|optimization)\b',
                r'\b(?:CNN|RNN|LSTM|GRU|BERT|GPT|Transformer)\b',
                r'\b(?:gradient descent|backpropagation|attention mechanism)\b'
            ],
            'METRIC': [
                r'\b(?:accuracy|precision|recall|F1|BLEU|ROUGE|perplexity)\b.*?(?:score|measure|metric)\b',
                r'\b(?:\d+(?:\.\d+)?%?\s*(?:accuracy|precision|recall))\b'
            ],
            'DATASET': [
                r'\b(?:ImageNet|COCO|WMT|GLUE|SuperGLUE|SQuAD|CoNLL)\b',
                r'\bdataset\b.*?\b(?:containing|with|of)\b.*?\b(?:samples|examples|instances)\b'
            ],
            'MODEL': [
                r'\b(?:ResNet|VGG|BERT|GPT|T5|BART|RoBERTa|XLNet)\b',
                r'\b(?:neural network|deep learning model|transformer model)\b'
            ]
        }
        
        entities = []
        for entity_type, patterns in cs_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': entity_type,
                        'description': f'Computer Science {entity_type}',
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return entities
    
    def extract_advanced_keywords(self, text, num_keywords=15):
        """
        Extract keywords using multiple techniques
        """
        # Clean text
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Method 1: TF-IDF based keywords
        tfidf_keywords = self.extract_tfidf_keywords(text, num_keywords//3)
        
        # Method 2: POS-based keywords (nouns and adjectives)
        pos_keywords = self.extract_pos_keywords(text, num_keywords//3)
        
        # Method 3: CS domain-specific keywords
        domain_keywords = self.extract_domain_keywords(text, num_keywords//3)
        
        # Combine and deduplicate
        all_keywords = list(set(tfidf_keywords + pos_keywords + domain_keywords))
        
        return all_keywords[:num_keywords]
    
    def extract_tfidf_keywords(self, text, num_keywords):
        """Extract keywords using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=num_keywords * 2,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = np.argsort(scores)[-num_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return keywords
        except:
            return []
    
    def extract_pos_keywords(self, text, num_keywords):
        """Extract keywords based on POS tags"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        keywords = []
        
        # Focus on nouns, proper nouns, and adjectives
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Count frequency and return top keywords
        freq_dist = Counter(keywords)
        return [word for word, freq in freq_dist.most_common(num_keywords)]
    
    def extract_domain_keywords(self, text, num_keywords):
        """Extract CS domain-specific keywords"""
        cs_terms = [
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
            'natural language processing', 'computer vision', 'data mining', 'algorithm',
            'classification', 'regression', 'clustering', 'optimization', 'training',
            'validation', 'testing', 'accuracy', 'precision', 'recall', 'transformer',
            'attention', 'convolution', 'embedding', 'feature', 'model', 'dataset',
            'performance', 'evaluation', 'benchmark', 'state-of-the-art'
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in cs_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms[:num_keywords]
    
    def generate_advanced_summary(self, text, max_length=200, min_length=50):
        """
        Generate advanced abstractive summary
        """
        if not self.summarizer:
            return self.extractive_summary(text, num_sentences=3)
        
        try:
            # Clean and prepare text
            cleaned_text = self.clean_text_for_summarization(text)
            
            if len(cleaned_text.split()) < min_length:
                return cleaned_text
            
            # Generate abstractive summary
            summary_result = self.summarizer(
                cleaned_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            
            summary = summary_result[0]['summary_text']
            
            # Post-process summary
            summary = self.post_process_summary(summary)
            
            return summary
            
        except Exception as e:
            print(f"Error in advanced summarization: {e}")
            return self.extractive_summary(text, num_sentences=3)
    
    def clean_text_for_summarization(self, text):
        """Clean text for better summarization"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short sentences (likely incomplete)
        sentences = text.split('.')
        cleaned_sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
        
        return '. '.join(cleaned_sentences)
    
    def post_process_summary(self, summary):
        """Post-process generated summary"""
        # Ensure proper capitalization
        summary = summary.strip()
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure proper ending
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def extractive_summary(self, text, num_sentences=3):
        """
        Enhanced extractive summarization
        """
        if not self.nlp:
            # Fallback to simple sentence ranking
            sentences = text.split('.')
            if len(sentences) <= num_sentences:
                return text
            
            # Score sentences by length and position
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) > 10:
                    # Simple scoring: length + position bias (earlier sentences get higher score)
                    score = len(sentence.split()) + (len(sentences) - i) * 0.1
                    scored_sentences.append((sentence, score))
            
            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent[0] for sent in scored_sentences[:num_sentences]]
            
            return '. '.join(top_sentences) + '.'
        
        # Advanced extractive summarization using spaCy
        doc = self.nlp(text)
        sentences = [sent for sent in doc.sents]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Score sentences based on multiple criteria
        sentence_scores = []
        
        # Calculate word frequencies
        word_freq = defaultdict(int)
        for token in doc:
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'ADJ', 'VERB']:
                word_freq[token.lemma_.lower()] += 1
        
        # Score each sentence
        for i, sent in enumerate(sentences):
            score = 0
            word_count = 0
            
            for token in sent:
                if not token.is_stop and not token.is_punct:
                    score += word_freq[token.lemma_.lower()]
                    word_count += 1
            
            if word_count > 0:
                score = score / word_count  # Normalize by sentence length
                
                # Boost score for sentences with CS keywords
                sent_text = sent.text.lower()
                cs_bonus = sum(1 for term in ['learning', 'model', 'algorithm', 'method', 'approach'] 
                              if term in sent_text)
                score += cs_bonus * 0.5
                
                # Position bias (slight preference for earlier sentences)
                position_score = 1 - (i / len(sentences)) * 0.2
                score *= position_score
            
            sentence_scores.append((sent.text.strip(), score))
        
        # Sort by score and return top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent[0] for sent in sentence_scores[:num_sentences]]
        
        return ' '.join(top_sentences)
    
    def perform_advanced_qa(self, context, question, max_answers=3):
        """
        Perform advanced question answering
        """
        if not self.qa_pipeline:
            return self.simple_qa(context, question)
        
        try:
            results = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=200,
                top_k=max_answers
            )
            
            # Process results
            processed_results = []
            for result in results:
                if result['score'] > 0.1:  # Filter low-confidence answers
                    processed_results.append({
                        'answer': result['answer'],
                        'confidence': result['score'],
                        'start': result['start'],
                        'end': result['end']
                    })
            
            return processed_results
            
        except Exception as e:
            print(f"Error in advanced QA: {e}")
            return self.simple_qa(context, question)
    
    def simple_qa(self, context, question):
        """Simple fallback QA using keyword matching"""
        question_words = set(question.lower().split())
        
        # Remove stop words from question
        question_keywords = [word for word in question_words if word not in self.stop_words]
        
        sentences = context.split('.')
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                sentence_words = set(sentence.lower().split())
                
                # Count keyword matches
                matches = len(set(question_keywords) & sentence_words)
                if matches > 0:
                    scored_sentences.append((sentence, matches))
        
        if scored_sentences:
            # Sort by match count and return best sentence
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            return [{'answer': scored_sentences[0][0], 'confidence': 0.7}]
        
        return [{'answer': "I couldn't find a specific answer in the context.", 'confidence': 0.1}]
    
    def analyze_text_complexity(self, text):
        """
        Analyze the complexity of text
        """
        analysis = {}
        
        # Basic metrics
        sentences = text.split('.')
        words = text.split()
        
        analysis['sentence_count'] = len(sentences)
        analysis['word_count'] = len(words)
        analysis['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Lexical diversity
        unique_words = set(word.lower() for word in words if word.isalpha())
        analysis['lexical_diversity'] = len(unique_words) / len(words) if words else 0
        
        # Reading level (simple approximation)
        avg_sentence_length = analysis['avg_sentence_length']
        if avg_sentence_length < 15:
            analysis['reading_level'] = 'Easy'
        elif avg_sentence_length < 25:
            analysis['reading_level'] = 'Medium'
        else:
            analysis['reading_level'] = 'Complex'
        
        # Technical term density
        technical_terms = ['algorithm', 'model', 'method', 'approach', 'technique', 
                          'analysis', 'evaluation', 'performance', 'optimization']
        tech_count = sum(1 for word in words if word.lower() in technical_terms)
        analysis['technical_density'] = tech_count / len(words) if words else 0
        
        return analysis
    
    def extract_concept_graph(self, texts, min_cooccurrence=2):
        """
        Extract concept co-occurrence graph from texts
        """
        # Extract concepts from all texts
        all_concepts = []
        
        for text in texts:
            keywords = self.extract_advanced_keywords(text, num_keywords=10)
            entities = self.extract_enhanced_entities(text)
            
            # Combine keywords and entities
            concepts = keywords + [ent['text'].lower() for ent in entities 
                                 if ent['label'] in ['PERSON', 'ORG', 'ALGORITHM', 'MODEL']]
            all_concepts.append(set(concepts))
        
        # Build co-occurrence matrix
        concept_counts = defaultdict(int)
        cooccurrence_counts = defaultdict(int)
        
        for concepts in all_concepts:
            for concept in concepts:
                concept_counts[concept] += 1
            
            # Count co-occurrences
            concepts_list = list(concepts)
            for i in range(len(concepts_list)):
                for j in range(i + 1, len(concepts_list)):
                    pair = tuple(sorted([concepts_list[i], concepts_list[j]]))
                    cooccurrence_counts[pair] += 1
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (concepts that appear frequently enough)
        for concept, count in concept_counts.items():
            if count >= 2:  # Minimum frequency threshold
                G.add_node(concept, weight=count)
        
        # Add edges (co-occurrences above threshold)
        for (concept1, concept2), count in cooccurrence_counts.items():
            if count >= min_cooccurrence and concept1 in G.nodes and concept2 in G.nodes:
                G.add_edge(concept1, concept2, weight=count)
        
        return G
