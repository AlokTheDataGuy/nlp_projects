"""
CS Domain Knowledge Base
"""

class CSKnowledgeBase:
    def __init__(self):
        self.concepts = {
            # Core CS Concepts
            'artificial intelligence': {
                'definition': 'A branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.',
                'subconcepts': ['machine learning', 'deep learning', 'natural language processing', 'computer vision'],
                'applications': ['chatbots', 'recommendation systems', 'autonomous vehicles', 'medical diagnosis'],
                'keywords': ['AI', 'intelligent systems', 'cognitive computing', 'expert systems']
            },
            'machine learning': {
                'definition': 'A subset of AI that enables computers to learn and improve from experience without being explicitly programmed.',
                'subconcepts': ['supervised learning', 'unsupervised learning', 'reinforcement learning', 'deep learning'],
                'applications': ['predictive analytics', 'image recognition', 'fraud detection', 'personalization'],
                'keywords': ['ML', 'algorithms', 'training data', 'model', 'prediction']
            },
            'deep learning': {
                'definition': 'A subset of machine learning using artificial neural networks with multiple layers to model complex patterns.',
                'subconcepts': ['neural networks', 'convolutional networks', 'transformers', 'attention mechanisms'],
                'applications': ['image classification', 'natural language processing', 'speech recognition', 'autonomous systems'],
                'keywords': ['neural networks', 'backpropagation', 'gradient descent', 'layers']
            },
            'natural language processing': {
                'definition': 'A branch of AI that helps computers understand, interpret, and manipulate human language.',
                'subconcepts': ['text classification', 'sentiment analysis', 'machine translation', 'question answering'],
                'applications': ['chatbots', 'language translation', 'text summarization', 'voice assistants'],
                'keywords': ['NLP', 'text processing', 'linguistics', 'language models', 'tokenization']
            },
            'transformers': {
                'definition': 'A neural network architecture that uses attention mechanisms to process sequential data, revolutionizing NLP.',
                'subconcepts': ['attention mechanism', 'self-attention', 'encoder-decoder', 'positional encoding'],
                'applications': ['language translation', 'text generation', 'question answering', 'text summarization'],
                'keywords': ['attention', 'BERT', 'GPT', 'sequence-to-sequence', 'parallel processing']
            },
            'computer vision': {
                'definition': 'A field of AI that enables computers to interpret and understand visual information from images and videos.',
                'subconcepts': ['image classification', 'object detection', 'semantic segmentation', 'facial recognition'],
                'applications': ['medical imaging', 'autonomous vehicles', 'surveillance systems', 'augmented reality'],
                'keywords': ['image processing', 'CNN', 'feature extraction', 'pattern recognition']
            },
            'reinforcement learning': {
                'definition': 'A type of machine learning where an agent learns optimal actions through trial-and-error interactions with an environment.',
                'subconcepts': ['policy learning', 'value functions', 'Q-learning', 'actor-critic methods'],
                'applications': ['game playing', 'robotics', 'autonomous systems', 'recommendation systems'],
                'keywords': ['agent', 'environment', 'reward', 'policy', 'exploration vs exploitation']
            },
            'neural networks': {
                'definition': 'Computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.',
                'subconcepts': ['perceptron', 'multilayer perceptron', 'backpropagation', 'activation functions'],
                'applications': ['pattern recognition', 'function approximation', 'classification', 'regression'],
                'keywords': ['neurons', 'weights', 'biases', 'activation', 'layers']
            },
            'data structures': {
                'definition': 'Ways of organizing and storing data in computers to enable efficient access and modification.',
                'subconcepts': ['arrays', 'linked lists', 'trees', 'graphs', 'hash tables'],
                'applications': ['database management', 'algorithm optimization', 'memory management', 'search algorithms'],
                'keywords': ['efficiency', 'time complexity', 'space complexity', 'operations']
            },
            'algorithms': {
                'definition': 'Step-by-step procedures or formulas for solving computational problems.',
                'subconcepts': ['sorting algorithms', 'search algorithms', 'graph algorithms', 'dynamic programming'],
                'applications': ['problem solving', 'optimization', 'data processing', 'system design'],
                'keywords': ['complexity analysis', 'efficiency', 'correctness', 'implementation']
            }
        }
        
        self.recent_topics = {
            'large language models': 'Advanced AI models trained on vast amounts of text data to understand and generate human-like text.',
            'generative ai': 'AI systems capable of creating new content including text, images, code, and other media.',
            'prompt engineering': 'The practice of designing effective prompts to guide AI models toward desired outputs.',
            'retrieval augmented generation': 'A technique that combines pre-trained language models with external knowledge retrieval.',
            'multimodal ai': 'AI systems that can process and understand multiple types of data simultaneously.',
            'federated learning': 'A distributed machine learning approach that trains algorithms across decentralized data.',
            'explainable ai': 'Methods and techniques to make AI decision-making processes transparent and interpretable.',
            'edge computing': 'Computing performed at or near the data source rather than in centralized cloud servers.'
        }
    
    def get_concept_info(self, concept):
        """Get detailed information about a concept"""
        concept_lower = concept.lower()
        if concept_lower in self.concepts:
            return self.concepts[concept_lower]
        elif concept_lower in self.recent_topics:
            return {'definition': self.recent_topics[concept_lower]}
        return None
    
    def find_related_concepts(self, concept):
        """Find concepts related to the given concept"""
        concept_lower = concept.lower()
        related = []
        
        for key, value in self.concepts.items():
            if concept_lower in value.get('subconcepts', []) or concept_lower in value.get('keywords', []):
                related.append(key)
            elif concept_lower == key:
                related.extend(value.get('subconcepts', []))
        
        return related[:5]  # Limit to 5 related concepts
    
    def search_concepts(self, query):
        """Search for concepts matching the query"""
        query_lower = query.lower()
        matches = []
        
        for concept, info in self.concepts.items():
            if query_lower in concept or query_lower in info['definition'].lower():
                matches.append(concept)
            elif any(query_lower in keyword for keyword in info.get('keywords', [])):
                matches.append(concept)
        
        for concept, definition in self.recent_topics.items():
            if query_lower in concept or query_lower in definition.lower():
                matches.append(concept)
        
        return list(set(matches))[:10]  # Remove duplicates and limit
