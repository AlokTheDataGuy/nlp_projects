import logging
from typing import Dict, Any, List, Optional
from app.core.llm import llm_manager
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self):
        """
        Initialize the response generator.
        """
        pass
    
    def generate_response(self, query_result: Dict[str, Any], follow_up_context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate a response based on query results.
        
        Args:
            query_result: Result from query processor
            follow_up_context: Optional context from previous interactions
            
        Returns:
            Response dictionary
        """
        try:
            # Extract query and context
            query = query_result.get('query', '')
            context = query_result.get('context', '')
            intent = query_result.get('intent', 'general')
            relevant_papers = query_result.get('relevant_papers', [])
            
            # Add follow-up context if available
            if follow_up_context:
                context = self._add_follow_up_context(context, follow_up_context)
            
            # Generate response using LLM
            llm_response = llm_manager.generate_response(context, query)
            
            # Format the response
            formatted_response = self._format_response(llm_response, relevant_papers, intent)
            
            return {
                'response': formatted_response,
                'papers': [self._format_paper_reference(paper) for paper in relevant_papers],
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Fallback to rule-based responses if LLM fails
            fallback_response = self._generate_fallback_response(query_result)
            
            return {
                'response': fallback_response,
                'papers': [self._format_paper_reference(paper) for paper in query_result.get('relevant_papers', [])],
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _add_follow_up_context(self, context: str, follow_up_context: List[Dict[str, Any]]) -> str:
        """
        Add follow-up context to the main context.
        
        Args:
            context: Main context
            follow_up_context: Follow-up context from previous interactions
            
        Returns:
            Updated context string
        """
        follow_up_parts = []
        
        for i, item in enumerate(follow_up_context):
            query = item.get('query', '')
            response = item.get('response', '')
            
            follow_up_part = f"Previous Question: {query}\nPrevious Answer: {response}\n\n"
            follow_up_parts.append(follow_up_part)
        
        # Add follow-up context at the beginning
        follow_up_section = "Previous Conversation:\n" + "\n".join(follow_up_parts)
        
        return follow_up_section + "\n\nCurrent Context:\n" + context
    
    def _format_response(self, llm_response: str, papers: List[Dict[str, Any]], intent: str) -> str:
        """
        Format the LLM response.
        
        Args:
            llm_response: Raw response from LLM
            papers: List of relevant papers
            intent: Query intent
            
        Returns:
            Formatted response
        """
        # For now, just return the LLM response
        # In a more advanced implementation, we could add citations, format sections, etc.
        return llm_response
    
    def _format_paper_reference(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a paper for reference in the response.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Formatted paper reference
        """
        return {
            'paper_id': paper.get('paper_id', ''),
            'title': paper.get('title', 'Untitled'),
            'authors': paper.get('authors', 'Unknown authors'),
            'published_date': paper.get('published_date', ''),
            'categories': paper.get('categories', ''),
            'url': paper.get('url', f"https://arxiv.org/abs/{paper.get('paper_id', '')}")
        }
    
    def _generate_fallback_response(self, query_result: Dict[str, Any]) -> str:
        """
        Generate a fallback response when LLM fails.
        
        Args:
            query_result: Result from query processor
            
        Returns:
            Fallback response
        """
        query = query_result.get('query', '')
        intent = query_result.get('intent', 'general')
        
        # Check for common CS topics and provide relevant responses
        query_lower = query.lower()
        
        if "neural network" in query_lower or "deep learning" in query_lower:
            return "Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes or 'neurons' that process and transform input data.\n\nDeep learning refers to neural networks with many layers (deep neural networks) that can learn hierarchical representations of data. Key architectures include Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks (RNNs) for sequential data, and Transformers for natural language processing.\n\nRecent advances in neural networks have led to breakthroughs in computer vision, natural language processing, and reinforcement learning."
        elif "machine learning" in query_lower or "ml" in query_lower:
            return "Machine learning is a field of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed.\n\nThere are three main types of machine learning:\n\n1. Supervised Learning: The algorithm learns from labeled training data to make predictions or decisions.\n\n2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.\n\n3. Reinforcement Learning: The algorithm learns by interacting with an environment and receiving rewards or penalties.\n\nPopular machine learning algorithms include decision trees, support vector machines, k-nearest neighbors, and neural networks."
        elif "transformer" in query_lower or "attention" in query_lower or "nlp" in query_lower:
            return "Transformers are a type of neural network architecture introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. They revolutionized natural language processing (NLP) by using self-attention mechanisms to process sequential data in parallel, rather than sequentially like RNNs.\n\nThe key innovation in Transformers is the self-attention mechanism, which allows the model to weigh the importance of different words in a sentence when processing each word. This enables the model to capture long-range dependencies and contextual relationships in text.\n\nTransformer-based models like BERT, GPT, and T5 have achieved state-of-the-art results on a wide range of NLP tasks, including text classification, question answering, and language generation."
        elif "llm" in query_lower or "large language model" in query_lower or "gpt" in query_lower:
            return "Large Language Models (LLMs) are advanced neural network architectures trained on vast amounts of text data to understand and generate human-like text. They represent a significant advancement in natural language processing and artificial intelligence.\n\nLLMs like GPT (Generative Pre-trained Transformer), LLaMA, and Claude are based on the Transformer architecture and are trained using self-supervised learning on diverse text corpora. These models have billions of parameters that allow them to capture complex patterns in language.\n\nKey capabilities of LLMs include:\n\n1. Text generation with coherence across long contexts\n2. Few-shot and zero-shot learning for new tasks\n3. Following instructions and responding to complex queries\n4. Code generation and understanding\n5. Reasoning about abstract concepts\n\nRecent research focuses on alignment techniques, reducing hallucinations, and improving factuality and safety."
        elif "algorithm" in query_lower or "algorithms" in query_lower or "data structure" in query_lower:
            return "Algorithms are step-by-step procedures or formulas for solving problems, particularly in computing. They are the foundation of computer science and are essential for tasks ranging from simple calculations to complex data processing.\n\nKey categories of algorithms include:\n\n1. Sorting algorithms (e.g., QuickSort, MergeSort, HeapSort)\n2. Search algorithms (e.g., Binary Search, Depth-First Search, Breadth-First Search)\n3. Graph algorithms (e.g., Dijkstra's, Bellman-Ford, Floyd-Warshall)\n4. Dynamic programming algorithms\n5. Greedy algorithms\n6. Divide and conquer algorithms\n\nThe efficiency of algorithms is typically measured using Big O notation, which describes how the runtime or space requirements grow as the input size increases.\n\nData structures are specialized formats for organizing and storing data. Common data structures include arrays, linked lists, stacks, queues, trees, graphs, and hash tables. The choice of data structure significantly impacts the efficiency of algorithms that operate on the data."
        elif "hello" in query_lower or "hi" in query_lower or "hey" in query_lower:
            return "Hello! I'm the ArXiv Expert Chatbot. I can help you understand computer science concepts and research papers. Feel free to ask me about topics like machine learning, neural networks, algorithms, or specific research areas. How can I assist you today?"
        else:
            return f"You asked about: {query}\n\nI found some relevant papers that might help answer your question. Please take a look at the paper references for more information on this topic. If you have more specific questions about any of these papers or need clarification on certain concepts, feel free to ask!"

# Create a singleton instance
response_generator = ResponseGenerator()
