import yaml
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from rag_system import RAGSystem
from knowledge_base import CSKnowledgeBase

class QueryType(Enum):
    FUNDAMENTAL = "fundamental"
    ADVANCED = "advanced" 
    RECENT = "recent"
    PAPER_SPECIFIC = "paper_specific"

@dataclass
class LLMResponse:
    content: str
    confidence: float
    sources: List[Dict]
    query_type: QueryType
    follow_up_suggestions: List[str]

class FoundationLLMEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.llm_config = self.config['llm']
        
        # Initialize components
        self.rag_system = RAGSystem(config_path)
        self.knowledge_base = CSKnowledgeBase()
        
        # Setup LLM
        self.setup_llm()
        
        # Conversation history for context
        self.conversation_history = []
        
    def setup_llm(self):
        """Setup the foundation LLM (Llama 3 via Ollama)"""
        if not OLLAMA_AVAILABLE:
            print("Ollama not available. Install with: pip install ollama")
            self.llm = None
            return
        
        try:
            # Check if model is available
            models = ollama.list()
            available_models = [model['name'] for model in models.get('models', [])]
            
            model_name = self.llm_config['model_name']
            if model_name not in available_models:
                print(f"Model {model_name} not found. Pulling from Ollama...")
                ollama.pull(model_name)
            
            # Test the model
            test_response = ollama.generate(
                model=model_name,
                prompt="Hello, world!",
                options={'max_tokens': 10}
            )
            
            self.model_name = model_name
            print(f"Successfully connected to {model_name}")
            
        except Exception as e:
            print(f"Error setting up {self.llm_config['model_name']}: {e}")
            
            # Try fallback model
            try:
                fallback_model = self.llm_config['fallback_model']
                print(f"Trying fallback model: {fallback_model}")
                
                test_response = ollama.generate(
                    model=fallback_model,
                    prompt="Hello, world!",
                    options={'max_tokens': 10}
                )
                
                self.model_name = fallback_model
                print(f"Successfully connected to fallback model: {fallback_model}")
                
            except Exception as fallback_error:
                print(f"Fallback model also failed: {fallback_error}")
                self.model_name = None
    
    def classify_query(self, query: str) -> QueryType:
        """Classify the type of query to determine response strategy"""
        query_lower = query.lower()
        
        # Check for recent/temporal keywords
        recent_keywords = ['2023', '2024', '2025', 'latest', 'recent', 'new', 'emerging', 'current']
        if any(keyword in query_lower for keyword in recent_keywords):
            return QueryType.RECENT
        
        # Check for paper-specific requests
        paper_keywords = ['paper', 'research', 'study', 'publication', 'article', 'summarize this', 'explain this paper']
        if any(keyword in query_lower for keyword in paper_keywords):
            return QueryType.PAPER_SPECIFIC
        
        # Check for advanced topics
        advanced_keywords = [
            'state-of-the-art', 'sota', 'breakthrough', 'novel', 'cutting-edge',
            'transformer', 'attention mechanism', 'bert', 'gpt', 'neural architecture',
            'deep learning', 'reinforcement learning', 'generative ai'
        ]
        if any(keyword in query_lower for keyword in advanced_keywords):
            return QueryType.ADVANCED
        
        # Default to fundamental
        return QueryType.FUNDAMENTAL
    
    def generate_response(self, query: str, context_papers: Optional[List[Dict]] = None) -> LLMResponse:
        """Generate comprehensive response using foundation LLM + RAG"""
        
        if not self.model_name:
            return self.fallback_response(query)
        
        # Classify query
        query_type = self.classify_query(query)
        
        # Get relevant context
        if context_papers is None:
            context_papers = self.rag_system.retrieve_relevant_papers(query)
        
        # Generate response based on query type
        if query_type == QueryType.FUNDAMENTAL:
            response = self.handle_fundamental_query(query, context_papers)
        elif query_type == QueryType.ADVANCED:
            response = self.handle_advanced_query(query, context_papers)
        elif query_type == QueryType.RECENT:
            response = self.handle_recent_query(query, context_papers)
        else:  # PAPER_SPECIFIC
            response = self.handle_paper_specific_query(query, context_papers)
        
        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'response': response.content,
            'query_type': query_type.value,
            'timestamp': self.get_timestamp()
        })
        
        return response
    
    def handle_fundamental_query(self, query: str, papers: List[Dict]) -> LLMResponse:
        """Handle queries about fundamental CS concepts"""
        
        # First, check knowledge base
        concepts = self.knowledge_base.search_concepts(query)
        base_knowledge = ""
        
        if concepts:
            concept = concepts[0]
            concept_info = self.knowledge_base.get_concept_info(concept)
            if concept_info:
                base_knowledge = f"""
                Concept: {concept.title()}
                Definition: {concept_info['definition']}
                Key Applications: {', '.join(concept_info.get('applications', []))}
                Related Topics: {', '.join(concept_info.get('subconcepts', []))}
                """
        
        # Create prompt for LLM
        prompt = self.create_fundamental_prompt(query, base_knowledge, papers[:3])
        
        # Generate response
        response_text = self.call_llm(prompt)
        
        # Generate follow-up suggestions
        follow_ups = self.generate_follow_up_questions(query, "fundamental")
        
        return LLMResponse(
            content=response_text,
            confidence=0.8,
            sources=papers[:3],
            query_type=QueryType.FUNDAMENTAL,
            follow_up_suggestions=follow_ups
        )
    
    def handle_advanced_query(self, query: str, papers: List[Dict]) -> LLMResponse:
        """Handle queries about advanced CS topics"""
        
        # Create rich context from papers
        context = self.rag_system.create_context_for_query(query, max_context_length=1500)
        
        # Create advanced prompt
        prompt = self.create_advanced_prompt(query, context)
        
        # Generate response
        response_text = self.call_llm(prompt)
        
        # Generate follow-up suggestions
        follow_ups = self.generate_follow_up_questions(query, "advanced")
        
        return LLMResponse(
            content=response_text,
            confidence=0.9,
            sources=papers[:5],
            query_type=QueryType.ADVANCED,
            follow_up_suggestions=follow_ups
        )
    
    def handle_recent_query(self, query: str, papers: List[Dict]) -> LLMResponse:
        """Handle queries about recent developments"""
        
        # Focus on most recent papers
        recent_papers = sorted(papers, key=lambda x: x.get('published', ''), reverse=True)[:5]
        
        # Create context emphasizing recency
        context = self.create_recent_context(recent_papers)
        
        # Create recent-focused prompt
        prompt = self.create_recent_prompt(query, context)
        
        # Generate response
        response_text = self.call_llm(prompt)
        
        # Generate follow-up suggestions
        follow_ups = self.generate_follow_up_questions(query, "recent")
        
        return LLMResponse(
            content=response_text,
            confidence=0.85,
            sources=recent_papers,
            query_type=QueryType.RECENT,
            follow_up_suggestions=follow_ups
        )
    
    def handle_paper_specific_query(self, query: str, papers: List[Dict]) -> LLMResponse:
        """Handle queries about specific papers"""
        
        if not papers:
            return LLMResponse(
                content="I couldn't find specific papers related to your query. Could you provide more details or try a different search term?",
                confidence=0.3,
                sources=[],
                query_type=QueryType.PAPER_SPECIFIC,
                follow_up_suggestions=["Could you specify the paper title or authors?", "What aspect of the research interests you most?"]
            )
        
        # Focus on top papers
        top_papers = papers[:3]
        
        # Create paper-specific prompt
        prompt = self.create_paper_specific_prompt(query, top_papers)
        
        # Generate response
        response_text = self.call_llm(prompt)
        
        # Generate follow-up suggestions
        follow_ups = self.generate_follow_up_questions(query, "paper_specific")
        
        return LLMResponse(
            content=response_text,
            confidence=0.9,
            sources=top_papers,
            query_type=QueryType.PAPER_SPECIFIC,
            follow_up_suggestions=follow_ups
        )
    
    def create_fundamental_prompt(self, query: str, base_knowledge: str, papers: List[Dict]) -> str:
        """Create prompt for fundamental concepts"""
        
        paper_context = ""
        if papers:
            paper_context = "\nRecent Research Context:\n"
            for i, paper in enumerate(papers, 1):
                paper_context += f"{i}. {paper['title']}\n   Abstract: {paper['document'][:200]}...\n\n"
        
        conversation_context = self.get_conversation_context()
        
        prompt = f"""You are a computer science expert professor. Your task is to provide clear, comprehensive explanations of fundamental CS concepts.

{conversation_context}

Base Knowledge:
{base_knowledge}

{paper_context}

Question: {query}

Provide a detailed but accessible explanation that:
1. Explains the concept clearly with proper definitions
2. Describes key principles and how they work
3. Gives practical examples and applications
4. Mentions current relevance and importance
5. Uses the research context to add depth and current insights

Keep the explanation educational and well-structured. Use examples to make complex ideas understandable."""
        
        return prompt
    
    def create_advanced_prompt(self, query: str, context: Dict) -> str:
        """Create prompt for advanced topics"""
        
        conversation_context = self.get_conversation_context()
        
        prompt = f"""You are a leading computer science researcher with deep expertise in cutting-edge technologies. 

{conversation_context}

Research Context:
{context['text']}

Question: {query}

Provide a comprehensive analysis that:
1. Explains the advanced concepts with technical accuracy
2. Discusses current state-of-the-art approaches
3. Analyzes the research findings and their implications
4. Compares different methodologies and their trade-offs
5. Identifies challenges and future research directions

Base your response on the provided research context while incorporating your expertise. Be thorough but clear in your explanations."""
        
        return prompt
    
    def create_recent_prompt(self, query: str, context: str) -> str:
        """Create prompt for recent developments"""
        
        conversation_context = self.get_conversation_context()
        
        prompt = f"""You are a computer science researcher specializing in the latest developments and trends in the field.

{conversation_context}

Latest Research:
{context}

Question: {query}

Provide an up-to-date analysis that:
1. Highlights the most recent developments and breakthroughs
2. Explains new methodologies and their advantages
3. Discusses current trends and their implications
4. Compares recent approaches with previous methods
5. Predicts future directions based on current research

Focus on recent findings and emerging trends. Emphasize what's new and significant in the current research landscape."""
        
        return prompt
    
    def create_paper_specific_prompt(self, query: str, papers: List[Dict]) -> str:
        """Create prompt for paper-specific queries"""
        
        conversation_context = self.get_conversation_context()
        
        papers_text = ""
        for i, paper in enumerate(papers, 1):
            papers_text += f"""
Paper {i}: {paper['title']}
Authors: {', '.join(paper['authors'])}
Abstract: {paper['document']}
Categories: {', '.join(paper['categories'])}
Published: {paper['published']}

"""
        
        prompt = f"""You are a research paper analyst with expertise in computer science literature.

{conversation_context}

Papers to Analyze:
{papers_text}

Question: {query}

Provide a detailed analysis that:
1. Summarizes the key contributions of each paper
2. Explains the methodologies and approaches used
3. Discusses the results and their significance
4. Compares different papers if multiple are provided
5. Explains the practical implications and applications

Focus on extracting and explaining the most important insights from the research papers."""
        
        return prompt
    
    def call_llm(self, prompt: str) -> str:
        """Call the foundation LLM with the given prompt"""
        
        if not self.model_name:
            return self.simple_fallback_response(prompt)
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': self.llm_config['temperature'],
                    'max_tokens': self.llm_config['max_tokens'],
                    'top_p': 0.9,
                    'stop': ['Human:', 'User:']
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self.simple_fallback_response(prompt)
    
    def generate_follow_up_questions(self, query: str, query_type: str) -> List[str]:
        """Generate contextual follow-up questions"""
        
        base_questions = {
            "fundamental": [
                "How is this concept applied in real-world systems?",
                "What are the main challenges and limitations?",
                "How has this field evolved over time?"
            ],
            "advanced": [
                "What are the current research challenges?",
                "How do different approaches compare?",
                "What are the future research directions?"
            ],
            "recent": [
                "What are the practical implications of these developments?",
                "How do these findings change current practices?",
                "What should we expect in the next few years?"
            ],
            "paper_specific": [
                "How do these findings compare to previous work?",
                "What are the limitations of this research?",
                "What future work do the authors suggest?"
            ]
        }
        
        return base_questions.get(query_type, base_questions["fundamental"])
    
    def get_conversation_context(self, max_turns: int = 3) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return ""
        
        recent_history = self.conversation_history[-max_turns:]
        context = "Previous conversation context:\n"
        
        for turn in recent_history:
            context += f"Q: {turn['query'][:100]}...\n"
            context += f"A: {turn['response'][:150]}...\n\n"
        
        return context
    
    def create_recent_context(self, papers: List[Dict]) -> str:
        """Create context focusing on recency"""
        context = "Recent Research Developments:\n\n"
        
        for i, paper in enumerate(papers, 1):
            context += f"{i}. {paper['title']} ({paper.get('published', 'Unknown date')})\n"
            context += f"   {paper['document'][:300]}...\n\n"
        
        return context
    
    def fallback_response(self, query: str) -> LLMResponse:
        """Provide fallback response when LLM is unavailable"""
        
        # Search knowledge base
        concepts = self.knowledge_base.search_concepts(query)
        papers = self.rag_system.retrieve_relevant_papers(query, top_k=3)
        
        response_text = "I'm currently running in simplified mode. Here's what I can tell you:\n\n"
        
        if concepts:
            concept = concepts[0]
            concept_info = self.knowledge_base.get_concept_info(concept)
            if concept_info:
                response_text += f"**{concept.title()}**: {concept_info['definition']}\n\n"
        
        if papers:
            response_text += "**Relevant Research Papers**:\n\n"
            for i, paper in enumerate(papers[:3], 1):
                response_text += f"{i}. **{paper['title']}**\n"
                response_text += f"   {paper['document'][:200]}...\n\n"
        
        return LLMResponse(
            content=response_text,
            confidence=0.6,
            sources=papers,
            query_type=QueryType.FUNDAMENTAL,
            follow_up_suggestions=["Could you ask about a specific aspect?", "Would you like more technical details?"]
        )
    
    def simple_fallback_response(self, prompt: str) -> str:
        """Simple text-based fallback response"""
        return "I'm currently unable to process your request with the full AI model. Please ensure Ollama is installed and running with the required models."
    
    def get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'available': self.model_name is not None,
            'conversation_turns': len(self.conversation_history)
        }
