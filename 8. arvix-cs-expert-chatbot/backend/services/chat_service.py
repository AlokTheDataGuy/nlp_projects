from typing import List, Dict, Any, Optional
import asyncio
import uuid

from services.arxiv_service import ArxivService
from services.paper_processor import PaperProcessor
from services.ollama_service import OllamaService
from services.vector_service import VectorService
from utils.prompt_templates import create_system_message, create_chat_prompt

class ChatService:
    def __init__(self):
        self.arxiv_service = ArxivService()
        self.paper_processor = PaperProcessor()
        self.ollama_service = OllamaService(model_name="llama3.1:8b")
        self.vector_service = VectorService()
        self.conversations = {}
        
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        # Initialize conversation if needed
        if user_id not in self.conversations:
            self.conversations[user_id] = []
            conversation_id = str(uuid.uuid4())
        else:
            conversation_id = self.conversations[user_id][0].get("conversation_id", str(uuid.uuid4()))
        
        # Extract query concepts and intent
        query_concepts = self._extract_concepts(message)
        
        # Search relevant papers
        papers = await self.arxiv_service.search_papers(
            query=message,
            concepts=query_concepts,
            max_results=10
        )
        
        # Get relevant papers from vector DB if available
        vector_results = await self.vector_service.semantic_search(message, top_k=5)
        
        # Combine results
        all_papers = self._merge_paper_results(papers, vector_results)
        
        # Process top papers
        paper_contents = []
        tasks = []
        for paper in all_papers[:3]:  # Process top 3 papers
            tasks.append(self.paper_processor.extract_content(paper['id']))
        
        content_results = await asyncio.gather(*tasks)
        for idx, content in enumerate(content_results):
            paper_contents.append({
                'paper': all_papers[idx],
                'content': content
            })
        
        # Generate embeddings and store for future retrieval
        await self.vector_service.index_papers(paper_contents)
        
        # Create prompt
        system_message = create_system_message()
        prompt = create_chat_prompt(
            query=message,
            papers=all_papers[:5],
            paper_contents=paper_contents,
            conversation_history=self.conversations.get(user_id, [])
        )
        
        # Generate response
        response = await self.ollama_service.generate(
            prompt=prompt,
            system_message=system_message
        )
        
        # Update conversation history
        message_entry = {
            "conversation_id": conversation_id,
            "user": message,
            "assistant": response,
            "papers": [p['id'] for p in all_papers[:3]],
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.conversations.setdefault(user_id, []).append(message_entry)
        
        return {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "response": response,
            "relevant_papers": all_papers[:5],
            "timestamp": message_entry["timestamp"]
        }
    
    def _extract_concepts(self, message: str) -> List[str]:
        # Simple implementation - could be enhanced with NLP
        # Split message into words, filter out common words
        words = message.lower().split()
        stopwords = {"the", "a", "an", "in", "of", "and", "or", "to", "is", "are", "what", "how"}
        return [word for word in words if word not in stopwords and len(word) > 2]
    
    def _merge_paper_results(self, api_results, vector_results):
        # Create a dictionary of paper IDs to eliminate duplicates
        paper_dict = {paper['id']: paper for paper in api_results}
        
        # Add vector results if not already in the dictionary
        for paper in vector_results:
            if paper['id'] not in paper_dict:
                paper_dict[paper['id']] = paper
        
        # Convert back to list and sort by relevance
        return list(paper_dict.values())