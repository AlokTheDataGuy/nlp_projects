from typing import Optional, List
from datetime import datetime
import openai
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from app.models.chat import ChatResponse, InsightReference
from app.core.config import settings
from app.db.mongodb import get_db

class ChatService:
    def __init__(self):
        self.db = get_db()
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        self.vector_store = Pinecone.from_existing_index(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=self.embeddings
        )
    
    def process_query(
        self,
        query: str,
        channel_id: Optional[str] = None,
        topic: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> ChatResponse:
        """Process a user query and return a response with sources"""
        # Search for relevant insights
        filter_dict = {}
        if channel_id:
            filter_dict["channel_id"] = channel_id
        
        # Note: topic filtering would need to be handled separately as Pinecone doesn't support array contains
        
        if from_date or to_date:
            date_filter = {}
            if from_date:
                date_filter["$gte"] = from_date.isoformat()
            if to_date:
                date_filter["$lte"] = to_date.isoformat()
            filter_dict["published_at"] = date_filter
        
        # Search vector store
        search_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=5,
            filter=filter_dict if filter_dict else None
        )
        
        # If we have topic filter, apply it manually
        if topic and search_results:
            # Get full insight documents for each result
            insight_ids = [result[0].metadata["insight_id"] for result in search_results]
            insights = list(self.db.insights.find({"_id": {"$in": insight_ids}}))
            
            # Filter by topic
            filtered_insights = [insight for insight in insights if topic in insight.get("topics", [])]
            
            # If we have filtered insights, update search_results
            if filtered_insights:
                filtered_ids = [str(insight["_id"]) for insight in filtered_insights]
                search_results = [result for result in search_results if result[0].metadata["insight_id"] in filtered_ids]
        
        # If no results found, return a generic response
        if not search_results:
            return ChatResponse(
                response="I don't have any specific insights about that topic from the YouTube channels I'm monitoring. Try a different query or check back later as I continue to analyze new content.",
                sources=[]
            )
        
        # Prepare context from search results
        context = "\n\n".join([
            f"Insight: {result[0].page_content}\nSource: {result[0].metadata['channel_name']} - {result[0].metadata['video_title']}"
            for result in search_results
        ])
        
        # Generate response using OpenAI
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides concise, informative responses based on insights extracted from YouTube videos. Use only the provided context to answer the question. If the context doesn't contain relevant information, acknowledge that you don't have specific insights on that topic."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Prepare sources
        sources = []
        for result in search_results:
            metadata = result[0].metadata
            sources.append(InsightReference(
                insight_id=metadata["insight_id"],
                video_id=metadata["video_id"],
                video_title=metadata["video_title"],
                channel_name=metadata["channel_name"],
                published_at=datetime.fromisoformat(metadata["published_at"])
            ))
        
        return ChatResponse(
            response=answer,
            sources=sources
        )
