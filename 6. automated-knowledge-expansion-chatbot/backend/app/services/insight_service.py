from typing import List, Optional
from datetime import datetime, timedelta
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from app.models.insight import Insight, InsightCreate
from app.core.config import settings
from app.db.mongodb import get_db
from app.services.transcript_service import TranscriptService

class InsightService:
    def __init__(self):
        self.db = get_db()
        self.transcript_service = TranscriptService()
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize Pinecone
        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )
        
        # Create index if it doesn't exist
        if settings.PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine"
            )
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        self.vector_store = Pinecone.from_existing_index(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=self.embeddings
        )
    
    def get_insights(
        self,
        channel_id: Optional[str] = None,
        topic: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 10,
        skip: int = 0
    ) -> List[Insight]:
        """Get insights with optional filtering"""
        query = {}
        
        if channel_id:
            query["channel_id"] = channel_id
            
        if topic:
            query["topics"] = topic
            
        if from_date or to_date:
            date_query = {}
            if from_date:
                date_query["$gte"] = from_date
            if to_date:
                date_query["$lte"] = to_date
            query["created_at"] = date_query
        
        insights = self.db.insights.find(query).sort("created_at", -1).skip(skip).limit(limit)
        return [Insight(**insight) for insight in insights]
    
    def get_insight(self, insight_id: str) -> Optional[Insight]:
        """Get a specific insight by ID"""
        insight = self.db.insights.find_one({"_id": insight_id})
        if insight:
            return Insight(**insight)
        return None
    
    def get_insights_by_video(self, video_id: str) -> List[Insight]:
        """Get all insights for a specific video"""
        insights = self.db.insights.find({"video_id": video_id}).sort("confidence_score", -1)
        return [Insight(**insight) for insight in insights]
    
    def extract_insights_from_transcript(self, video_id: str):
        """Extract insights from a video transcript"""
        # Get video details
        video = self.db.videos.find_one({"youtube_id": video_id})
        if not video:
            raise ValueError(f"Video with ID {video_id} not found")
        
        # Get channel details
        channel = self.db.channels.find_one({"_id": video["channel_id"]})
        if not channel:
            raise ValueError(f"Channel with ID {video['channel_id']} not found")
        
        # Get transcript
        transcript = self.transcript_service.get_transcript(video_id)
        if not transcript:
            raise ValueError(f"Transcript for video {video_id} not available")
        
        # Split transcript into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(transcript)
        
        # Process each chunk to extract insights
        for chunk in chunks:
            try:
                # Use OpenAI to extract insights
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that extracts key insights from YouTube video transcripts. Identify the most important, high-impact information that would be valuable to someone who hasn't watched the video. Focus on concrete facts, novel ideas, and actionable takeaways."},
                        {"role": "user", "content": f"Extract the most important insight from this transcript chunk of a video titled '{video['title']}' from the channel '{channel['name']}':\n\n{chunk}"}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                
                insight_text = response.choices[0].message.content.strip()
                
                # Skip if the insight is too short or not meaningful
                if len(insight_text) < 20 or "no significant insight" in insight_text.lower():
                    continue
                
                # Use OpenAI to identify topics
                topics_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that identifies relevant topics for a given insight. Return only a JSON array of topic strings."},
                        {"role": "user", "content": f"Identify 1-3 relevant topics for this insight:\n\n{insight_text}"}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )
                
                topics_text = topics_response.choices[0].message.content.strip()
                
                # Extract topics from the response
                import re
                import json
                
                # Try to extract JSON array
                try:
                    topics_match = re.search(r'\[.*\]', topics_text)
                    if topics_match:
                        topics = json.loads(topics_match.group(0))
                    else:
                        topics = []
                except:
                    topics = []
                
                # Create insight
                insight_data = {
                    "content": insight_text,
                    "video_id": video_id,
                    "channel_id": video["channel_id"],
                    "topics": topics,
                    "confidence_score": 0.8,  # Default confidence score
                    "created_at": datetime.now(),
                    "video_title": video["title"],
                    "channel_name": channel["name"],
                    "video_published_at": video["published_at"]
                }
                
                # Store in MongoDB
                result = self.db.insights.insert_one(insight_data)
                insight_id = str(result.inserted_id)
                insight_data["id"] = insight_id
                
                # Store in vector database
                self.vector_store.add_texts(
                    texts=[insight_text],
                    metadatas=[{
                        "insight_id": insight_id,
                        "video_id": video_id,
                        "video_title": video["title"],
                        "channel_id": video["channel_id"],
                        "channel_name": channel["name"],
                        "published_at": video["published_at"].isoformat()
                    }]
                )
                
            except Exception as e:
                print(f"Error processing chunk for video {video_id}: {str(e)}")
                continue
        
        # Mark video as insights extracted
        self.db.videos.update_one(
            {"youtube_id": video_id},
            {"$set": {"insights_extracted": True}}
        )
    
    def process_pending_insights(self):
        """Process insights for all videos with transcripts but no insights"""
        pending_videos = self.db.videos.find({
            "transcript_processed": True,
            "insights_extracted": False
        })
        
        for video in pending_videos:
            try:
                self.extract_insights_from_transcript(video["youtube_id"])
            except Exception as e:
                print(f"Error extracting insights for video {video['youtube_id']}: {str(e)}")
                continue
    
    def cleanup_old_insights(self, days: int = 30):
        """Remove insights older than the specified number of days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get IDs of insights to remove
        old_insights = self.db.insights.find({"created_at": {"$lt": cutoff_date}})
        insight_ids = [str(insight["_id"]) for insight in old_insights]
        
        # Remove from MongoDB
        self.db.insights.delete_many({"_id": {"$in": insight_ids}})
        
        # Remove from vector store
        # Note: This depends on Pinecone's API for batch deletion
        index = pinecone.Index(settings.PINECONE_INDEX_NAME)
        for insight_id in insight_ids:
            try:
                # This assumes we're using insight_id as the vector ID
                index.delete(ids=[insight_id])
            except:
                continue
