from typing import List, Optional
from datetime import datetime
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from app.models.channel import Channel, ChannelCreate
from app.models.video import Video
from app.core.config import settings
from app.db.mongodb import get_db

class YouTubeService:
    def __init__(self):
        self.api_key = settings.YOUTUBE_API_KEY
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=self.api_key
        )
        self.db = get_db()
        
    def get_all_channels(self) -> List[Channel]:
        """Get all monitored YouTube channels"""
        channels = self.db.channels.find()
        return [Channel(**channel) for channel in channels]
    
    def add_channel(self, channel: ChannelCreate) -> Channel:
        """Add a new YouTube channel to monitor"""
        # Verify channel exists on YouTube
        try:
            response = self.youtube.channels().list(
                part="snippet",
                id=channel.youtube_id
            ).execute()
            
            if not response["items"]:
                raise ValueError(f"Channel with ID {channel.youtube_id} not found on YouTube")
            
            # Create channel in database
            channel_data = channel.dict()
            channel_data["created_at"] = datetime.now()
            
            result = self.db.channels.insert_one(channel_data)
            
            # Return the created channel
            return Channel(
                id=str(result.inserted_id),
                **channel_data
            )
            
        except HttpError as e:
            raise ValueError(f"YouTube API error: {str(e)}")
    
    def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get a specific YouTube channel by ID"""
        channel = self.db.channels.find_one({"_id": channel_id})
        if channel:
            return Channel(**channel)
        return None
    
    def delete_channel(self, channel_id: str) -> bool:
        """Remove a YouTube channel from monitoring"""
        result = self.db.channels.delete_one({"_id": channel_id})
        return result.deleted_count > 0
    
    def get_videos(self, channel_id: Optional[str] = None, limit: int = 10, skip: int = 0) -> List[Video]:
        """Get videos, optionally filtered by channel"""
        query = {}
        if channel_id:
            query["channel_id"] = channel_id
            
        videos = self.db.videos.find(query).sort("published_at", -1).skip(skip).limit(limit)
        return [Video(**video) for video in videos]
    
    def get_video(self, video_id: str) -> Optional[Video]:
        """Get a specific video by ID"""
        video = self.db.videos.find_one({"_id": video_id})
        if video:
            return Video(**video)
        return None
    
    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get the transcript for a specific video"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([item["text"] for item in transcript_list])
            return transcript_text
        except (TranscriptsDisabled, Exception) as e:
            return None
    
    def check_new_videos(self):
        """Check for new videos from all monitored channels"""
        channels = self.get_all_channels()
        
        for channel in channels:
            try:
                # Get latest videos from channel
                response = self.youtube.search().list(
                    part="snippet",
                    channelId=channel.youtube_id,
                    maxResults=10,
                    order="date",
                    type="video"
                ).execute()
                
                # Update last_checked timestamp
                self.db.channels.update_one(
                    {"_id": channel.id},
                    {"$set": {"last_checked": datetime.now()}}
                )
                
                # Process each video
                for item in response.get("items", []):
                    video_id = item["id"]["videoId"]
                    
                    # Check if video already exists in database
                    existing_video = self.db.videos.find_one({"youtube_id": video_id})
                    if existing_video:
                        continue
                    
                    # Get full video details
                    video_response = self.youtube.videos().list(
                        part="snippet,contentDetails",
                        id=video_id
                    ).execute()
                    
                    if not video_response["items"]:
                        continue
                    
                    video_data = video_response["items"][0]
                    snippet = video_data["snippet"]
                    
                    # Create new video in database
                    new_video = {
                        "youtube_id": video_id,
                        "title": snippet["title"],
                        "description": snippet.get("description", ""),
                        "channel_id": channel.id,
                        "published_at": datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00")),
                        "created_at": datetime.now(),
                        "transcript_processed": False,
                        "insights_extracted": False
                    }
                    
                    self.db.videos.insert_one(new_video)
                    
            except HttpError as e:
                print(f"Error checking videos for channel {channel.id}: {str(e)}")
                continue
