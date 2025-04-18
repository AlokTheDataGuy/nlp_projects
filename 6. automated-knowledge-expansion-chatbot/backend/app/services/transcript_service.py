from typing import Optional, List
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from app.db.mongodb import get_db

class TranscriptService:
    def __init__(self):
        self.db = get_db()
    
    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript for a YouTube video"""
        try:
            # Check if we already have the transcript in the database
            transcript_doc = self.db.transcripts.find_one({"video_id": video_id})
            if transcript_doc:
                return transcript_doc["text"]
            
            # If not, fetch from YouTube
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([item["text"] for item in transcript_list])
            
            # Store in database
            self.db.transcripts.insert_one({
                "video_id": video_id,
                "text": transcript_text
            })
            
            # Mark video as transcript processed
            self.db.videos.update_one(
                {"youtube_id": video_id},
                {"$set": {"transcript_processed": True}}
            )
            
            return transcript_text
            
        except (TranscriptsDisabled, Exception) as e:
            print(f"Error getting transcript for video {video_id}: {str(e)}")
            return None
    
    def process_pending_transcripts(self):
        """Process transcripts for all videos that haven't been processed yet"""
        pending_videos = self.db.videos.find({"transcript_processed": False})
        
        for video in pending_videos:
            try:
                self.get_transcript(video["youtube_id"])
            except Exception as e:
                print(f"Error processing transcript for video {video['youtube_id']}: {str(e)}")
                continue
