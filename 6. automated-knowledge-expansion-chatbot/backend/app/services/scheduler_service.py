import schedule
import time
import threading
from datetime import datetime

from app.services.youtube_service import YouTubeService
from app.services.transcript_service import TranscriptService
from app.services.insight_service import InsightService

class SchedulerService:
    def __init__(self):
        self.youtube_service = YouTubeService()
        self.transcript_service = TranscriptService()
        self.insight_service = InsightService()
        self.scheduler_thread = None
        self.running = False
    
    def setup_schedule(self):
        """Set up the scheduled tasks"""
        # Check for new videos every hour
        schedule.every(1).hour.do(self.youtube_service.check_new_videos)
        
        # Process pending transcripts every 2 hours
        schedule.every(2).hours.do(self.transcript_service.process_pending_transcripts)
        
        # Process pending insights every 4 hours
        schedule.every(4).hours.do(self.insight_service.process_pending_insights)
        
        # Clean up old insights once a day
        schedule.every(1).day.at("03:00").do(self.insight_service.cleanup_old_insights)
    
    def run_scheduler(self):
        """Run the scheduler in a loop"""
        self.running = True
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start(self):
        """Start the scheduler in a background thread"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return
        
        self.setup_schedule()
        self.scheduler_thread = threading.Thread(target=self.run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
    
    def run_now(self, task_name: str):
        """Run a specific task immediately"""
        if task_name == "check_videos":
            self.youtube_service.check_new_videos()
        elif task_name == "process_transcripts":
            self.transcript_service.process_pending_transcripts()
        elif task_name == "process_insights":
            self.insight_service.process_pending_insights()
        elif task_name == "cleanup_insights":
            self.insight_service.cleanup_old_insights()
        else:
            raise ValueError(f"Unknown task: {task_name}")
