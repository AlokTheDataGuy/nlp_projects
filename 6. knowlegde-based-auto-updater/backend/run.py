import uvicorn
from app.services.scheduler_service import SchedulerService

if __name__ == "__main__":
    # Start the scheduler service
    scheduler = SchedulerService()
    scheduler.start()
    
    # Run the FastAPI application
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
