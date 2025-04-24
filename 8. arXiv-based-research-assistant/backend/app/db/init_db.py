from app.db.database import engine
from app.models import models
import logging

logger = logging.getLogger(__name__)

def init_db():
    """
    Initialize the database by creating all tables.
    """
    try:
        # Create tables
        models.Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize database
    init_db()
