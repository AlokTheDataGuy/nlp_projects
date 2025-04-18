from pymongo import MongoClient
from app.core.config import settings

# MongoDB client instance
_client = None

def get_client():
    """Get MongoDB client instance"""
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI)
    return _client

def get_db():
    """Get MongoDB database instance"""
    client = get_client()
    return client[settings.MONGODB_DB]

def close_client():
    """Close MongoDB client connection"""
    global _client
    if _client is not None:
        _client.close()
        _client = None
