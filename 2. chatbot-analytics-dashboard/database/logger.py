# File: database/logger.py
# Functions for logging chat interactions to the database

import sqlite3
import uuid
import os
from datetime import datetime
import json
from typing import Optional, Dict, Any

def get_db_path():
    db_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(db_dir, 'chatbot_analytics.db')

def create_session():
    """Create a new session and return its ID"""
    session_id = str(uuid.uuid4())
    try:
        conn = sqlite3.connect(get_db_path(), timeout=20)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id) VALUES (?)",
            (session_id,)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error in create_session: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
    return session_id

def end_session(session_id):
    """End a session and calculate its duration"""
    try:
        conn = sqlite3.connect(get_db_path(), timeout=20)
        cursor = conn.cursor()

        # Get the start time
        cursor.execute("SELECT start_time FROM sessions WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        if not result:
            return

        start_time = datetime.fromisoformat(result[0])
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Update the session
        cursor.execute(
            "UPDATE sessions SET end_time = ?, duration_seconds = ? WHERE session_id = ?",
            (end_time.isoformat(), duration, session_id)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error in end_session: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def log_conversation(
    session_id: str,
    user_query: str,
    bot_response: str,
    topic: Optional[str] = None,
    satisfaction_rating: Optional[int] = None,
    query_success: Optional[bool] = None
):
    """Log a conversation interaction to the database"""
    conversation_id = None
    try:
        conn = sqlite3.connect(get_db_path(), timeout=20)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO conversations
               (session_id, user_query, bot_response, topic, satisfaction_rating, query_success)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, user_query, bot_response, topic, satisfaction_rating, query_success)
        )
        conversation_id = cursor.lastrowid
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error in log_conversation: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
    return conversation_id

def update_conversation_feedback(conversation_id, satisfaction_rating=None, query_success=None):
    """Update feedback metrics for a conversation"""
    try:
        conn = sqlite3.connect(get_db_path(), timeout=20)
        cursor = conn.cursor()

        updates = []
        params = []

        if satisfaction_rating is not None:
            updates.append("satisfaction_rating = ?")
            params.append(satisfaction_rating)

        if query_success is not None:
            updates.append("query_success = ?")
            params.append(query_success)

        if updates:
            query = f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?"
            params.append(conversation_id)
            cursor.execute(query, params)
            conn.commit()
    except sqlite3.Error as e:
        print(f"Database error in update_conversation_feedback: {e}")
    finally:
        if 'conn' in locals():
            conn.close()