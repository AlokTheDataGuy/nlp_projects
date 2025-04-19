# File: database/setup.py
# Setup and initialize the SQLite database

import sqlite3
import os

def setup_database():
    db_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'chatbot_analytics.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_query TEXT NOT NULL,
        bot_response TEXT NOT NULL,
        topic TEXT,
        satisfaction_rating INTEGER,
        query_success BOOLEAN
    )
    ''')
    
    # Create sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        end_time DATETIME,
        duration_seconds INTEGER
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized at {db_path}")
    return db_path