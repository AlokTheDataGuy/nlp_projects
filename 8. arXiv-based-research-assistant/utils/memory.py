"""
Memory management for conversation history.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from .logger import logger
from .config import BASE_DIR

@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

@dataclass
class Conversation:
    """Manages a conversation session with memory."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 10  # Default limit for conversation history
    
    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation."""
        message = Message(role=role, content=content)
        self.messages.append(message)
        
        # Trim conversation if it exceeds max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_history(self, as_dict: bool = False) -> List[Any]:
        """Get conversation history."""
        if as_dict:
            return [msg.to_dict() for msg in self.messages]
        return self.messages
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "max_messages": self.max_messages
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create conversation from dictionary."""
        conv = cls(
            session_id=data["session_id"],
            max_messages=data.get("max_messages", 10)
        )
        conv.messages = [Message.from_dict(msg) for msg in data["messages"]]
        return conv
    
    def save(self, directory: Optional[Path] = None) -> None:
        """Save conversation to file."""
        if directory is None:
            directory = BASE_DIR / "conversations"
        
        directory.mkdir(exist_ok=True)
        file_path = directory / f"{self.session_id}.json"
        
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.debug(f"Saved conversation to {file_path}")
    
    @classmethod
    def load(cls, session_id: str, directory: Optional[Path] = None) -> Optional['Conversation']:
        """Load conversation from file."""
        if directory is None:
            directory = BASE_DIR / "conversations"
        
        file_path = directory / f"{session_id}.json"
        
        if not file_path.exists():
            logger.debug(f"Conversation file {file_path} not found")
            return None
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            logger.debug(f"Loaded conversation from {file_path}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return None


class ConversationManager:
    """Manages multiple conversations."""
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or (BASE_DIR / "conversations")
        self.storage_dir.mkdir(exist_ok=True)
        self.active_conversations: Dict[str, Conversation] = {}
    
    def get_conversation(self, session_id: str) -> Conversation:
        """Get or create a conversation."""
        if session_id not in self.active_conversations:
            # Try to load from disk
            conv = Conversation.load(session_id, self.storage_dir)
            if conv is None:
                # Create new conversation
                conv = Conversation(session_id=session_id)
            
            self.active_conversations[session_id] = conv
        
        return self.active_conversations[session_id]
    
    def save_all(self) -> None:
        """Save all active conversations."""
        for conv in self.active_conversations.values():
            conv.save(self.storage_dir)
    
    def clear_conversation(self, session_id: str) -> None:
        """Clear a conversation's history."""
        if session_id in self.active_conversations:
            self.active_conversations[session_id].clear()
    
    def delete_conversation(self, session_id: str) -> None:
        """Delete a conversation."""
        if session_id in self.active_conversations:
            del self.active_conversations[session_id]
        
        # Delete file if it exists
        file_path = self.storage_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted conversation file {file_path}")
    
    def list_conversations(self) -> List[str]:
        """List all saved conversations."""
        return [f.stem for f in self.storage_dir.glob("*.json")]


# Create global conversation manager
conversation_manager = ConversationManager()
