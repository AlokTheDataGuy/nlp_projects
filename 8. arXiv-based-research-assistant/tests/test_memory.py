"""
Tests for the memory module.
"""
import pytest
from datetime import datetime
import json
import tempfile
from pathlib import Path

from utils.memory import Message, Conversation, ConversationManager

def test_message():
    """Test Message class."""
    # Create message
    message = Message(role="user", content="Hello")
    
    # Check attributes
    assert message.role == "user"
    assert message.content == "Hello"
    assert isinstance(message.timestamp, datetime)
    
    # Test to_dict
    message_dict = message.to_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == "Hello"
    assert isinstance(message_dict["timestamp"], str)
    
    # Test from_dict
    new_message = Message.from_dict(message_dict)
    assert new_message.role == message.role
    assert new_message.content == message.content
    assert new_message.timestamp.isoformat() == message.timestamp.isoformat()

def test_conversation():
    """Test Conversation class."""
    # Create conversation
    conversation = Conversation(session_id="test_session")
    
    # Check attributes
    assert conversation.session_id == "test_session"
    assert conversation.messages == []
    assert conversation.max_messages == 10
    
    # Test add_message
    conversation.add_message("user", "Hello")
    conversation.add_message("assistant", "Hi there")
    
    assert len(conversation.messages) == 2
    assert conversation.messages[0].role == "user"
    assert conversation.messages[0].content == "Hello"
    assert conversation.messages[1].role == "assistant"
    assert conversation.messages[1].content == "Hi there"
    
    # Test get_history
    history = conversation.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Hello"
    
    # Test get_history as dict
    history_dict = conversation.get_history(as_dict=True)
    assert len(history_dict) == 2
    assert history_dict[0]["role"] == "user"
    assert history_dict[0]["content"] == "Hello"
    
    # Test clear
    conversation.clear()
    assert conversation.messages == []
    
    # Test to_dict and from_dict
    conversation.add_message("user", "Hello again")
    conversation_dict = conversation.to_dict()
    
    new_conversation = Conversation.from_dict(conversation_dict)
    assert new_conversation.session_id == conversation.session_id
    assert new_conversation.max_messages == conversation.max_messages
    assert len(new_conversation.messages) == len(conversation.messages)
    assert new_conversation.messages[0].role == conversation.messages[0].role
    assert new_conversation.messages[0].content == conversation.messages[0].content

def test_conversation_max_messages():
    """Test Conversation max_messages limit."""
    # Create conversation with small max_messages
    conversation = Conversation(session_id="test_session", max_messages=3)
    
    # Add more messages than max_messages
    for i in range(5):
        conversation.add_message("user", f"Message {i}")
    
    # Check that only the last max_messages are kept
    assert len(conversation.messages) == 3
    assert conversation.messages[0].content == "Message 2"
    assert conversation.messages[1].content == "Message 3"
    assert conversation.messages[2].content == "Message 4"

def test_conversation_manager():
    """Test ConversationManager class."""
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create conversation manager
        manager = ConversationManager(storage_dir=Path(temp_dir))
        
        # Test get_conversation (creates new conversation)
        conversation = manager.get_conversation("test_session")
        assert conversation.session_id == "test_session"
        assert conversation.messages == []
        
        # Add messages
        conversation.add_message("user", "Hello")
        conversation.add_message("assistant", "Hi there")
        
        # Test save_all
        manager.save_all()
        
        # Check that file was created
        assert (Path(temp_dir) / "test_session.json").exists()
        
        # Test get_conversation (loads existing conversation)
        conversation2 = manager.get_conversation("test_session")
        assert conversation2.session_id == "test_session"
        assert len(conversation2.messages) == 2
        assert conversation2.messages[0].role == "user"
        assert conversation2.messages[0].content == "Hello"
        
        # Test clear_conversation
        manager.clear_conversation("test_session")
        conversation3 = manager.get_conversation("test_session")
        assert conversation3.messages == []
        
        # Test list_conversations
        conversation.add_message("user", "New message")
        manager.save_all()
        
        # Create another conversation
        conversation4 = manager.get_conversation("another_session")
        conversation4.add_message("user", "Hello from another session")
        manager.save_all()
        
        sessions = manager.list_conversations()
        assert len(sessions) == 2
        assert "test_session" in sessions
        assert "another_session" in sessions
        
        # Test delete_conversation
        manager.delete_conversation("test_session")
        assert not (Path(temp_dir) / "test_session.json").exists()
        
        sessions = manager.list_conversations()
        assert len(sessions) == 1
        assert "another_session" in sessions
