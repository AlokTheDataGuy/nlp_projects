from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

class ConversationManager:
    def __init__(self):
        # Store conversations in memory (in production, use a database)
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}

        # Store metrics
        self.metrics = {
            "total_conversations": 0,
            "total_messages": 0,
            "sentiment_distribution": {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            },
            "escalations": 0
        }

        # Escalation threshold - if negative sentiment is detected in 3 consecutive messages
        self.escalation_threshold = 3

    def add_message(self, conversation_id: str, role: str, content: str, sentiment: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation history"""
        # Create conversation if it doesn't exist
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            self.metrics["total_conversations"] += 1

        # Create message object
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "sentiment": sentiment
        }

        # Add message to conversation
        self.conversations[conversation_id].append(message)
        self.metrics["total_messages"] += 1

        # Update sentiment metrics if sentiment is provided
        if sentiment and role == "user":
            sentiment_label = sentiment.get("label")
            if sentiment_label in self.metrics["sentiment_distribution"]:
                self.metrics["sentiment_distribution"][sentiment_label] += 1

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get the conversation history for a given conversation ID"""
        return self.conversations.get(conversation_id, [])

    def check_escalation(self, conversation_id: str) -> bool:
        """Check if a conversation should be escalated based on sentiment"""
        conversation = self.get_conversation_history(conversation_id)

        # Get user messages with sentiment
        user_messages = [
            msg for msg in conversation
            if msg["role"] == "user" and "sentiment" in msg and msg["sentiment"] is not None
        ]

        # Check if we have enough messages to analyze
        if len(user_messages) < 3:  # Need at least 3 messages for meaningful analysis
            return False

        # Get the last few messages for analysis
        recent_messages = user_messages[-5:] if len(user_messages) >= 5 else user_messages

        # Count negative messages
        negative_count = sum(
            1 for msg in recent_messages
            if msg["sentiment"].get("label") == "negative"
        )

        # Calculate negative percentage
        negative_percentage = negative_count / len(recent_messages)

        # Check negative sentiment confidence
        high_confidence_negative = sum(
            1 for msg in recent_messages
            if msg["sentiment"].get("label") == "negative" and
               msg["sentiment"].get("confidence", 0) > 0.7  # High confidence threshold
        )

        # Escalation criteria:
        # 1. Either 3 consecutive negative messages OR
        # 2. More than 60% of recent messages are negative OR
        # 3. At least 2 high-confidence negative messages

        # Check for consecutive negative messages
        consecutive_negative = 0
        max_consecutive = 0

        for msg in user_messages:
            if msg["sentiment"].get("label") == "negative":
                consecutive_negative += 1
                max_consecutive = max(max_consecutive, consecutive_negative)
            else:
                consecutive_negative = 0

        should_escalate = (
            max_consecutive >= self.escalation_threshold or
            negative_percentage >= 0.6 or
            high_confidence_negative >= 2
        )

        # Update metrics if escalating
        if should_escalate and conversation_id in self.conversations:
            # Only count new escalations
            already_escalated = any(msg.get("escalated", False) for msg in conversation)
            if not already_escalated:
                self.metrics["escalations"] += 1
                # Mark conversation as escalated
                for msg in self.conversations[conversation_id]:
                    msg["escalated"] = True

        return should_escalate

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about conversations and sentiment"""
        return self.metrics

    def get_sentiment_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get the sentiment history for a conversation"""
        conversation = self.get_conversation_history(conversation_id)

        # Extract sentiment data from user messages
        sentiment_history = [
            {
                "timestamp": msg["timestamp"],
                "sentiment": msg["sentiment"]
            }
            for msg in conversation
            if msg["role"] == "user" and "sentiment" in msg and msg["sentiment"] is not None
        ]

        return sentiment_history
