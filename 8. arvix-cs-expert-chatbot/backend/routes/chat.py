from fastapi import APIRouter, HTTPException
from datetime import datetime

from schemas.chat import MessageRequest, MessageResponse, ConversationModel, ConversationListResponse
from services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.post("/message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """
    Send a message to the chatbot and get a response
    """
    try:
        response = await chat_service.process_message(
            user_id=request.user_id,
            message=request.message
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.get("/conversations/{user_id}", response_model=ConversationListResponse)
async def get_conversations(user_id: str):
    """
    Get all conversations for a user
    """
    # This is a simplified implementation - in a real app, this would use a database
    conversations = []

    # Get conversations from chat service
    user_conversations = chat_service.conversations.get(user_id, [])

    # Group by conversation ID
    conversation_map = {}
    for message in user_conversations:
        conv_id = message.get("conversation_id")
        if conv_id not in conversation_map:
            # Create a new conversation
            conversation_map[conv_id] = {
                "id": conv_id,
                "user_id": user_id,
                "title": message.get("user", "New conversation")[:30] + "...",  # Use first user message as title
                "created_at": datetime.fromisoformat(message.get("timestamp")),
                "updated_at": datetime.fromisoformat(message.get("timestamp")),
                "messages": []
            }

        # Add message to conversation
        conversation_map[conv_id]["messages"].append({
            "user": message.get("user"),
            "assistant": message.get("assistant"),
            "timestamp": message.get("timestamp")
        })

        # Update the updated_at timestamp
        conversation_map[conv_id]["updated_at"] = datetime.fromisoformat(message.get("timestamp"))

    # Convert to list
    conversations = list(conversation_map.values())

    return ConversationListResponse(conversations=conversations)

@router.get("/conversation/{conversation_id}", response_model=ConversationModel)
async def get_conversation(conversation_id: str):
    """
    Get a specific conversation by ID
    """
    # Search for the conversation in all users
    for user_id, user_conversations in chat_service.conversations.items():
        for message in user_conversations:
            if message.get("conversation_id") == conversation_id:
                # Found the conversation, now collect all messages
                messages = []
                for msg in user_conversations:
                    if msg.get("conversation_id") == conversation_id:
                        messages.append({
                            "user": msg.get("user"),
                            "assistant": msg.get("assistant"),
                            "timestamp": msg.get("timestamp")
                        })

                # Create the conversation model
                conversation = ConversationModel(
                    id=conversation_id,
                    user_id=user_id,
                    title=messages[0].get("user", "New conversation")[:30] + "...",
                    created_at=datetime.fromisoformat(messages[0].get("timestamp")),
                    updated_at=datetime.fromisoformat(messages[-1].get("timestamp")),
                    messages=messages
                )

                return conversation

    # If we get here, the conversation was not found
    raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation
    """
    deleted = False

    # Search for the conversation in all users
    for user_id, user_conversations in chat_service.conversations.items():
        # Filter out messages for this conversation
        new_conversations = [msg for msg in user_conversations if msg.get("conversation_id") != conversation_id]

        # If we removed any messages, update the user's conversations
        if len(new_conversations) < len(user_conversations):
            chat_service.conversations[user_id] = new_conversations
            deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

    return {"message": f"Conversation {conversation_id} deleted"}