from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.chatbot import Chatbot
from app.services.conversation_manager import ConversationManager
from app.models.message import Message, MessageResponse
from typing import List, Dict
import uuid

app = FastAPI(title="Sentiment-Based Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize services
sentiment_analyzer = SentimentAnalyzer()
chatbot = Chatbot()
conversation_manager = ConversationManager()

# Store active connections
active_connections: Dict[str, WebSocket] = {}

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment-Based Chatbot API"}

@app.post("/chat", response_model=MessageResponse)
async def chat(message: Message):
    # Process the message
    conversation_id = message.conversation_id or str(uuid.uuid4())
    
    # Analyze sentiment
    sentiment_result = sentiment_analyzer.analyze(message.content)
    
    # Get chatbot response
    response = chatbot.generate_response(
        message.content, 
        sentiment_result, 
        conversation_manager.get_conversation_history(conversation_id)
    )
    
    # Update conversation history
    conversation_manager.add_message(
        conversation_id, 
        "user", 
        message.content, 
        sentiment_result
    )
    conversation_manager.add_message(
        conversation_id, 
        "bot", 
        response
    )
    
    # Check for escalation
    should_escalate = conversation_manager.check_escalation(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "response": response,
        "sentiment": sentiment_result,
        "escalate": should_escalate
    }

@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    await websocket.accept()
    active_connections[conversation_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Analyze sentiment
            sentiment_result = sentiment_analyzer.analyze(message_data["content"])
            
            # Get chatbot response
            response = chatbot.generate_response(
                message_data["content"], 
                sentiment_result, 
                conversation_manager.get_conversation_history(conversation_id)
            )
            
            # Update conversation history
            conversation_manager.add_message(
                conversation_id, 
                "user", 
                message_data["content"], 
                sentiment_result
            )
            conversation_manager.add_message(
                conversation_id, 
                "bot", 
                response
            )
            
            # Check for escalation
            should_escalate = conversation_manager.check_escalation(conversation_id)
            
            # Send response
            await websocket.send_json({
                "response": response,
                "sentiment": sentiment_result,
                "escalate": should_escalate
            })
    except WebSocketDisconnect:
        del active_connections[conversation_id]

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    return conversation_manager.get_conversation_history(conversation_id)

@app.get("/metrics")
async def get_metrics():
    return conversation_manager.get_metrics()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
