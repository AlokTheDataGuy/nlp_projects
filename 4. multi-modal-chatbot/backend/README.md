# Multi-Modal Chatbot Backend

This is the backend for the multi-modal chatbot, which handles text processing and image understanding.

## Models Used

- **Text Processing**: qwen2.5:7b (via Ollama)
- **Image Understanding**: llama3.2-vision:11b (via Ollama)

## Running the Backend

To run the backend server:

```bash
# Navigate to the backend directory
cd backend

# Run the server
python run.py
```

The server will start on http://localhost:8000

## API Endpoints

- **GET /** - Health check endpoint
- **POST /chat** - Main chat endpoint that handles text and image inputs

## Requirements

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

Also, ensure that Ollama is installed and the required models are pulled:

```bash
ollama pull qwen2.5:7b
ollama pull llama3.2-vision:11b
```
