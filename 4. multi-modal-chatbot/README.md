# Multi-Modal Chatbot

A chatbot that can understand image inputs, generate relevant images, and seamlessly integrate visual and textual information in conversations.

## Features

- Text-based conversation with AI
- Image understanding and analysis
- Image generation based on text prompts
- Seamless integration of text and visual content

## Architecture

The chatbot follows this flow:

1. User sends input (text, image, or both) via the UI
2. API Gateway validates the request and passes it to the Input Classifier
3. Input Classifier determines required processing types and sends to Task Dispatcher
4. Task Dispatcher creates specific tasks and sends them to appropriate models
5. Models process their assigned tasks
6. Response Aggregator collects all outputs and creates a coherent response
7. API Gateway returns the unified response to the UI

## Models Used

- **Image Understanding**: llama3.2-vision:11b (via Ollama)
- **Text Processing/Generation**: qwen2.5:7b (via Ollama)
- **Image Generation**: stabilityai/stable-diffusion-3-medium-diffusers (via Hugging Face)

## Prerequisites

- Python 3.9+
- Node.js 16+
- [Ollama](https://ollama.ai/) installed locally
- GPU recommended for faster image generation

## Setup

### 1. Install Ollama and required models

Download and install Ollama from [ollama.ai](https://ollama.ai/), then pull the required models:

```bash
ollama pull llama3.2-vision:11b
ollama pull qwen2.5:7b
```

### 2. Set up the backend

```bash
# Install dependencies
pip install -r requirements.txt

# Run the backend server
cd backend
python app.py
```

The backend server will run at http://localhost:8000

### 3. Set up the frontend

```bash
# Install dependencies
cd frontend
npm install

# Run the development server
npm run dev
```

The frontend will be available at http://localhost:3000

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Type a message or upload an image (or both)
3. The chatbot will process your input and respond accordingly
4. You can ask the chatbot to generate images by using phrases like "show me", "create an image of", etc.

## Project Structure

```
multi-modal-chatbot/
├── backend/
│   ├── app.py                 # FastAPI main application
│   ├── models/
│   │   ├── image_processor.py # llama3.2-vision:11b integration
│   │   ├── text_processor.py  # qwen2.5:7b integration
│   │   └── image_generator.py # stable-diffusion-3-medium integration
│   ├── services/
│   │   ├── input_classifier.py
│   │   ├── task_dispatcher.py
│   │   └── response_aggregator.py
│   └── utils/
│       └── helpers.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── MessageList.jsx
│   │   │   ├── MessageItem.jsx
│   │   │   ├── ImageUpload.jsx
│   │   │   └── ImageDisplay.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
└── README.md
```

## Notes

- The first image generation might take some time as the model needs to be loaded into memory
- For optimal performance, a GPU with at least 8GB of VRAM is recommended
- The chatbot stores uploaded images in the `uploads` directory, which are automatically cleaned up after 24 hours
