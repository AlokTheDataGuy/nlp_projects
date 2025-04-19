# Multilingual Chatbot - Comprehensive Guide

This guide provides detailed instructions for setting up and running the multilingual chatbot with FastAPI backend and React frontend.

## System Requirements

- Python 3.9+
- Node.js 14+ and npm
- CUDA-compatible GPU (optional but recommended)
- 16GB RAM

## Setup Instructions

### 1. Python Environment Setup

First, set up the Python environment:

```bash
# Create a conda environment (recommended)
conda create -n multilingual-chatbot python=3.9
conda activate multilingual-chatbot

# Or use venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Required Models

Download the necessary models:

```bash
# Download both fastText and NLLB models
python download_models.py --all

# Or download them separately
python download_models.py --fasttext
python download_models.py --nllb
```

If you encounter issues with automatic downloads, you can manually download:

- NLLB-200 model: `git clone https://huggingface.co/facebook/nllb-200-distilled-600M`
- fastText language detection model: Download from [fastText website](https://fasttext.cc/docs/en/language-identification.html)

### 3. Frontend Setup

Set up the React frontend:

```bash
# Run the setup script to create directory structure
python setup_frontend.py

# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Return to project root
cd ..
```

## Running the Chatbot

### Option 1: Interactive Console Mode

For a simple console-based interaction:

```bash
python main.py --mode interactive
```

Commands in interactive mode:
- Type `switch <language_code>` to switch language (e.g., `switch hin_Deva`)
- Type `clear` to clear the conversation history
- Type `exit`, `quit`, or `bye` to exit

### Option 2: FastAPI Backend + React Frontend

For a full web application experience:

#### Terminal 1 - Start the FastAPI backend:

```bash
# Start with auto-reload enabled (for development)
python run_api.py --reload

# Or for production
python run_api.py
```

The API will be available at:
- API endpoints: `http://localhost:5000/api/...`
- API documentation: `http://localhost:5000/docs`

#### Terminal 2 - Start the React frontend:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /api/chat` - Send a message to the chatbot
  ```json
  {
    "message": "Hello, how are you?",
    "session_id": "optional-session-id",
    "language": "eng_Latn"
  }
  ```

- `POST /api/switch-language` - Switch the conversation language
  ```json
  {
    "language": "hin_Deva",
    "session_id": "your-session-id"
  }
  ```

- `POST /api/clear-conversation` - Clear the conversation history
  ```json
  {
    "session_id": "your-session-id"
  }
  ```

- `GET /api/languages` - Get supported languages

## Supported Languages

- English (`eng_Latn`)
- Hindi (`hin_Deva`)
- Bengali (`ben_Beng`)
- Marathi (`mar_Deva`)

## Troubleshooting

### Model Loading Issues

If you encounter issues loading the NLLB model:

1. Ensure the model files are in the correct location (`./nllb-200-distilled-600M/`)
2. Check that you have sufficient disk space and RAM
3. For GPU issues, try running with CPU by modifying `config.py`:
   ```python
   TRANSLATION = {
       "model_name": "facebook/nllb-200-distilled-600M",
       "local_model_path": "./nllb-200-distilled-600M",
       "device": "cpu",  # Change this from "cuda" to "cpu"
       "quantization": False,  # Disable quantization for CPU
       "max_length": 128,
       "batch_size": 8
   }
   ```

### Frontend Connection Issues

If the frontend cannot connect to the backend:

1. Ensure the backend is running on port 5000
2. Check that the proxy settings in `vite.config.js` are correct
3. Look for CORS errors in the browser console

## Performance Optimization

For better performance:

1. Use a GPU with at least 4GB VRAM
2. Enable 8-bit quantization (default in `config.py`)
3. Adjust batch size and max length in `config.py` based on your hardware
4. The translation cache will improve performance for repeated phrases
