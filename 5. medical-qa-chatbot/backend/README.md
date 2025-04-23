# Medical Q&A Chatbot Backend

This is the backend for the Medical Q&A Chatbot, which uses the MedQuAD dataset and Meditron LLM to answer medical questions.

## Features

- FastAPI-based REST API
- Vector-based semantic search with FAISS
- Medical entity recognition
- Integration with Meditron LLM via Ollama
- Hybrid retrieval approach (MedQuAD + Meditron)

## Project Structure

```
backend/
├── app/
│   ├── api/           # API endpoints
│   ├── core/          # Core functionality
│   ├── models/        # Data models
│   ├── services/      # Business logic
│   │   ├── entity_recognition.py  # Medical entity recognition
│   │   ├── llm_service.py         # Meditron LLM integration
│   │   ├── query_processor.py     # Query processing
│   │   └── retrieval_service.py   # Retrieval logic
│   └── utils/         # Utility functions
│       └── create_index.py        # FAISS index creation
├── requirements.txt   # Python dependencies
└── run.py             # Entry point
```

## Setup

1. Make sure you have Python 3.8+ installed
2. Install Ollama from https://ollama.ai/
3. Pull the Meditron model:
   ```
   ollama pull meditron:7b
   ```
4. Create a virtual environment:
   ```
   python -m venv venv
   ```
5. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
6. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
7. Download the spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```
8. Generate the FAISS index:
   ```
   python -m app.utils.create_index
   ```

## Running the Backend

1. Make sure Ollama is running:
   ```
   ollama serve
   ```
2. Start the backend server:
   ```
   python run.py
   ```
3. The API will be available at http://localhost:8000

## API Endpoints

- `GET /`: Welcome message
- `GET /api/health`: Health check endpoint
- `POST /api/ask`: Process a medical question and return relevant answers

### Example Request

```json
POST /api/ask
{
  "question": "What are the symptoms of diabetes?",
  "max_results": 3
}
```

### Example Response

```json
{
  "answers": [
    {
      "answer": "The symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, irritability, blurred vision, slow-healing sores, and frequent infections.",
      "source": "MedQuAD - https://www.example.com/diabetes",
      "confidence": 0.85,
      "entities": [
        {
          "text": "diabetes",
          "type": "diseases",
          "start": 21,
          "end": 29,
          "source": "custom"
        }
      ]
    }
  ],
  "entities_detected": [
    {
      "text": "diabetes",
      "type": "diseases",
      "start": 21,
      "end": 29,
      "source": "custom"
    }
  ]
}
```
