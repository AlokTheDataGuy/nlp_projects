# Enhanced Medical Q&A Chatbot

An advanced, LLM-enhanced medical Q&A chatbot that provides accurate, factual responses based on the MedQuAD dataset.

## Features

- **Dual LLM Architecture**: Uses LLaMA 3.1 for understanding queries and Meditron 7B for medical answers
- **Query Rephrasing**: LLaMA 3.1 interprets and rephrases user queries for better dataset search
- **Dataset Search**: Searches MedQuAD dataset for relevant answers with high accuracy
- **Answer Formatting**: Formats dataset answers to be concise and user-friendly
- **Meditron Fallback**: Uses Meditron 7B for direct answers when dataset search fails
- **Concise Responses**: All answers are kept short and to the point
- **Related Questions**: Suggests relevant follow-up questions for continued exploration

## Architecture

```
                          ┌──────────────────────┐
                          │   User Input (UI)    │
                          │          │
                          └─────────┬────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────┐
                    │  Direct Search in FAISS  │
                    │    (Fast First Pass)     │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │   Good Match Found?      │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  Common Medical Topic?   │◄─────────────┐
                    └──────────┬───────────────┘              │
                               │                              │
                               ▼                              │
                ┌──────────────────────────────┐              │
                │  Meditron 7B Direct Answer   │              │
                └──────────┬───────────────────┘              │
                           │                                  │
                           ▼                                  │
         ┌───────────────────────────────────────┐            │
         │ LLM Query Rephrasing (Meditron 7B)    │            │
         └──────────────────┬────────────────────┘            │
                            ▼                                 │
              ┌──────────────────────────────┐                │
              │    Advanced FAISS Search     │                │
              └──────────────┬───────────────┘                │
                             │                                │
                             ▼                                │
              ┌──────────────────────────────┐                │
              │ LLM-based Reranking          │                │
              │      (Meditron 7B)           │                │
              └──────────────┬───────────────┘                │
                             │                                │
                             ▼                                │
              ┌──────────────────────────────┐                │
              │ Return Best Answer to UI     │◄───────────────┘
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Generate Related Questions   │
              │      (Meditron 7B)           │
              └──────────────────────────────┘
```

## Requirements

- Python 3.8+
- Flask and Flask-CORS
- Pandas and NumPy
- Sentence Transformers
- FAISS
- LangChain
- Ollama with Meditron 7B (specialized medical LLM)

## Installation

1. Install the required Python dependencies:

```bash
pip install flask flask-cors pandas numpy sentence-transformers faiss-cpu langchain pydantic
```

2. Install and run Ollama:

```bash
# Follow instructions at https://ollama.ai/ to install Ollama
# Then pull the Meditron 7B model
ollama pull meditron:7b
```

3. Make sure the MedQuAD dataset is processed and available at `data/processed/medquad_complete.csv`

## Usage

1. Start the Ollama server:

```bash
ollama serve
```

2. Run the enhanced chatbot:

```bash
python run_enhanced_chatbot.py
```

3. Open your browser and navigate to `http://localhost:5000`

4. Start asking medical questions!

## How It Works

1. **Fast First Pass**: The system first tries a direct search to find a quick match without using the LLM
2. **LLM for Common Questions**: For common medical topics, Meditron 7B provides direct, accurate answers
3. **Advanced Pipeline**: For complex queries, the system uses query rephrasing, advanced search, and reranking
4. **Query Rephrasing**: The user's question is analyzed and rephrased by Meditron 7B to improve retrieval
5. **Embedding and Retrieval**: The rephrased query is embedded and used to find similar questions in the MedQuAD dataset
6. **Reranking**: Meditron 7B evaluates each candidate answer for relevance to the original query
7. **Related Questions**: The system suggests follow-up questions related to the user's query

## Command-line Options

```bash
python run_enhanced_chatbot.py --help
```

Options:
- `--ollama-url`: URL for the Ollama API (default: http://localhost:11434)
- `--port`: Port to run the Flask app on (default: 5000)
- `--host`: Host to run the Flask app on (default: 0.0.0.0)
- `--debug`: Run Flask in debug mode

## Files

- `enhanced_chatbot.py`: Main implementation of the enhanced chatbot
- `run_enhanced_chatbot.py`: Script to run the chatbot
- `templates/enhanced_index.html`: HTML template for the chat interface

## Example Queries

Try asking questions like:
- "What are the symptoms of lung cancer?"
- "What are the symptoms of diabetes?"
- "How is Adult Acute Lymphoblastic Leukemia treated?"
- "What causes high blood pressure?"
- "What are the risk factors for heart disease?"
- "How is pneumonia diagnosed?"
- "What are the early signs of a stroke?"
