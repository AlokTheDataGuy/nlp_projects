# Enhanced Medical Q&A Chatbot

An advanced, LLM-enhanced medical Q&A chatbot that provides accurate, factual responses based on the MedQuAD dataset.

## Features

- **Query Rephrasing**: Uses LLaMA 3 to interpret and rephrase user queries for better retrieval
- **Enhanced Retrieval**: Combines embedding similarity with text similarity for better results
- **LLM-based Reranking**: Uses LLaMA 3 to evaluate and rerank candidate answers
- **Answer Simplification**: Option to simplify complex medical answers into layman's terms
- **Source Transparency**: Shows sources with relevance scores and explanations
- **Modern UI**: Clean, responsive design with light/dark mode support

## Architecture

```
                          ┌──────────────────────┐
                          │   User Input (UI)    │
                          │          │ 
                          └─────────┬────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────┐
                    │  LLM Query Rephrasing    │
                    │      (LLaMA 3)           │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │    Sentence Embedding    │
                    │          Model           │
                    └──────────┬───────────────┘
                               ▼
                    ┌──────────────────────────┐
                    │    FAISS Vector Store    │◄─────────────┐
                    │ (Pre-built from MedQuAD) │              │
                    └──────────┬───────────────┘              │
                               ▼                              │
                ┌──────────────────────────────┐              │
                │ Retrieve Top-N QA Candidates │              │
                └──────────┬───────────────────┘              │
                           ▼                                  │
         ┌───────────────────────────────────────┐            │
         │ LLM-based Reranking (LLaMA 3)         │            │
         └──────────────────┬────────────────────┘            │
                            ▼                                 │
              ┌──────────────────────────────┐                │
              │ Return Best Answer to UI │◄────────┘ 
              └──────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────────┐
              │ Optional Answer Simplification│
              │        (LLaMA 3)             │
              └──────────────────────────────┘
```

## Requirements

- Python 3.8+
- Flask and Flask-CORS
- Pandas and NumPy
- Sentence Transformers
- FAISS
- LangChain
- Ollama with LLaMA 3 (or other compatible model)

## Installation

1. Install the required Python dependencies:

```bash
pip install flask flask-cors pandas numpy sentence-transformers faiss-cpu langchain pydantic
```

2. Install and run Ollama:

```bash
# Follow instructions at https://ollama.ai/ to install Ollama
# Then pull the LLaMA 3 model
ollama pull llama3
```

3. Make sure the MedQuAD dataset is processed and available at `data/processed/medquad_complete.csv`

## Usage

1. Run the enhanced chatbot:

```bash
python run_enhanced_chatbot.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Start asking medical questions!

## How It Works

1. **Query Rephrasing**: The user's question is analyzed and rephrased by LLaMA 3 to improve retrieval
2. **Embedding and Retrieval**: The rephrased query is embedded and used to find similar questions in the MedQuAD dataset
3. **Reranking**: LLaMA 3 evaluates each candidate answer for relevance to the original query
4. **Response Generation**: The most relevant answer is returned, along with source information
5. **Optional Simplification**: Users can request a simplified version of complex medical answers

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

## Customization

You can customize the chatbot by:
- Changing the LLM model in the `EnhancedMedicalChatbot` class
- Adjusting the prompts in the setup methods
- Modifying the UI in the `enhanced_index.html` template
