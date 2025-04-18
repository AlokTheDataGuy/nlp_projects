# LLM-based Medical Q&A Chatbot

A simplified, LLM-based medical Q&A chatbot that provides accurate, factual responses based on the MedQuAD dataset.

## Features

- Clean, minimalist design with light/dark mode support
- Simple chat interface for asking medical questions
- Responses grounded in the MedQuAD dataset
- Source citations for transparency
- Fast response times using FAISS similarity search

## Architecture

```
                          ┌──────────────────────┐
                          │   User Input (UI)    │
                          │          │ 
                          └─────────┬────────────┘
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
              ┌──────────────────────────────┐                │
              │ Return Best Answer to UI │◄────────┘ 
              └──────────────────────────────┘
```

## Installation

1. Make sure you have the required dependencies:

```bash
pip install flask flask-cors pandas numpy sentence-transformers faiss-cpu
```

2. Make sure the MedQuAD dataset is processed and available at `data/processed/medquad_complete.csv`

## Usage

1. Run the chatbot:

```bash
python run_llm_chatbot.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Start asking medical questions!

## How It Works

1. **Embedding**: The user's question is embedded using a Sentence Transformer model
2. **Similarity Search**: The embedding is used to find similar questions in the MedQuAD dataset using FAISS
3. **Response Generation**: The answer to the most similar question is returned, along with source information
4. **UI Display**: The answer is displayed in the chat interface, with source citations for transparency

## Files

- `llm_chatbot.py`: Main implementation of the chatbot
- `run_llm_chatbot.py`: Script to run the chatbot
- `templates/llm_index.html`: HTML template for the chat interface

## Customization

You can customize the chatbot by:

- Changing the embedding model in the `MedicalChatbot` class
- Adjusting the number of results returned in the `search` method
- Modifying the UI in the `llm_index.html` template
