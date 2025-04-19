# arXiv Research Assistant

A chatbot that serves as an expert in computer science using the arXiv dataset. This system implements a complete RAG (Retrieval-Augmented Generation) pipeline with hybrid retrieval, Phi-2 integration, and BAAI/bge-small-en-v1.5 embeddings.

## Features

- **Data Pipeline**: Extract, clean, chunk, and embed scientific papers from arXiv
- **Hybrid Retrieval**: Combine dense (semantic) and sparse (BM25) retrieval with re-ranking
- **LLM Integration**: Use Phi-2 with GGUF quantization for efficient inference via llama.cpp
- **Advanced Capabilities**: Summarization, concept explanation, and multi-turn reasoning
- **React Interface**: Paper search filters, conversation history, and concept visualization

## Architecture

The system consists of the following components:

1. **Data Pipeline**
   - arXiv dataset acquisition and filtering
   - PDF extraction and text cleaning
   - Document chunking and embedding generation
   - FAISS vector database integration

2. **RAG System**
   - Phi-2 integration via llama.cpp with GGUF quantization
   - Hybrid retrieval (dense + sparse)
   - Result re-ranking
   - Response generation with different prompt templates

3. **API Backend**
   - FastAPI server for handling requests
   - Conversation management
   - Session handling

4. **React Frontend**
   - Chat interface
   - Paper search and filtering
   - Visualization components

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Node.js and npm (for frontend)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/arxiv-research-assistant.git
   cd arxiv-research-assistant
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the Phi-2 model:
   ```
   python download_model.py
   ```

   You can also see available model variants:
   ```
   python download_model.py --list
   ```

   And choose a specific variant:
   ```
   python download_model.py --model phi-2-q5_k_m
   ```

4. Download the embedding model:
   ```
   python download_embedding_model.py
   ```

   Or manually clone the repository:
   ```
   cd models
   git clone https://huggingface.co/BAAI/bge-small-en-v1.5
   cd ..
   ```

5. Run the data pipeline to download and process papers:
   ```
   python -m data.pipeline
   ```

6. Set up the frontend:
   ```
   cd frontend
   npm install
   cd ..
   ```

## Usage

### Running the System

1. Start the API server:
   ```
   python main.py --api
   ```

2. Start the frontend development server:
   ```
   python main.py --frontend
   ```

3. Or run both at once:
   ```
   python main.py
   ```

### Configuration

You can configure the system by editing the `.env` file. Key configuration options include:

- `MAX_PAPERS`: Maximum number of papers to download
- `EMBEDDING_MODEL`: Model to use for embeddings
- `EMBEDDING_DEVICE`: Device to run embeddings on (`cuda` or `cpu`)
- `LLM_MODEL_PATH`: Path to the Phi-2 model
- `NUM_DOCUMENTS`: Number of documents to retrieve for each query
- `HYBRID_ALPHA`: Weight for hybrid search (0 = BM25 only, 1 = Vector only)

## Development

### Project Structure

```
arxiv-research-assistant/
├── api/                # FastAPI backend
├── data/               # Data pipeline components
├── frontend/           # React frontend
├── models/             # LLaMA model storage
├── rag/                # RAG system components
├── utils/              # Utility functions
├── .env                # Configuration file
├── download_model.py   # Script to download LLaMA model
├── main.py             # Main script to run the system
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### Adding New Features

- **New Paper Categories**: Edit `ARXIV_CATEGORIES` in `utils/config.py`
- **Custom Prompts**: Add new templates in `rag/prompts.py`
- **Additional Retrieval Methods**: Extend `rag/retriever.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [arXiv](https://arxiv.org/) for providing access to research papers
- [Phi-2](https://huggingface.co/microsoft/phi-2) by Microsoft
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient model inference
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [LangChain](https://github.com/langchain-ai/langchain) for RAG components
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) for embeddings