# arXiv-based Research Assistant

A domain-specific (Computer Science) expert chatbot using arXiv papers as a knowledge source, powered by fine-tuned Mistral 7B and retrieval-augmented generation.

## Overview

This project creates an AI assistant that can answer questions about Computer Science research by leveraging arXiv papers. It uses a combination of:

- **Fine-tuned Mistral 7B**: A powerful language model fine-tuned on arXiv papers
- **Retrieval-Augmented Generation (RAG)**: Enhances responses with relevant information from arXiv papers
- **Vector Search**: Efficiently finds relevant papers based on semantic similarity
- **Document Processing**: Extracts and processes information from research papers

## Key Components

1. **Data Pipeline**
   - Paper Processor: Handles PDF extraction from arXiv dataset
   - Text Extractor: Cleans and segments papers into sections
   - Feature Generator: Identifies key entities, concepts, and relationships
   - Embedding Creator: Generates embeddings using Sentence-Transformers' all-mpnet-base-v2

2. **Knowledge Base**
   - Faiss: Open-source vector database for storing embeddings
   - MongoDB Document Store: Stores paper metadata, content, and relationships
   - Knowledge Graph: Optional component for concept relationships and domain taxonomy

3. **Inference Engine**
   - Fine-tuned Mistral 7B: LLM model fine-tuned on arXiv papers
   - RAG System: Retrieval-augmented generation using LangChain
   - Context Manager: Handles context window optimization and document chunking
   - Response Generator: Formats final responses with citations and explanations

4. **Query Processor**
   - Query Analyzer: Classifies questions and extracts key entities
   - Conversation Manager: Maintains dialog context and handles follow-up questions
   - Search Engine: Provides semantic and keyword search functionality

5. **User Interface**
   - Chat Interface: Main interaction point for questions and answers
   - Search Interface: Allows direct paper searching and browsing
   - Visualization Panel: Displays concept relationships, paper networks, etc.

## Technical Stack

| Component | Technology |
|-----------|------------|
| Base Language | Python 3.10+ |
| LLM | Mistral 7B (fine-tuned) |
| Embeddings | Sentence-Transformers all-mpnet-base-v2 |
| Vector Database | Faiss |
| Document Store | MongoDB |
| Backend API | FastAPI |
| Frontend | React + TypeScript |
| Visualization | D3.js + React-Force-Graph |
| Deployment | Docker + Docker Compose |
| RAG Framework | LangChain |

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- GPU with at least 8GB VRAM (for fine-tuning)
- MongoDB
- Node.js and npm (for UI development)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arXiv-based-research-assistant.git
   cd arXiv-based-research-assistant
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download arXiv papers:
   ```bash
   python scripts/download_papers.py --limit 100 --store
   ```

4. Process papers and generate embeddings:
   ```bash
   python scripts/process_papers.py --store
   ```

5. Fine-tune the Mistral 7B model on arXiv papers:
   ```bash
   bash scripts/run_finetune.sh
   ```
   This will fine-tune the model on the downloaded papers and save it to `models/mistral-7b-arxiv-finetuned`.

6. Start the backend server:
   ```bash
   python scripts/run_server.py
   ```

7. In a separate terminal, start the frontend development server:
   ```bash
   cd ui
   npm install
   npm run dev
   ```

8. Access the UI at http://localhost:5173

## Usage

1. Ask research questions through the chat interface
2. Search for specific papers or topics
3. Explore related concepts and papers through the visualization panel

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- arXiv for providing access to research papers
- Mistral AI for developing the Mistral 7B model
- The open-source community for tools like LangChain, Faiss, and more
