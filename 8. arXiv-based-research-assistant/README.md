# ArXiv Expert Chatbot

A chatbot that can discuss advanced topics in computer science, provide summaries of research papers, and explain complex concepts. The system can handle follow-up questions on complex topics and includes features for paper searching and concept visualization.

## System Architecture

The ArXiv Expert Chatbot follows a hybrid architecture with the following components:

### 1. Frontend Layer
- Web interface with chat functionality and visualization area
- Built with React, TypeScript, and Material-UI
- Features conversation history, paper reference display, and topic filtering

### 2. API Layer (FastAPI)
- Endpoints for chat interaction, paper search, and concept visualization
- Session management for multi-turn conversations

### 3. Core Processing
- Query Processor: Analyzes user queries, identifies key concepts, and determines query intent
- Response Generator: Aggregates information from multiple sources and formats responses

### 4. Knowledge Layer
- ArXiv API integration for real-time paper retrieval
- Local Knowledge Base (SQLite) for storing processed papers, summaries, and concepts
- Vector Store (FAISS) for semantic search capabilities
- Quantized LLM (Phi-3-mini-4k-instruct) for generating explanations

### 5. Background Processing
- Processing Queue for managing resource-intensive tasks
- Paper Processors for extracting concepts, generating summaries, and mapping relations

## Getting Started

### Prerequisites
- Python 3.8+ with conda environment
- Node.js 14+ and npm
- GPU with at least 4GB VRAM (recommended for LLM)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd arvix-bot
```

2. Set up the backend:
```bash
cd backend
# Activate your conda environment
conda activate arvix_bot
# Run the application
python run.py
```

3. Set up the frontend:
```bash
cd frontend
npm install
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

## Usage

### Chat Interface
- Ask questions about computer science research papers
- Request explanations of complex concepts
- Get summaries of research papers
- Follow up with related questions

### Paper Search
- Search for papers on specific topics
- View paper details including abstract, authors, and categories
- Access links to the original papers on ArXiv

### Concept Visualization
- Explore concepts extracted from papers
- View relationships between concepts
- Understand how concepts are connected

## Implementation Details

### LLM Integration
The system uses Phi-3-mini-4k-instruct (3.8B parameters) quantized to 4-bit precision to fit in 4GB VRAM. The model is loaded on demand to conserve resources.

### Vector Search
FAISS is used for efficient similarity search across paper content and concepts, enabling the system to find relevant information quickly.

### Resource Management
The system includes monitoring for RAM and GPU usage, with mechanisms to free resources when needed and process tasks in the background.

## Required Libraries

### Core Dependencies
- **FastAPI**: Web framework for building the API
- **SQLAlchemy**: ORM for database operations
- **FAISS**: Vector similarity search library
- **Transformers**: For working with the LLM
- **Sentence-Transformers**: For generating embeddings
- **Torch**: Deep learning framework with GPU support
- **BitsAndBytes**: For model quantization

### ArXiv Integration
- **ArXiv**: Python wrapper for the ArXiv API
- **PyMuPDF/pdf2image**: For processing PDF papers

### NLP and Processing
- **NLTK**: Natural language processing toolkit
- **Spacy**: Advanced NLP library
- **Scikit-learn**: For machine learning utilities

### Visualization
- **NetworkX**: For graph operations and concept relationships
- **Plotly/Matplotlib**: For data visualization

See `backend/requirements.txt` for the complete list with version specifications.

## License
[MIT License](LICENSE)
