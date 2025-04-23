# Professional Internship Projects

This repository contains a collection of AI and NLP projects developed during my professional internship. Each project demonstrates different aspects of AI application development, from text summarization to multilingual chatbots.

## Projects Overview

### 1. Extractive Summarization Tool
An intelligent text summarization tool powered by BERT that automatically extracts the most important sentences from documents to create concise summaries.

![Summarization Tool](1.%20extractive-summarization_tool/screenshot.png)

**Key Technologies**: Flask, React, PyTorch, Hugging Face Transformers, NetworkX, PyPDF2, pdfplumber, TailwindCSS, Vite

### 2. Chatbot Analytics Dashboard
A comprehensive system for chatbot interactions with an analytics dashboard to track user engagement, satisfaction ratings, and conversation topics.

![Analytics Dashboard](2.%20chatbot-analytics-dashboard/screenshots/dashboard_2.png)

**Key Technologies**: Python, Ollama (Cogito:8b), Plotly, Dash, Pandas, SQLite, Chart.js, Bootstrap

### 3. Article Generator with 3 Different LLMs
A tool for generating articles using different open-source LLMs (Mistral, Qwen2.5, Llama3.1) and comparing their performance.

![Article Generator](3.%20article-generator-with-3-different-llms/screenshots/web_interface.png)

**Key Technologies**: Python, Ollama, Flask, NLTK, spaCy, Matplotlib, scikit-learn, TextBlob, Jinja2

### 4. Multi-Modal Chatbot
A chatbot that can understand both text and image inputs, providing seamless multi-modal interactions.

![Multi-Modal Chatbot](4.%20multi-modal-chatbot/Screenshots/Image_Understanding.png)

**Key Technologies**: FastAPI, React, Ollama (llama3.2-vision:11b, qwen2.5:7b), Pillow, Material-UI, Axios

### 5. Medical Q&A Chatbot
A medical question-answering chatbot that provides appropriate and relevant medical information using retrieval-based methods and LLMs.

![Medical Chatbot](5.%20medical-qa-chatbot/screenshot.png)

**Key Technologies**: FastAPI, React, Ollama (Meditron 7B, Llama 3.1 8B), MedQuAD Dataset, FAISS, Pandas, NumPy, Tailwind CSS, TypeScript

### 6. Knowledge-Based Auto-Updater
An AI-powered system that monitors YouTube channels, extracts transcripts, identifies key insights, and allows users to query this information through a chatbot interface.

**Key Technologies**: FastAPI, React, MongoDB, Pinecone, YouTube Data API, YouTube Transcript API, OpenAI GPT-4, LangChain, Pydantic

### 7. Chatbot Sentiment Analysis
A sophisticated chatbot with sentiment analysis capabilities to recognize and respond appropriately to customer emotions.

![Sentiment Analysis](7.%20chatbot-sentiment-analysis/screeenshots/chatbot.png)

**Key Technologies**: FastAPI, React, Hugging Face Transformers (RoBERTa), Ollama (Llama3.1:8b), Material-UI, Chart.js, Pydantic, WebSockets

### 8. arXiv-based Research Assistant
A domain-specific expert chatbot using arXiv papers as a knowledge source, powered by fine-tuned Mistral 7B and retrieval-augmented generation.

**Key Technologies**: Python, Mistral 7B, Faiss, MongoDB, FastAPI, React, Docker, LangChain, Sentence-Transformers, PyPDF2, D3.js, React-Force-Graph

### 9. Multi-Lingual Chatbot
A multilingual chatbot supporting English, Hindi, Bengali, and Marathi with automatic language detection, translation, and transliteration capabilities.

![Multi-Lingual Chatbot](9.%20multi-lingual%20chatbot/screenshot.png)

**Key Technologies**: FastAPI, React, fastText, Ollama (LLaMA3.1-8B), IndicBART, IndicXlit, styled-components, Pydantic, Uvicorn

## Technical Stack

The projects in this repository utilize a variety of technologies:

- **Backend Frameworks**: FastAPI, Flask, Uvicorn
- **Frontend**: React, TypeScript, Vite, Material-UI, TailwindCSS, styled-components
- **AI/ML**: PyTorch, Transformers, Hugging Face models, LangChain, NLTK, spaCy
- **LLMs**: Mistral, Llama, Qwen, Meditron (via Ollama), OpenAI GPT-4
- **Databases**: MongoDB, Faiss, Pinecone, SQLite
- **Vector Embeddings**: Sentence-Transformers, all-mpnet-base-v2
- **Visualization**: Chart.js, D3.js, Plotly, Matplotlib
- **Deployment**: Docker, Docker Compose
- **APIs**: YouTube Data API, YouTube Transcript API

## Skills Demonstrated

Through these projects, I've demonstrated proficiency in:

- **Large Language Model Integration**: Implementing and fine-tuning various LLMs for specific tasks
- **Retrieval-Augmented Generation (RAG)**: Creating systems that combine knowledge bases with generative AI
- **Multi-Modal AI**: Building applications that process both text and image inputs
- **Natural Language Processing**: Implementing text summarization, sentiment analysis, and language detection
- **Full-Stack Development**: Creating end-to-end applications with modern frontend and backend technologies
- **API Development**: Designing and implementing RESTful APIs
- **Database Design**: Working with both traditional and vector databases
- **Data Processing**: Extracting, transforming, and analyzing structured and unstructured data
- **UI/UX Design**: Creating intuitive user interfaces for complex AI applications
- **System Architecture**: Designing scalable, modular systems with clear separation of concerns

## Learning Outcomes

Through this internship, I've gained valuable experience and insights:

- **Practical AI Application**: Moving beyond theoretical knowledge to build real-world AI applications
- **Technical Integration**: Combining multiple technologies and frameworks into cohesive systems
- **Problem-Solving**: Addressing challenges in AI implementation, from model selection to deployment
- **Performance Optimization**: Balancing model accuracy with computational efficiency
- **User-Centered Design**: Creating AI systems that are accessible and useful to end-users
- **Ethical AI Development**: Considering privacy, bias, and ethical implications in AI applications
- **Project Management**: Planning, executing, and documenting complex technical projects
- **Continuous Learning**: Adapting to rapidly evolving AI technologies and methodologies

## Getting Started

Each project has its own README with specific setup instructions. Generally, the projects follow this pattern:

1. Set up a Python environment
2. Install dependencies from requirements.txt
3. Install Ollama and pull required models (for LLM-based projects)
4. Set up the backend server
5. Set up the frontend application

## License

This repository is for demonstration purposes. All projects are provided under the MIT License.
