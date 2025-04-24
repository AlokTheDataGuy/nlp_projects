# Automated Knowledge Expansion Chatbot

An AI-powered smart curator system that monitors top YouTube channels in AI, ML, and tech, extracts transcripts from newly published videos, and intelligently identifies and stores the most relevant, high-impact insights. A chatbot interface allows users to query curated insights by channel, topic, or timeframe — without watching the full videos.

## System Architecture

```
YouTube API → Scheduler → Transcript Extractor
       ↓
Content Analyzer → Insight Extractor → Metadata Enricher
       ↓
Vector DB ← → Metadata DB ← → Retention Manager
       ↑
API Gateway ← → Query Processor ← → Chatbot Interface
```

## Features

- Automatically monitors YouTube channels for new videos
- Extracts transcripts from videos
- Uses AI to identify key insights from video content
- Stores insights in a vector database for semantic search
- Provides a chatbot interface for querying insights
- Supports filtering by channel, topic, or timeframe
- Automatically manages retention of insights (default: 30 days)

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: React
- **Database**: MongoDB (metadata), Pinecone (vector database)
- **AI**: OpenAI GPT-4 for insight extraction and query processing
- **APIs**: YouTube Data API, YouTube Transcript API

## Setup

### Backend

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment:
   ```
   conda create -n youtube_curator python=3.10
   conda activate youtube_curator
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example` and fill in your API keys.

5. Run the server:
   ```
   uvicorn app.main:app --reload
   ```

### Frontend

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run the development server:
   ```
   npm run dev
   ```

## API Endpoints

- `/api/channels`: Manage monitored YouTube channels
- `/api/videos`: Access video information
- `/api/insights`: Query extracted insights
- `/api/chat`: Interact with the chatbot interface

## Example Usage

User: "What are the key takeaways from last Two Minute Papers video?"

🤖 Bot:

🧠 New Reinforcement Learning method cuts training time by 40%

📈 Google's Gemini 1.5 outperforms GPT-4 in long-context benchmarks

🧪 "This could lead to faster training for robotics," said the researcher

(Published 2 days ago | Watch Video)