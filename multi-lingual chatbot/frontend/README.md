# Multilingual Chatbot Frontend

A modern React frontend for the multilingual chatbot that supports English, Hindi, Bengali, and Marathi.

## Features

- Clean, responsive UI built with React and styled-components
- Real-time language switching
- Message history with visual distinction between user and bot messages
- Language detection display
- Session management

## Getting Started

### Prerequisites

- Node.js 14+ and npm

### Installation

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Project Structure

- `src/App.jsx` - Main application component
- `src/App.css` - Styles for the application
- `src/index.css` - Global styles
- `src/main.jsx` - Entry point

## API Integration

The frontend communicates with the FastAPI backend through the following endpoints:

- `POST /api/chat` - Send a message to the chatbot
- `POST /api/switch-language` - Switch the conversation language
- `POST /api/clear-conversation` - Clear the conversation history
- `GET /api/languages` - Get supported languages

## Configuration

The API base URL can be configured in `vite.config.js` by modifying the proxy settings.
