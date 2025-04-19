"""
Script to set up the React frontend.
"""
import os
import subprocess
import sys
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger("frontend_setup", "frontend_setup.log")

def setup_frontend():
    """Set up the React frontend."""
    # Check if Node.js is installed
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Node.js or npm not found. Please install Node.js and npm.")
        return False
    
    # Create frontend directory
    frontend_dir = Path("frontend")
    os.makedirs(frontend_dir, exist_ok=True)
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    
    # Initialize React app with Vite
    logger.info("Initializing React app with Vite...")
    subprocess.run(["npm", "create", "vite@latest", ".", "--", "--template", "react-ts"], check=True)
    
    # Install dependencies
    logger.info("Installing dependencies...")
    subprocess.run(["npm", "install"], check=True)
    
    # Install additional dependencies
    logger.info("Installing additional dependencies...")
    subprocess.run([
        "npm", "install",
        "axios",                  # HTTP client
        "react-router-dom",       # Routing
        "react-markdown",         # Markdown rendering
        "react-syntax-highlighter", # Code highlighting
        "react-icons",            # Icons
        "tailwindcss",            # CSS framework
        "postcss",                # CSS processing
        "autoprefixer",           # CSS vendor prefixing
        "plotly.js",              # Visualization
        "react-plotly.js",        # React wrapper for Plotly
        "@headlessui/react",      # UI components
        "@tailwindcss/forms",     # Form styling
        "@tailwindcss/typography" # Typography styling
    ], check=True)
    
    # Initialize Tailwind CSS
    logger.info("Initializing Tailwind CSS...")
    subprocess.run(["npx", "tailwindcss", "init", "-p"], check=True)
    
    # Create Tailwind config
    with open("tailwind.config.js", "w") as f:
        f.write("""/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
""")
    
    # Update index.css
    with open("src/index.css", "w") as f:
        f.write("""@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  font-weight: 400;
  
  color-scheme: light dark;
  color: rgba(255, 255, 255, 0.87);
  background-color: #242424;
  
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

@media (prefers-color-scheme: light) {
  :root {
    color: #213547;
    background-color: #ffffff;
  }
}
""")
    
    # Create API client
    os.makedirs("src/api", exist_ok=True)
    with open("src/api/client.ts", "w") as f:
        f.write("""import axios from 'axios';

const API_URL = 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface QueryRequest {
  query: string;
  session_id?: string;
  response_type?: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  include_context?: boolean;
}

export interface SessionRequest {
  session_id: string;
}

export interface ContextItem {
  title: string;
  authors: string[];
  content: string;
  paper_id: string;
  chunk_id: string;
}

export interface Response {
  response: string;
  context?: ContextItem[];
  session_id: string;
}

export interface SessionList {
  sessions: string[];
}

export const api = {
  query: async (request: QueryRequest): Promise<Response> => {
    const response = await apiClient.post<Response>('/query', request);
    return response.data;
  },
  
  chat: async (request: ChatRequest): Promise<Response> => {
    const response = await apiClient.post<Response>('/chat', request);
    return response.data;
  },
  
  clearConversation: async (request: SessionRequest): Promise<any> => {
    const response = await apiClient.post('/clear_conversation', request);
    return response.data;
  },
  
  deleteConversation: async (request: SessionRequest): Promise<any> => {
    const response = await apiClient.post('/delete_conversation', request);
    return response.data;
  },
  
  listConversations: async (): Promise<SessionList> => {
    const response = await apiClient.get<SessionList>('/list_conversations');
    return response.data;
  },
};

export default api;
""")
    
    # Create components directory
    os.makedirs("src/components", exist_ok=True)
    
    # Return to original directory
    os.chdir("..")
    
    logger.info("Frontend setup complete!")
    logger.info("You can now run the frontend with: npm run dev")
    
    return True

if __name__ == "__main__":
    setup_frontend()
