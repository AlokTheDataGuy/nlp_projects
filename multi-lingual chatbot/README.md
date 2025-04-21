# Multilingual Chatbot

A unique multilingual chatbot that supports English, Hindi, Bengali, and Marathi with automatic language detection, translation, and transliteration capabilities.

![Multilingual Chatbot Demo](screenshot.png)
*Screenshot of the chatbot interface*

## Features

- **Multilingual Support**: Communicate in English, Hindi, Bengali, and Marathi
- **Automatic Language Detection**: Automatically detects the language of user input
- **Neural Machine Translation**: Uses AI4Bharat's IndicBART for high-quality translations
- **Transliteration**: Type in Latin script and get responses in native scripts (Devanagari, Bengali)
- **Responsive UI**: Works on desktop and mobile devices
- **Conversation History**: Maintains conversation context across language switches
- **Automatic Script Conversion**: Intelligently converts between Latin and native scripts

## Architecture

The application consists of two main components:

1. **Backend**: FastAPI-based Python server that handles:
   - Language detection using fastText
   - LLM integration with LLaMA3.1-8B via Ollama
   - Translation using IndicBART
   - Transliteration using IndicXlit

2. **Frontend**: React-based web interface that provides:
   - Clean, intuitive chat interface
   - Real-time language detection display
   - Session management

![Architecture Diagram](docs/images/architecture.png)
*Screenshot placeholder: Replace with an architecture diagram*

## Installation

### Prerequisites

- Python 3.8+
- Ollama (for running LLaMA3.1-8B locally)

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-lingual-chatbot.git
   cd multi-lingual-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Download required models:
   ```bash
   python download_models.py
   ```

5. Start the backend server:
   ```bash
   python main.py
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Start typing messages in any supported language
3. The chatbot will automatically detect the language and respond accordingly
4. You can type in Latin script (e.g., "namaste") and the system will transliterate to the appropriate script (e.g., "नमस्ते")

![Language Detection Demo](docs/images/language-detection.png)
*Screenshot placeholder: Replace with a screenshot showing language detection*

![Transliteration Demo](docs/images/transliteration.png)
*Screenshot placeholder: Replace with a screenshot showing transliteration*

## Supported Languages

| Language | Code | Script |
|----------|------|--------|
| English  | eng_Latn | Latin |
| Hindi    | hin_Deva | Devanagari |
| Bengali  | ben_Beng | Bengali |
| Marathi  | mar_Deva | Devanagari |

## Technical Details

### Language Detection

The system uses fastText's language identification model to automatically detect the language of user input. This model supports 176 languages and is highly accurate even for short text.

### Translation

Translation is handled by AI4Bharat's IndicBART, a multilingual sequence-to-sequence model specifically designed for Indian languages. It provides high-quality translations between English and various Indian languages.

### Transliteration

The system uses AI4Bharat's IndicXlit for transliteration, which allows users to type in Latin script and get responses in native scripts. This makes it easier for users who are more comfortable typing in English but want to communicate in their native language.

### LLM Integration

The chatbot uses LLaMA3.1-8B via Ollama for generating responses. This provides a powerful foundation for understanding and generating text in multiple languages.

## Project Structure

```
multi-lingual-chatbot/
├── backend/
│   ├── config.py                    # Configuration settings
│   ├── download_models.py           # Script to download required models
│   ├── indicxlit_transliteration.py # Transliteration module using IndicXlit
│   ├── IndicXlit-v1.0/              # IndicXlit library for transliteration
│   ├── language_detection.py        # Language detection module using fastText
│   ├── llm.py                       # LLaMA integration via Ollama
│   ├── main.py                      # FastAPI application
│   ├── models/                      # Directory for downloaded models
│   ├── models.py                    # Pydantic models for API
│   ├── requirements.txt             # Python dependencies
│   └── translation.py               # Translation module using IndicBART
├── frontend/
│   ├── public/                      # Static files
│   ├── src/                         # React source code
│   │   ├── App.jsx                  # Main React component
│   │   ├── App.css                  # Styling
│   │   └── index.js                 # Entry point
│   ├── package.json                 # Node.js dependencies
│   └── README.md                    # Frontend documentation
└── README.md                        # Project documentation
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Errors**:
   - Make sure Ollama is running (`ollama serve`)
   - Verify that LLaMA3.1-8B is downloaded (`ollama pull llama3.1:8b`)

2. **Model Download Issues**:
   - Check your internet connection
   - Ensure you have sufficient disk space
   - Try running `download_models.py` again

3. **Slow Response Times**:
   - LLaMA3.1-8B requires significant computational resources
   - Consider using a machine with a GPU for better performance
   - Try shorter prompts if responses are timing out

## Future Improvements

- Session Management: Maintain language preference throughout the conversation
- Error Handling: Gracefully handle misidentified languages or translation errors
- Performance Optimization: Consider batch processing or caching for efficiency
- User Feedback Loop: Allow users to correct language issues
- Fallback Mechanisms: Have fallbacks if specific language features fail

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [AI4Bharat](https://ai4bharat.org/) for IndicBART and IndicXlit
- [Meta AI](https://ai.meta.com/) for LLaMA3 and fastText
- [Ollama](https://ollama.ai/) for making LLMs accessible locally
