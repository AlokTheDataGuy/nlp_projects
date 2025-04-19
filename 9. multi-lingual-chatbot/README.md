# Multi-Lingual Chatbot

A chatbot that supports English plus three Indian languages (Hindi, Bengali, and Marathi) with automatic language detection, seamless language switching, and culturally appropriate responses.

## Features

- **Automatic Language Detection**: Identifies which language the user is typing in using fastText
- **Seamless Language Switching**: Maintains conversation context across language switches
- **Cultural Adaptation**: Provides culturally appropriate responses for each language
- **Efficient Translation**: Uses lightweight NLLB-200 models optimized for limited hardware
- **Memory Efficient**: Designed to work with 4GB GPU and 16GB RAM constraints

## Architecture

### Core Components

1. **Language Detection Module**
   - Uses fastText's compressed language identification model
   - Pre-filters to just detect the 4 target languages

2. **Translation Engine**
   - Primary model: NLLB-200 Distilled 600M (8-bit quantized)
   - Two translation directions: Source→English and English→Source
   - Implements caching to avoid re-translating common phrases

3. **Core Chatbot Logic**
   - Keeps all NLU/dialogue management in English only
   - Simple intent classification and response generation
   - Context tracking across language switches

4. **Cultural Adaptation Layer**
   - Applies cultural adjustments to translations
   - Handles honorifics for each language
   - Formats dates, numbers, etc. according to local conventions

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional but recommended)
- 16GB RAM

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-lingual-chatbot.git
   cd multi-lingual-chatbot
   ```

2. Create a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate multilingual-chatbot
   ```

   Or install dependencies with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required models:
   ```bash
   python download_models.py --all
   ```

## Usage

### Interactive Mode

Run the chatbot in interactive console mode:

```bash
python main.py --mode interactive
```

Commands in interactive mode:
- Type `switch <language_code>` to switch language (e.g., `switch hin_Deva`)
- Type `clear` to clear the conversation history
- Type `exit`, `quit`, or `bye` to exit

### FastAPI Backend

Run the chatbot with a FastAPI backend:

```bash
python run_api.py --port 5000
```

This will start the FastAPI server with automatic API documentation at `http://localhost:5000/docs`.

### React Frontend

The chatbot includes a modern React frontend built with Vite:

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:5173`

The React frontend will automatically connect to the FastAPI backend running on port 5000.

## Supported Languages

- English (`eng_Latn`)
- Hindi (`hin_Deva`)
- Bengali (`ben_Beng`)
- Marathi (`mar_Deva`)

## Configuration

Configuration settings can be modified in `config.py`:

- Language detection settings
- Translation model settings
- Cache settings
- Cultural adaptation rules
- API settings

## Performance Optimization

- Uses 8-bit quantization for the NLLB model when GPU is available
- Implements caching for translations to improve response time
- Optimizes memory usage for limited hardware

## License

MIT
