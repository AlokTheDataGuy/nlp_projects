# Medical Q&A Chatbot using MedQuAD dataset

A retrieval-based medical Q&A chatbot that takes a user's question, performs medical entity recognition using SciSpacy, embeds the question using Sentence Transformers, and retrieves the most relevant answer from the MedQuAD dataset using FAISS similarity search.

## Features

- Medical entity recognition using SciSpacy
- Question embedding using Sentence Transformers
- Efficient similarity search with FAISS
- Filtering and ranking based on semantic types and entity overlap
- Modern, responsive UI with light/dark mode

## Architecture

```
                          ┌──────────────────────┐
                          │   User Input (UI)    │
                          │          │
                          └─────────┬────────────┘
                                    │
                                    ▼
                        ┌──────────────────────────┐
                        │ Medical NER (SciSpacy)   │
                        └─────────┬────────────────┘
                                  │
                ┌────────────────┴─────────────────┐
                │       Embed user query           │
                │ (Sentence Transformers model)    │
                └────────────────┬─────────────────┘
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
         ┌───────────────────────────────────────┐            │
         │ Rerank/filter by NER, semantic group  │            │
         └──────────────────┬────────────────────┘            │
                            ▼                                 │
              ┌──────────────────────────────┐                │
              │ Return Best Answer to UI │◄────────┘
              └──────────────────────────────┘
```

## Installation

### Option 1: Automated Setup (Recommended)

1. Clone the repository

2. Create a virtual environment (recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Run the setup script:

```bash
# For a clean installation (removes existing packages first)
python setup_environment.py --clean --scispacy

# Or, to install without removing existing packages
python setup_environment.py --scispacy
```

This script will:
- Install compatible versions of NumPy, SciPy, and other dependencies
- Install FAISS and Sentence Transformers
- Clone and install SciSpacy with the nmslib dependency removed
- Install the scientific spaCy model
- Test the installation

### Option 2: Manual Installation

1. Clone the repository

2. Create a virtual environment (recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install NumPy first (specific version for compatibility):

```bash
pip install numpy>=1.19.5,<1.25.0
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Install SciSpacy (special installation due to dependency issues):

```bash
# Clone the SciSpacy repository
git clone https://github.com/allenai/scispacy.git
cd scispacy

# Edit setup.py and remove "nmslib>=1.7.3.6" from install_requires
# Then install SciSpacy
pip install .
cd ..
```

6. Install the scientific spaCy model:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

7. Test your installation:

```bash
python test_installation.py
```

### Troubleshooting

1. **NumPy/SciPy compatibility issues**:
   - If you see errors about "numpy.dtype size changed" or binary incompatibility:
   ```bash
   pip uninstall -y numpy scipy
   pip install numpy>=1.19.5,<1.25.0
   pip install scipy>=1.7.0,<1.11.0
   ```

2. **FAISS installation issues**:
   - Try installing a different version:
   ```bash
   pip install faiss-cpu
   ```

3. **Flask and Werkzeug issues**:
   - If you encounter issues with the `url_quote` import error:
   ```bash
   pip uninstall -y werkzeug
   pip install werkzeug==2.0.3
   ```

4. **Sentence Transformers issues**:
   - Make sure you have PyTorch installed:
   ```bash
   pip install torch
   pip install sentence-transformers
   ```

## Usage

1. Prepare the data and build the index (if not already done):

```bash
python prepare_data.py
```

2. Run the application:

```bash
python run.py
```

3. Open your browser and navigate to `http://localhost:5000`

4. Start asking medical questions!

## Development

- To rebuild the FAISS index:

```bash
python prepare_data.py --force
```

- To process the MedQuAD dataset (if needed):

```bash
python processing.py
```

- To test the installation:

```bash
python test_installation.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.