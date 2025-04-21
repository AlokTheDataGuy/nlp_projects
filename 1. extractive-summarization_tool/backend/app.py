# app.py
from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
import numpy as np
from flask_cors import CORS
import tempfile
import PyPDF2
import pdfplumber
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure max content length (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

class BERTSUMExtractiveTextSummarizer:
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        """
        Initialize the BERTSUM-style extractive summarizer.
        """
        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)

    def _split_into_sentences(self, text: str):
        """Split text into sentences."""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _get_sentence_embeddings(self, sentences):
        """Get BERT embeddings for each sentence."""
        embeddings = []

        # Process sentences in batches
        batch_size = 8
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]

            # Tokenize sentences
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt").to(self.device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get the [CLS] token embedding for each sentence
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)

        return np.array(embeddings)

    def _build_similarity_matrix(self, embeddings):
        """Build a similarity matrix between all sentences."""
        return cosine_similarity(embeddings)

    def _rank_sentences(self, similarity_matrix):
        """Rank sentences using the TextRank algorithm."""
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, alpha=0.85, max_iter=100)
        ranked_sentences = sorted(((scores[i], i) for i in range(len(scores))), reverse=True)
        return [idx for _, idx in ranked_sentences]

    def summarize(self, text, ratio=0.3, min_sentences=3, max_sentences=10):
        """Generate an extractive summary."""
        # Split text into sentences
        sentences = self._split_into_sentences(text)

        # If there are very few sentences, return the original text
        if len(sentences) <= min_sentences:
            return text

        # Get embeddings for each sentence
        embeddings = self._get_sentence_embeddings(sentences)

        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(embeddings)

        # Rank sentences
        ranked_sentences = self._rank_sentences(similarity_matrix)

        # Determine number of sentences for the summary
        num_sentences = max(min_sentences, min(max_sentences, int(len(sentences) * ratio)))

        # Select the top sentences, but preserve original order
        selected_indices = sorted(ranked_sentences[:num_sentences])

        # Create summary
        summary = " ".join([sentences[idx] for idx in selected_indices])

        return summary

# Initialize summarizer
summarizer = BERTSUMExtractiveTextSummarizer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using multiple methods for better reliability."""
    text = ""
    pdf_content = pdf_file.read()
    pdf_file.seek(0)  # Reset file pointer for potential reuse

    # Try PyPDF2 first
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))

        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            raise ValueError("The PDF file is password-protected.")

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"PyPDF2 extraction failed: {str(e)}")

    # If PyPDF2 didn't extract much text, try pdfplumber
    if len(text.strip()) < 100:
        try:
            pdf_file.seek(0)  # Reset file pointer
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber extraction failed: {str(e)}")

            # If both methods failed and we have no text, raise an error
            if len(text.strip()) < 100:
                raise ValueError("Failed to extract text from the PDF file. The file might be corrupted or contain only images.")

    return text

@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If user does not select file, browser may submit an empty file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # Get parameters
        ratio = float(request.form.get('ratio', 0.3))
        min_sentences = int(request.form.get('min', 3))
        max_sentences = int(request.form.get('max', 10))

        # Read file content based on file type
        file_extension = file.filename.rsplit('.', 1)[1].lower()

        try:
            if file_extension == 'pdf':
                text = extract_text_from_pdf(file)
            else:  # txt file
                text = file.read().decode('utf-8')

            # Check if we have enough text to summarize
            if len(text.strip()) < 100:
                return jsonify({'error': 'The file contains too little text to summarize.'}), 400
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

        # Generate summary
        summary = summarizer.summarize(
            text,
            ratio=ratio,
            min_sentences=min_sentences,
            max_sentences=max_sentences
        )

        # Create temporary file for download
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        temp_path = temp.name
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        # Store path for download endpoint
        filename = secure_filename(file.filename)
        base_name = os.path.splitext(filename)[0]
        download_filename = f"{base_name}_summary.txt"

        return jsonify({
            'success': True,
            'summary': summary,
            'file_path': temp_path,
            'download_name': download_filename
        })

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/download/<path:file_path>/<download_name>', methods=['GET'])
def download_file(file_path, download_name):
    try:
        return send_file(file_path, as_attachment=True, download_name=download_name)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handler for file too large
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum file size is 10MB.'}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)