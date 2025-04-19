# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data_storage/papers/pdfs \
    data_storage/papers/metadata \
    data_storage/processed/text \
    data_storage/chunks \
    models \
    vector_db/embeddings \
    logs \
    conversations

# Download LLaMA model (commented out to reduce build time, uncomment if needed)
# RUN python download_model.py

# Expose API port
EXPOSE 8000

# Command to run the API server
CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8000"]
