#!/bin/bash
# Run the fine-tuning process

# Set the number of samples to use for fine-tuning
MAX_SAMPLES=1000

# Set the output directory
OUTPUT_DIR="models/mistral-7b-arxiv-finetuned"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the fine-tuning script
echo "Starting fine-tuning process with $MAX_SAMPLES samples..."
python scripts/finetune_model.py --papers mongodb --max_samples $MAX_SAMPLES --output_dir $OUTPUT_DIR

echo "Fine-tuning complete. Model saved to $OUTPUT_DIR"
