#!/usr/bin/env python
"""
Fine-tune a Mistral 7B model on arXiv papers.

This script fine-tunes the Mistral 7B model on a dataset of arXiv papers
to create a domain-specific model for scientific research assistance.
"""

import os
import logging
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/model_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_dataset(papers_path: str, max_samples: int = 1000):
    """
    Prepare the dataset for fine-tuning.
    
    Args:
        papers_path: Path to the papers data (CSV or MongoDB)
        max_samples: Maximum number of samples to use
        
    Returns:
        Dataset for fine-tuning
    """
    logger.info(f"Preparing dataset from {papers_path}")
    
    # Check if the path is a CSV file
    if papers_path.endswith('.csv'):
        # Load from CSV
        df = pd.read_csv(papers_path)
        logger.info(f"Loaded {len(df)} papers from CSV")
    else:
        # Load from MongoDB
        from pymongo import MongoClient
        
        # Load database configuration
        with open("config/app_config.yaml", 'r') as f:
            app_config = yaml.safe_load(f)
        
        # Connect to MongoDB
        uri = os.environ.get("MONGODB_URI", app_config["database"]["mongodb"]["uri"])
        db_name = os.environ.get("MONGODB_DB", app_config["database"]["mongodb"]["db_name"])
        collection_name = app_config["database"]["mongodb"]["collections"]["papers"]
        
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Get papers from MongoDB
        papers = list(collection.find({"processed": True}).limit(max_samples))
        logger.info(f"Loaded {len(papers)} papers from MongoDB")
        
        # Convert to DataFrame
        df = pd.DataFrame(papers)
    
    # Limit the number of samples
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
        logger.info(f"Sampled {max_samples} papers for fine-tuning")
    
    # Create training examples
    training_examples = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing examples"):
        # Extract paper information
        title = row.get('title', '')
        abstract = row.get('abstract', '')
        
        # Get sections if available
        sections = []
        if 'sections' in row and isinstance(row['sections'], list):
            for section in row['sections']:
                if isinstance(section, dict) and 'title' in section and 'content' in section:
                    sections.append(f"{section['title']}: {section['content']}")
        
        # Create instruction examples
        
        # Example 1: Summarize the paper
        training_examples.append({
            'instruction': f"Summarize the following research paper.",
            'input': f"Title: {title}\n\nAbstract: {abstract}",
            'output': f"This paper titled '{title}' explores {abstract[:200]}... The research contributes to the field by providing insights into {abstract[-200:] if len(abstract) > 200 else abstract}"
        })
        
        # Example 2: Explain a concept from the paper
        if abstract:
            training_examples.append({
                'instruction': f"Explain the main concept discussed in this research paper.",
                'input': f"Title: {title}\n\nAbstract: {abstract}",
                'output': f"The main concept in the paper '{title}' is related to {abstract[:300]}... This is significant because it helps advance our understanding of the field."
            })
        
        # Example 3: Answer a question about the paper
        if len(sections) > 0:
            section_text = "\n\n".join(sections[:2])  # Use first two sections
            training_examples.append({
                'instruction': f"Based on the paper sections, what methodology did the researchers use?",
                'input': f"Title: {title}\n\nSections:\n{section_text}",
                'output': f"In the paper '{title}', the researchers used a methodology that involves {section_text[:200]}... This approach allowed them to effectively investigate their research questions."
            })
    
    # Create dataset
    dataset = Dataset.from_pandas(pd.DataFrame(training_examples))
    
    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    
    return DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

def format_instruction(example):
    """Format the instruction, input, and output into a single text."""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    
    if input_text:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": text}

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the examples and prepare them for training."""
    # Tokenize the texts
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Set the labels (needed for training)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs

def fine_tune_model(config, dataset_dict, output_dir):
    """
    Fine-tune the model on the prepared dataset.
    
    Args:
        config: Model configuration
        dataset_dict: Dataset dictionary with train and test splits
        output_dir: Directory to save the fine-tuned model
    """
    logger.info("Starting fine-tuning process")
    
    # Load model configuration
    base_model_id = config["model"]["base_model_id"]
    fine_tuning_config = config["model"]["fine_tuning"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Format the dataset
    formatted_dataset = dataset_dict.map(format_instruction)
    
    # Tokenize the dataset
    max_length = config["model"]["context_window"]["max_length"]
    tokenized_dataset = formatted_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output", "text"]
    )
    
    logger.info(f"Tokenized dataset: {tokenized_dataset}")
    
    # Configure quantization
    if config["model"]["quantization"]["method"] == "bitsandbytes" and torch.cuda.is_available():
        logger.info(f"Using {config['model']['quantization']['bits']}-bit quantization with bitsandbytes")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config["model"]["quantization"]["bits"] == 4,
            load_in_8bit=config["model"]["quantization"]["bits"] == 8,
            bnb_4bit_compute_dtype=torch.float16 if config["model"]["quantization"]["bits"] == 4 else None,
            bnb_4bit_quant_type="nf4" if config["model"]["quantization"]["bits"] == 4 else None,
        )
    else:
        quantization_config = None
    
    # Load base model
    logger.info(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training if using quantization
    if quantization_config:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Configuring LoRA adapter")
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=fine_tuning_config["num_train_epochs"],
        per_device_train_batch_size=fine_tuning_config["train_batch_size"],
        per_device_eval_batch_size=fine_tuning_config["eval_batch_size"],
        gradient_accumulation_steps=fine_tuning_config["gradient_accumulation_steps"],
        evaluation_strategy="steps",
        eval_steps=fine_tuning_config["eval_steps"],
        logging_dir=f"{output_dir}/logs",
        logging_steps=fine_tuning_config["logging_steps"],
        save_steps=fine_tuning_config["save_steps"],
        learning_rate=fine_tuning_config["learning_rate"],
        weight_decay=fine_tuning_config["weight_decay"],
        fp16=torch.cuda.is_available(),
        bf16=False,
        max_grad_norm=fine_tuning_config["max_grad_norm"],
        warmup_steps=fine_tuning_config["warmup_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Fine-tuning complete")
    
    return model, tokenizer

def main():
    """Main function to run the fine-tuning process."""
    parser = argparse.ArgumentParser(description="Fine-tune a model on arXiv papers")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to the configuration file")
    parser.add_argument("--papers", type=str, default="mongodb", help="Path to the papers data (CSV or 'mongodb')")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the fine-tuned model")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use for fine-tuning")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set output directory
    output_dir = args.output_dir or config["model"]["fine_tuned_model_path"]
    
    # Prepare dataset
    dataset_dict = prepare_dataset(args.papers, args.max_samples)
    
    # Fine-tune model
    fine_tune_model(config, dataset_dict, output_dir)

if __name__ == "__main__":
    main()
