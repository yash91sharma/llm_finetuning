#!/usr/bin/env python3
"""
Training Script
Handles the complete training, testing, and validation pipeline for GPT-2 fine-tuning.
"""

import os
import json
import yaml
import torch
import argparse
from pathlib import Path
from datetime import datetime
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import logging

def setup_logging(log_level="INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class CustomDataset(Dataset):
    """Custom dataset for GPT-2 fine-tuning."""
    
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the text based on the instruction-input-output format
        if item.get('input', '').strip():
            text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}"
        else:
            text = f"Instruction: {item['instruction']}\nOutput: {item['output']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def load_data(config):
    """Load and prepare training and validation data."""
    train_file = config['data']['train_file']
    eval_file = config['data']['eval_file']
    test_split = config['data']['train_test_split']
    
    # Load training data
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    # Load evaluation data if it exists, otherwise split training data
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        logging.info(f"Loaded {len(train_data)} training examples and {len(eval_data)} evaluation examples")
    else:
        # Split training data
        train_data, eval_data = train_test_split(
            train_data, 
            test_size=test_split, 
            random_state=42
        )
        logging.info(f"Split data: {len(train_data)} training, {len(eval_data)} validation examples")
    
    return train_data, eval_data

def get_device(config):
    """Determine the best available device."""
    device_config = config.get('device', {})
    
    if device_config.get('use_mps', False) and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_config.get('use_cuda', False) and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def setup_model_and_tokenizer(config, device):
    """Load and setup model and tokenizer."""
    model_name = config['model']['name']
    save_path = config['model']['save_path']
    
    model_path = os.path.join(save_path, model_name)
    tokenizer_path = os.path.join(save_path, f"{model_name}-tokenizer")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    
    # Resize token embeddings if necessary
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def create_training_arguments(config):
    """Create training arguments from config."""
    training_config = config['training']
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(training_config['output_dir'], f"gpt2_finetuned_{timestamp}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Ensure numeric values are properly converted from YAML
    learning_rate = float(training_config['learning_rate'])
    weight_decay = float(training_config['weight_decay'])
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(training_config['num_train_epochs']),
        per_device_train_batch_size=int(training_config['per_device_train_batch_size']),
        per_device_eval_batch_size=int(training_config['per_device_eval_batch_size']),
        gradient_accumulation_steps=int(training_config['gradient_accumulation_steps']),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=int(training_config['warmup_steps']),
        logging_steps=int(training_config['logging_steps']),
        eval_steps=int(training_config['eval_steps']),
        save_steps=int(training_config['save_steps']),
        save_total_limit=int(training_config['save_total_limit']),
        eval_strategy=training_config.get('eval_strategy', training_config.get('evaluation_strategy', 'steps')),
        load_best_model_at_end=bool(training_config['load_best_model_at_end']),
        metric_for_best_model=training_config['metric_for_best_model'],
        greater_is_better=bool(training_config['greater_is_better']),
        dataloader_pin_memory=False,  # Disable for MPS compatibility
        remove_unused_columns=False,
        report_to=[],  # Empty list instead of None to avoid warnings
    )

def train_model(config):
    """Main training function."""
    # Get device
    device = get_device(config)
    logging.info(f"Using device: {device}")
    
    # Load data
    logging.info("Loading data...")
    train_data, eval_data = load_data(config)
    
    # Setup model and tokenizer
    logging.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config, device)
    
    # Create datasets
    max_length = config['data']['max_length']
    train_dataset = CustomDataset(train_data, tokenizer, max_length)
    eval_dataset = CustomDataset(eval_data, tokenizer, max_length)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is causal LM, not masked LM
    )
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Updated from tokenizer to processing_class
    )
    
    # Start training
    logging.info("Starting training...")
    train_result = trainer.train()
    
    # Save the final model
    logging.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Final evaluation
    logging.info("Running final evaluation...")
    eval_result = trainer.evaluate()
    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)
    
    logging.info(f"Training completed! Model saved to {training_args.output_dir}")
    
    return training_args.output_dir, trainer

def test_model(model_path, config):
    """Test the fine-tuned model with sample prompts."""
    logging.info("Testing fine-tuned model...")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    device = get_device(config)
    model = model.to(device)
    model.eval()
    
    # Test prompts
    test_prompts = [
        "Instruction: Explain what artificial intelligence is.\nOutput:",
        "Instruction: What is machine learning?\nOutput:",
        "Instruction: How does deep learning work?\nOutput:"
    ]
    
    generation_config = config['generation']
    
    logging.info("Generating test outputs...")
    
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=generation_config['max_new_tokens'],
                temperature=generation_config['temperature'],
                do_sample=generation_config['do_sample'],
                pad_token_id=generation_config['pad_token_id'],
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logging.info(f"\n--- Test {i+1} ---")
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Generated: {generated_text[len(prompt):]}")

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-only",
        type=str,
        help="Path to model directory for testing only"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config.get('logging', {}).get('log_level', 'INFO'))
    
    if args.test_only:
        # Test existing model
        test_model(args.test_only, config)
    else:
        # Train new model
        model_path, trainer = train_model(config)
        
        # Test the trained model
        test_model(model_path, config)

if __name__ == "__main__":
    main()
