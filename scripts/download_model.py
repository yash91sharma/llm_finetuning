#!/usr/bin/env python3
"""
Model Download Script
Downloads and saves GPT-2 model and tokenizer to local directory.
"""

import os
import yaml
import argparse
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
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

def download_model(config):
    """Download and save GPT-2 model and tokenizer."""
    model_name = config['model']['name']
    save_path = config['model']['save_path']
    cache_dir = config['model']['cache_dir']
    
    # Create directories if they don't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    model_save_path = os.path.join(save_path, model_name)
    tokenizer_save_path = os.path.join(save_path, f"{model_name}-tokenizer")
    
    logging.info(f"Downloading {model_name} model...")
    
    try:
        # Download and save tokenizer
        logging.info("Downloading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        tokenizer.save_pretrained(tokenizer_save_path)
        logging.info(f"Tokenizer saved to {tokenizer_save_path}")
        
        # Download and save model
        logging.info("Downloading model...")
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        model.save_pretrained(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(tokenizer_save_path)
            logging.info("Added pad token to tokenizer")
        
        logging.info("Model and tokenizer downloaded successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download GPT-2 model and tokenizer")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config.get('logging', {}).get('log_level', 'INFO'))
    
    # Download model
    success = download_model(config)
    
    if success:
        logging.info("Model download completed successfully!")
    else:
        logging.error("Model download failed!")
        exit(1)

if __name__ == "__main__":
    main()
