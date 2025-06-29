#!/usr/bin/env python3
"""
Device Setup Script
Sets up the model for training on the specified device (MPS/CUDA/CPU).
"""

import os
import yaml
import torch
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

def get_device(config):
    """Determine the best available device based on configuration."""
    device_config = config.get('device', {})
    
    if device_config.get('use_mps', False) and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon GPU)")
    elif device_config.get('use_cuda', False) and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    return device

def load_and_setup_model(config, device):
    """Load model and tokenizer and transfer to specified device."""
    model_name = config['model']['name']
    save_path = config['model']['save_path']
    
    model_path = os.path.join(save_path, model_name)
    tokenizer_path = os.path.join(save_path, f"{model_name}-tokenizer")
    
    # Check if local model exists
    if not os.path.exists(model_path):
        logging.error(f"Model not found at {model_path}. Please run download_model.py first.")
        return None, None
    
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer not found at {tokenizer_path}. Please run download_model.py first.")
        return None, None
    
    try:
        # Load tokenizer
        logging.info("Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        logging.info("Loading model...")
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Move model to device
        logging.info(f"Moving model to {device}...")
        model = model.to(device)
        
        # Set model to training mode
        model.train()
        
        logging.info("Model setup completed successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Model device: {next(model.parameters()).device}")
        
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Error setting up model: {str(e)}")
        return None, None

def test_device_compatibility(device):
    """Test if the device is working properly."""
    try:
        logging.info("Testing device compatibility...")
        
        # Create a simple tensor and move it to device
        test_tensor = torch.randn(10, 10).to(device)
        result = torch.matmul(test_tensor, test_tensor.T)
        
        logging.info(f"Device test passed! Tensor shape: {result.shape}")
        return True
        
    except Exception as e:
        logging.error(f"Device test failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup model for training device")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test device compatibility"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config.get('logging', {}).get('log_level', 'INFO'))
    
    # Get device
    device = get_device(config)
    
    # Test device
    if not test_device_compatibility(device):
        logging.error("Device compatibility test failed!")
        exit(1)
    
    if args.test_only:
        logging.info("Device test completed successfully!")
        return
    
    # Load and setup model
    model, tokenizer = load_and_setup_model(config, device)
    
    if model is None or tokenizer is None:
        logging.error("Model setup failed!")
        exit(1)
    
    logging.info("Device setup completed successfully!")

if __name__ == "__main__":
    main()
