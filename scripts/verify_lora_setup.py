#!/usr/bin/env python3
"""
LoRA Setup Verification Script
Verifies that LoRA dependencies are properly installed and configured.
"""

import sys
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_peft_import():
    """Test if PEFT library can be imported."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        logging.info("✓ PEFT library imported successfully")
        return True
    except ImportError as e:
        logging.error(f"✗ Failed to import PEFT: {e}")
        return False

def test_lora_setup():
    """Test basic LoRA setup with a small model."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Load a small model for testing
        logging.info("Loading test model...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn"],
            bias="none",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print parameters
        model.print_trainable_parameters()
        
        logging.info("✓ LoRA setup test completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"✗ LoRA setup test failed: {e}")
        return False

def test_device_compatibility():
    """Test device compatibility for LoRA training."""
    try:
        # Check MPS availability
        if torch.backends.mps.is_available():
            logging.info("✓ MPS (Apple Silicon) is available")
            device = torch.device("mps")
        elif torch.cuda.is_available():
            logging.info("✓ CUDA is available")
            device = torch.device("cuda")
        else:
            logging.info("✓ Using CPU (no GPU acceleration)")
            device = torch.device("cpu")
        
        # Test tensor creation on device
        test_tensor = torch.randn(2, 3).to(device)
        logging.info(f"✓ Test tensor created on {device}")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Device compatibility test failed: {e}")
        return False

def main():
    """Main verification function."""
    setup_logging()
    
    logging.info("LoRA Setup Verification")
    logging.info("=" * 40)
    
    tests = [
        ("PEFT Import", test_peft_import),
        ("LoRA Setup", test_lora_setup),
        ("Device Compatibility", test_device_compatibility),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        logging.info(f"\nTesting: {test_name}")
        logging.info("-" * 30)
        
        if not test_func():
            all_passed = False
            logging.error(f"❌ {test_name} failed")
        else:
            logging.info(f"✅ {test_name} passed")
    
    logging.info("\n" + "=" * 40)
    
    if all_passed:
        logging.info("🎉 All LoRA setup tests passed!")
        logging.info("You can now use LoRA training with:")
        logging.info("  python scripts/train_lora.py")
        logging.info("  python scripts/main.py --train-lora")
        logging.info("  python scripts/main.py --full-lora")
    else:
        logging.error("❌ Some tests failed. Please check your setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()
