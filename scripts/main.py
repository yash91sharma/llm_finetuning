#!/usr/bin/env python3
"""
Main Management Script
Orchestrates the entire GPT-2 fine-tuning pipeline.
"""

import os
import sys
import yaml
import argparse
import subprocess
import logging
from pathlib import Path


def setup_logging(log_level="INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        return None

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def check_python_environment():
    """Check if we're in a virtual environment and have required packages."""
    if not hasattr(sys, "real_prefix") and not (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        logging.warning(
            "Not running in a virtual environment. Consider using one for better package management."
        )

    try:
        import torch
        import transformers

        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"Transformers version: {transformers.__version__}")
    except ImportError as e:
        logging.error(f"Required packages not found: {e}")
        logging.error(
            "Please install required packages: pip install -r requirements.txt"
        )
        return False

    return True


def run_script(script_name, args=None, config_path="configs/config.yaml"):
    """Run a Python script with the given arguments."""
    script_path = os.path.join("scripts", script_name)

    if not os.path.exists(script_path):
        logging.error(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, script_path, "--config", config_path]
    if args:
        cmd.extend(args)

    logging.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logging.error(f"Script failed with return code {e.returncode}")
        return False


def download_phase(config_path):
    """Execute the model download phase."""
    logging.info("=== PHASE 1: DOWNLOADING MODEL ===")
    return run_script("download_model.py", config_path=config_path)


def setup_phase(config_path):
    """Execute the device setup phase."""
    logging.info("=== PHASE 2: SETTING UP DEVICE ===")
    return run_script("setup_device.py", config_path=config_path)


def training_phase(config_path):
    """Execute the training phase."""
    logging.info("=== PHASE 3: TRAINING MODEL ===")
    return run_script("train_model.py", config_path=config_path)


def training_lora_phase(config_path):
    """Execute the LoRA training phase."""
    logging.info("=== PHASE 3: TRAINING MODEL WITH LORA ===")
    return run_script("train_lora.py", config_path=config_path)


def validate_config(config):
    """Validate the configuration file."""
    required_sections = ["model", "data", "training", "device"]

    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required configuration section: {section}")
            return False

    # Check if this is a LoRA config
    if "lora" in config:
        logging.info("Detected LoRA configuration")
        lora_config = config["lora"]
        required_lora_fields = ["r", "lora_alpha", "lora_dropout", "target_modules"]

        for field in required_lora_fields:
            if field not in lora_config:
                logging.error(f"Missing required LoRA configuration field: {field}")
                return False

    # Check if data files exist
    train_file = config["data"]["train_file"]
    if not os.path.exists(train_file):
        logging.error(f"Training data file not found: {train_file}")
        return False

    # Create output directories if they don't exist
    Path(config["model"]["save_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    return True


def full_pipeline(config_path):
    """Execute the complete fine-tuning pipeline."""
    logging.info("Starting GPT-2 Fine-tuning Pipeline")
    logging.info("=" * 50)

    # Load and validate configuration
    config = load_config(config_path)
    if config is None:
        return False

    if not validate_config(config):
        return False

    # Check environment
    if not check_python_environment():
        return False

    # Execute phases
    phases = [
        ("Download", download_phase),
        ("Setup", setup_phase),
        ("Training", training_phase),
    ]

    for phase_name, phase_func in phases:
        logging.info(f"\nStarting {phase_name} phase...")
        if not phase_func(config_path):
            logging.error(f"{phase_name} phase failed!")
            return False
        logging.info(f"{phase_name} phase completed successfully!")

    logging.info("\n" + "=" * 50)
    logging.info("GPT-2 Fine-tuning Pipeline completed successfully!")
    logging.info("Check the outputs/ directory for your fine-tuned model.")

    return True


def full_lora_pipeline(config_path):
    """Execute the complete LoRA fine-tuning pipeline."""
    logging.info("Starting GPT-2 LoRA Fine-tuning Pipeline")
    logging.info("=" * 50)

    # Load and validate configuration
    config = load_config(config_path)
    if config is None:
        return False

    if not validate_config(config):
        return False

    # Check environment
    if not check_python_environment():
        return False

    # Execute phases
    phases = [
        ("Download", download_phase),
        ("Setup", setup_phase),
        ("LoRA Training", training_lora_phase),
    ]

    for phase_name, phase_func in phases:
        logging.info(f"\nStarting {phase_name} phase...")
        if not phase_func(config_path):
            logging.error(f"{phase_name} phase failed!")
            return False
        logging.info(f"{phase_name} phase completed successfully!")

    logging.info("\n" + "=" * 50)
    logging.info("GPT-2 LoRA Fine-tuning Pipeline completed successfully!")
    logging.info("Check the outputs/ directory for your LoRA adapter.")

    return True


def list_models():
    """List available models in the models directory."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        logging.info("No models directory found.")
        return

    logging.info("Available models:")
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            logging.info(f"  - {item}")


def list_outputs():
    """List training outputs in the outputs directory."""
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        logging.info("No outputs directory found.")
        return

    logging.info("Training outputs:")
    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            logging.info(f"  - {item}")


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 Fine-tuning Management Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full                     # Run complete pipeline (full fine-tuning)
  python main.py --full-lora                # Run complete pipeline with LoRA
  python main.py --download                 # Download model only
  python main.py --setup                    # Setup device only
  python main.py --train                    # Train model only (full fine-tuning)
  python main.py --train-lora               # Train model with LoRA
  python main.py --test outputs/model_dir   # Test specific model
  python main.py --test-lora outputs/lora_dir # Test LoRA adapter
  python main.py --list-models              # List available models
  python main.py --list-outputs             # List training outputs
  python main.py --verify-lora              # Verify LoRA setup
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )

    # Pipeline options
    parser.add_argument("--full", action="store_true", help="Run complete pipeline")
    parser.add_argument("--download", action="store_true", help="Download model only")
    parser.add_argument("--setup", action="store_true", help="Setup device only")
    parser.add_argument("--train", action="store_true", help="Train model only")
    parser.add_argument(
        "--train-lora", action="store_true", help="Train model with LoRA"
    )
    parser.add_argument(
        "--full-lora", action="store_true", help="Run complete pipeline with LoRA"
    )
    parser.add_argument("--test", type=str, help="Test model at specified path")
    parser.add_argument(
        "--test-lora", type=str, help="Test LoRA adapter at specified path"
    )

    # Utility options
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--list-outputs", action="store_true", help="List training outputs"
    )
    parser.add_argument(
        "--validate-config", action="store_true", help="Validate configuration file"
    )
    parser.add_argument("--verify-lora", action="store_true", help="Verify LoRA setup")

    args = parser.parse_args()

    # Load config for logging setup
    config = load_config(args.config)
    if config:
        setup_logging(config.get("logging", {}).get("log_level", "INFO"))
    else:
        setup_logging()

    # Handle different options
    if args.full:
        success = full_pipeline(args.config)
        sys.exit(0 if success else 1)

    elif args.full_lora:
        # Use LoRA config by default if not specified
        config_path = args.config
        if config_path == "configs/config.yaml":
            config_path = "configs/config_lora.yaml"
        success = full_lora_pipeline(config_path)
        sys.exit(0 if success else 1)

    elif args.download:
        success = download_phase(args.config)
        sys.exit(0 if success else 1)

    elif args.setup:
        success = setup_phase(args.config)
        sys.exit(0 if success else 1)

    elif args.train:
        success = training_phase(args.config)
        sys.exit(0 if success else 1)

    elif args.train_lora:
        # Use LoRA config by default if not specified
        config_path = args.config
        if config_path == "configs/config.yaml":
            config_path = "configs/config_lora.yaml"
        success = training_lora_phase(config_path)
        sys.exit(0 if success else 1)

    elif args.test:
        success = run_script("train_model.py", ["--test-only", args.test], args.config)
        sys.exit(0 if success else 1)

    elif args.test_lora:
        # Use LoRA config by default if not specified
        config_path = args.config
        if config_path == "configs/config.yaml":
            config_path = "configs/config_lora.yaml"
        success = run_script(
            "train_lora.py", ["--test-only", args.test_lora], config_path
        )
        sys.exit(0 if success else 1)

    elif args.list_models:
        list_models()

    elif args.list_outputs:
        list_outputs()

    elif args.validate_config:
        if config and validate_config(config):
            logging.info("Configuration is valid!")
        else:
            logging.error("Configuration validation failed!")
            sys.exit(1)

    elif args.verify_lora:
        success = run_script("verify_lora_setup.py", config_path=args.config)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
