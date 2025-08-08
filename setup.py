#!/usr/bin/env python3
"""
Environment Setup Script
Sets up the complete environment for GPT-2 fine-tuning.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_environment():
    """Check if we're in the right environment."""
    print("Checking environment...")

    # Check if we're in the right directory
    if not os.path.exists("configs") or not os.path.exists("scripts"):
        print("Please run this script from the project root directory")
        return False

    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        return False

    print("Environment checks passed")
    return True


def install_dependencies():
    """Install Python dependencies."""
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found")
        return False

    return run_command(
        f"{sys.executable} -m pip install --no-cache-dir -r requirements.txt",
        "Installing Python dependencies",
    )


def verify_installation():
    """Verify that the installation was successful."""
    return run_command(
        f"{sys.executable} scripts/verify_setup.py", "Verifying installation"
    )


def setup_config():
    """Check and optionally create default config."""
    config_path = "configs/config.yaml"

    if os.path.exists(config_path):
        print("Configuration file already exists")
        return True

    print("Configuration file not found")
    print("Please ensure configs/config.yaml exists")
    return False


def main():
    """Main setup function."""
    print("=" * 50)
    print("GPT-2 Fine-tuning Environment Setup")
    print("=" * 50)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("\nFailed to install dependencies")
        sys.exit(1)

    # Verify installation
    if not verify_installation():
        print("\nInstallation verification failed")
        sys.exit(1)

    # Check config
    if not setup_config():
        print("\nConfiguration setup failed")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Environment setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Review configs/config.yaml")
    print("2. Run: python scripts/main.py --full")
    print("3. Open notebooks/model_comparison.ipynb")
    print("\nFor help: python scripts/main.py --help")


if __name__ == "__main__":
    main()
