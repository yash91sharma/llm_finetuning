"""
Environment verification script for LLM training setup.
Run this after installing dependencies to verify everything is working correctly.
"""

import sys
import torch
import transformers
import datasets
from packaging import version


def check_python_version():
    """Check if Python version is suitable for ML training."""
    print(f"Python version: {sys.version}")
    if sys.version_info >= (3, 8):
        print("Python version is suitable")
        return True
    else:
        print("Python version should be 3.8 or higher")
        return False


def check_pytorch():
    """Check PyTorch installation and Apple Silicon support."""
    print(f"\nPyTorch version: {torch.__version__}")

    # Check MPS availability (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        print("Metal Performance Shaders (MPS) is available")
        print("Apple Silicon GPU acceleration is ready")

        # Test MPS functionality
        try:
            device = torch.device("mps")
            x = torch.randn(3, 3, device=device)
            y = torch.randn(3, 3, device=device)
            z = torch.matmul(x, y)
            print("MPS tensor operations working correctly")
            return True
        except Exception as e:
            print(f"MPS test failed: {e}")
            return False
    else:
        print("MPS not available - will fall back to CPU")
        return False


def check_transformers():
    """Check Transformers library."""
    print(f"\nTransformers version: {transformers.__version__}")

    # Check if version supports recent features
    min_version = "4.30.0"
    if version.parse(transformers.__version__) >= version.parse(min_version):
        print("Transformers version is up to date")
        return True
    else:
        print(f"Consider upgrading transformers to {min_version} or higher")
        return True  # Still functional


def check_datasets():
    """Check datasets library."""
    print(f"\nDatasets version: {datasets.__version__}")
    print("Datasets library is ready")
    return True


def check_memory():
    """Check available memory."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)

        print(f"\nSystem Memory:")
        print(f"Total: {total_gb:.1f} GB")
        print(f"Available: {available_gb:.1f} GB")

        if total_gb >= 16:
            print("Sufficient memory for small LLM training")
        elif total_gb >= 8:
            print("Limited memory - consider smaller models or batch sizes")
        else:
            print("Insufficient memory for LLM training")

        return total_gb >= 8
    except ImportError:
        print("\npsutil not installed - cannot check memory")
        return True


def main():
    """Run all verification checks."""
    print("Verifying LLM Training Environment Setup\n")
    print("=" * 50)

    checks = [
        check_python_version(),
        check_pytorch(),
        check_transformers(),
        check_datasets(),
        check_memory(),
    ]

    print("\n" + "=" * 50)

    if all(checks):
        print("Environment setup is complete and ready for LLM training!")
        print("\nNext steps:")
        print("1. Add your training data to the 'data/' directory")
        print("2. Configure training parameters in 'configs/'")
        print("3. Start with the example notebook in 'notebooks/'")
    else:
        print("Some issues detected. Please address them before proceeding.")

    return all(checks)


if __name__ == "__main__":
    main()
