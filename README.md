# LLM Retraining Project

This project contains code and resources for retraining a small open-source LLM on custom text data.

## Setup Instructions

### 1. Create Virtual Environment
```bash
python3 -m venv llm_training_env
source llm_training_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Metal available: {torch.backends.mps.is_available()}')"
```

## Project Structure
```
llm/
├── data/              # Training data
├── models/            # Model checkpoints and configs
├── scripts/           # Training and evaluation scripts
├── notebooks/         # Jupyter notebooks for experimentation
├── configs/           # Configuration files
└── outputs/           # Training outputs and logs
```

## Getting Started

1. Place your custom text data in the `data/` directory
2. Configure training parameters in `configs/`
3. Run training scripts from `scripts/`
4. Monitor progress with tensorboard or wandb

## Apple Silicon Optimization

This setup is optimized for Apple Silicon (M3 Pro) with:
- Native ARM64 PyTorch builds
- Metal Performance Shaders (MPS) support
- Optimized dependencies for macOS
