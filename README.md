# GPT-2 Fine-tuning Project

Practise codes for learning and revising the concepts by fine-tuning GPT-2 models with Apple Silicon (MPS) support.


## Quick Start Guide

### Step 1: Initial Setup

```bash
# Create virtual environment
python3 -m venv llm_training_env

source llm_training_env/bin/activate

which python 

python setup.py
```

### Step 3: Run the Complete Pipeline

```bash
# One command to rule them all
python scripts/main.py --full

# This will:
# 1. Download GPT-2 model and tokenizer
# 2. Setup device (MPS/CUDA/CPU)
# 3. Train the model on your data
# 4. Save the fine-tuned model with timestamp
```

### Step 4: Test and Compare Models

```bash
# Start Jupyter notebook server
jupyter notebook

# Open and run these notebooks:
# 1. notebooks/model_comparison.ipynb - Compare original vs fine-tuned
# 2. notebooks/training_monitor.ipynb - Analyze training progress
```

## Detailed Usage

### Environment Management

```bash
# Always activate your virtual environment first
source llm_training_env/bin/activate

# Install or update dependencies
pip install -r requirements.txt

# Check if everything is working
python scripts/verify_setup.py

# Deactivate when done
deactivate
```

### Individual Script Usage

```bash
# Download model only
python scripts/main.py --download

# Setup and test device only  
python scripts/main.py --setup

# Train model only (requires download first)
python scripts/main.py --train

# Test a specific trained model
python scripts/main.py --test outputs/gpt2_finetuned_20241228_143022

# List available models and outputs
python scripts/main.py --list-models
python scripts/main.py --list-outputs

# Validate your configuration
python scripts/main.py --validate-config
```

### Running with Custom Configurations

```bash
# Create custom config
cp configs/config.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your settings

# Run with custom config
python scripts/main.py --full --config configs/my_experiment.yaml
```

### Interactive Testing with Jupyter

Start Jupyter and open the interactive notebooks:

```bash
# Start Jupyter (make sure virtual environment is activated)
jupyter notebook
```

## Monitoring and Debugging

### Real-time Training Monitoring

```bash
tail -f outputs/gpt2_finetuned_*/training_log.txt

sudo powermetrics -n 1 -i 1000 --samplers gpu_power

htop
```

### Debug Commands

```bash
# Comprehensive system check
python scripts/verify_setup.py

# Test individual components
python scripts/download_model.py --config configs/config.yaml
python scripts/setup_device.py --test-only --config configs/config.yaml

# Validate configuration file
python scripts/main.py --validate-config

# List all available resources
python scripts/main.py --list-models
python scripts/main.py --list-outputs
```

## 📊 Model Outputs and Results

### Using Trained Models

#### In Python Scripts
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load your fine-tuned model
model_path = "outputs/gpt2_finetuned_20241228_143022"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Generate text
prompt = "Instruction: Explain neural networks.\nOutput:"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

#### In Jupyter Notebooks
The provided notebooks (`model_comparison.ipynb` and `training_monitor.ipynb`) automatically load and test your models.

## Troubleshooting Guide

```bash
# If pip install fails
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# If virtual environment issues
rm -rf llm_training_env
python3 -m venv llm_training_env
source llm_training_env/bin/activate
pip install -r requirements.txt

# If PyTorch MPS issues on macOS
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

## LoRA Training (Low-Rank Adaptation)

```bash
# Verify lora setup
python scripts/verify_lora_setup.py

# Train with LoRA using default config
python scripts/train_lora.py

# Train with custom LoRA config
python scripts/train_lora.py --config configs/config_lora.yaml

# Test existing LoRA adapter
python scripts/train_lora.py --test-only outputs/gpt2_lora_20250703_120000
```
