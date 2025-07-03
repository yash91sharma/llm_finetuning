# GPT-2 Fine-tuning Project

Practise codes for learning and revising the concepts by fine-tuning GPT-2 models with Apple Silicon (MPS) support.


## 🏁 Quick Start Guide

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

## Configuration Guide

The `configs/config.yaml` file controls all aspects of training. Here are the key sections:

### Model Configuration
```yaml
model:
  name: "gpt2"              # Model size: gpt2, gpt2-medium, gpt2-large, gpt2-xl
  save_path: "models/"      # Where to save downloaded models
  cache_dir: "models/cache/" # Hugging Face cache directory
```

### Training Configuration
```yaml
training:
  output_dir: "outputs/"              # Where to save fine-tuned models
  num_train_epochs: 3                 # Number of training epochs
  per_device_train_batch_size: 4      # Batch size per device
  gradient_accumulation_steps: 2      # Gradient accumulation
  learning_rate: 5e-5                 # Learning rate
  weight_decay: 0.01                  # Weight decay for regularization
  warmup_steps: 100                   # Learning rate warmup
  eval_steps: 100                     # How often to evaluate
  save_steps: 500                     # How often to save checkpoints
```

### Device Configuration
```yaml
device:
  use_mps: true     # Use Apple Silicon GPU (M1/M2/M3 Macs)
  use_cuda: false   # Use NVIDIA GPU (set to true for CUDA)
  # If both false, will use CPU
```

### Generation Configuration
```yaml
generation:
  max_new_tokens: 100    # Maximum tokens to generate
  temperature: 0.7       # Randomness (0.1 = focused, 1.0 = creative)
  do_sample: true        # Use sampling vs greedy decoding
```

## 🖥️ Device Support and Optimization

### Apple Silicon (M1/M2/M3 Macs)
```yaml
# Optimal settings for Apple Silicon
device:
  use_mps: true
  use_cuda: false

training:
  per_device_train_batch_size: 4    # Start here, increase if memory allows
  gradient_accumulation_steps: 2    # Effective batch size = 4 * 2 = 8
```

### CPU Only
```yaml
# CPU settings (slower but works everywhere)
device:
  use_mps: false
  use_cuda: false

training:
  per_device_train_batch_size: 2    # Smaller batches for CPU
  gradient_accumulation_steps: 4    # Compensate with more accumulation
```

### Interactive Testing with Jupyter

Start Jupyter and open the interactive notebooks:

```bash
# Start Jupyter (make sure virtual environment is activated)
jupyter notebook

# Open these notebooks:
# 1. model_comparison.ipynb - Side-by-side comparison
# 2. training_monitor.ipynb - Training analysis
```

**In model_comparison.ipynb you can:**
- Load both original and fine-tuned models
- Test custom prompts interactively
- Compare outputs with different parameters
- Save comparison results for later analysis

**In training_monitor.ipynb you can:**
- Visualize training loss curves
- Compare multiple training runs
- Analyze model performance metrics
- Monitor overfitting/underfitting

## 🔍 Monitoring and Debugging

### Real-time Training Monitoring

```bash
# Watch training progress in real-time
tail -f outputs/gpt2_finetuned_*/training_log.txt

# Monitor GPU/MPS usage (macOS)
sudo powermetrics -n 1 -i 1000 --samplers gpu_power

# Monitor CPU and memory
htop  # or Activity Monitor on macOS
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

## 🚀 Advanced Usage Scenarios

### Multiple Experiments

Run multiple training experiments with different configurations:

```bash
# Create experiment configs
cp configs/config.yaml configs/experiment_1.yaml
cp configs/config.yaml configs/experiment_2.yaml

# Edit each config with different parameters
# experiment_1.yaml: learning_rate: 3e-5, epochs: 5
# experiment_2.yaml: learning_rate: 1e-4, epochs: 3

# Run experiments
python scripts/main.py --full --config configs/experiment_1.yaml
python scripts/main.py --full --config configs/experiment_2.yaml

# Compare results in training_monitor.ipynb
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

#### As a Web API
You can deploy your model using frameworks like FastAPI:

```python
# api.py
from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()
model = GPT2LMHeadModel.from_pretrained("outputs/your_model_directory")
tokenizer = GPT2Tokenizer.from_pretrained("outputs/your_model_directory")

@app.post("/generate")
async def generate_text(prompt: str):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=100)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}

# Run with: uvicorn api:app --reload
```

## 🛠️ Troubleshooting Guide

### Installation Issues

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

## 🎯 LoRA Training (Low-Rank Adaptation)

### LoRA Training Usage

```bash
# Train with LoRA using default config
python scripts/train_lora.py

# Train with custom LoRA config
python scripts/train_lora.py --config configs/config_lora.yaml

# Test existing LoRA adapter
python scripts/train_lora.py --test-only outputs/gpt2_lora_20250703_120000
```

### LoRA Configuration

The `configs/config_lora.yaml` file contains LoRA-specific settings:

```yaml
# LoRA Configuration
lora:
  r: 16                          # Rank of adaptation (4, 8, 16, 32, 64)
  lora_alpha: 32                 # LoRA scaling parameter (usually 2*r)
  lora_dropout: 0.1              # Dropout probability for LoRA layers
  target_modules: ["c_attn", "c_proj", "c_fc"]  # Target modules for LoRA
  bias: "none"                   # Bias type ('none', 'all', 'lora_only')
```
