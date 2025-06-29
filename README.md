# GPT-2 Fine-tuning Project

Practise codes for fine-tuning GPT-2 models with Apple Silicon (MPS) support, configurable training parameters, and interactive model comparison capabilities.

## 📁 Project Structure

```
llm/
├── configs/
│   └── config.yaml             # Main configuration file
├── data/
│   ├── train_data.json         # Training data (instruction-output format)
│   └── eval_data.json          # Evaluation data (optional)
├── scripts/
│   ├── main.py                 # Main management script
│   ├── download_model.py       # Model download and caching
│   ├── setup_device.py         # Device setup and testing
│   ├── train_model.py          # Training pipeline
│   └── verify_setup.py         # Environment verification
├── notebooks/
│   ├── model_comparison.ipynb  # Interactive model testing
│   └── training_monitor.ipynb  # Training progress analysis
├── models/                     # Downloaded models and tokenizers
├── outputs/                    # Fine-tuned models with timestamps
├── llm_training_env/           # Virtual environment (created by you)
├── setup.py                    # One-click environment setup
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🏁 Quick Start Guide

### Step 1: Initial Setup

```bash
# Clone or navigate to the project directory
cd /path/to/llm

# Create virtual environment
python3 -m venv llm_training_env

# Activate virtual environment
# On macOS/Linux:
source llm_training_env/bin/activate

# Verify you're in the virtual environment
which python  # Should show path to llm_training_env

# Run automated setup (installs dependencies and verifies everything)
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

## Data Format and Preparation

### Data Structure Requirements

- **instruction**: The task or question (required)
- **input**: Additional context (optional, can be empty string)
- **output**: Expected response (required)


### Data Size Recommendations

- **Small dataset**: 100-1000 examples, 1-2 epochs
- **Medium dataset**: 1000-10000 examples, 2-3 epochs  
- **Large dataset**: 10000+ examples, 1-3 epochs

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

**Performance Tips for Apple Silicon:**
- Use MPS for 2-5x speedup over CPU
- Start with batch size 4, increase if you have enough memory
- Monitor memory usage with Activity Monitor
- Use `python scripts/setup_device.py --test-only` to verify MPS works

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

### Common Issues and Solutions

#### Memory Issues
```bash
# Reduce batch size in config.yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 4  # Increase to maintain effective batch size
```

#### MPS Issues on macOS
```bash
# Test MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# If MPS fails, fall back to CPU
device:
  use_mps: false
  use_cuda: false
```

#### Training Stalls or Crashes
```bash
# Validate your configuration
python scripts/main.py --validate-config

# Check data format
python -c "
import json
with open('data/train_data.json') as f:
    data = json.load(f)
print(f'Data samples: {len(data)}')
print(f'First sample: {data[0]}')
"

# Test device setup
python scripts/setup_device.py --test-only
```

#### Model Not Generating Good Outputs
```bash
# Try different generation parameters in config.yaml
generation:
  temperature: 0.3      # More focused (vs 0.7)
  max_new_tokens: 150   # Longer outputs (vs 100)
  
# Or train for more epochs
training:
  num_train_epochs: 5   # More training (vs 3)
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

### Different Model Sizes

```bash
# Train multiple model sizes
echo "name: 'gpt2'" > configs/small_model.yaml
cat configs/config.yaml | grep -v "name:" >> configs/small_model.yaml

echo "name: 'gpt2-medium'" > configs/medium_model.yaml  
cat configs/config.yaml | grep -v "name:" >> configs/medium_model.yaml

# Train both
python scripts/main.py --full --config configs/small_model.yaml
python scripts/main.py --full --config configs/medium_model.yaml
```

### Custom Data Preprocessing

If you need custom data preprocessing, modify the `CustomDataset` class in `scripts/train_model.py`:

```python
# Example: Add special formatting for your data
def __getitem__(self, idx):
    item = self.data[idx]
    
    # Custom formatting logic here
    if item.get('category') == 'question':
        text = f"Q: {item['instruction']}\nA: {item['output']}"
    else:
        text = f"Task: {item['instruction']}\nResult: {item['output']}"
    
    # Rest of tokenization code...
```

### Automated Training Pipeline

Create a script to run multiple training sessions:

```bash
#!/bin/bash
# automated_training.sh

configs=("config.yaml" "experiment_1.yaml" "experiment_2.yaml")

for config in "${configs[@]}"; do
    echo "Starting training with $config"
    python scripts/main.py --full --config "configs/$config"
    
    if [ $? -eq 0 ]; then
        echo "✅ Training with $config completed successfully"
    else
        echo "❌ Training with $config failed"
    fi
done

echo "All training sessions completed!"
```

## 📈 Performance Optimization

### Memory Optimization

```yaml
# For limited memory (8GB RAM or less)
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  dataloader_num_workers: 0
  
# For more memory (16GB+ RAM)  
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 1
  dataloader_num_workers: 2
```

### Speed Optimization

```yaml
# For faster training
training:
  fp16: true                    # Mixed precision (CUDA only)
  dataloader_pin_memory: true   # Faster data loading (CUDA only)
  remove_unused_columns: true   # Reduce memory usage
  
# Apple Silicon specific
device:
  use_mps: true
  # MPS doesn't support all optimizations, but still faster than CPU
```

### Quality Optimization

```yaml
# For better model quality
training:
  num_train_epochs: 5           # More epochs
  learning_rate: 3e-5           # Lower learning rate
  warmup_steps: 200             # More warmup
  weight_decay: 0.01            # Regularization
  
data:
  max_length: 1024              # Longer sequences
```

## 📊 Model Outputs and Results

### Output Directory Structure

After training, you'll find your models in timestamped directories:

```
outputs/
├── gpt2_finetuned_20241228_143022/
│   ├── pytorch_model.bin          # Model weights
│   ├── config.json                # Model configuration
│   ├── tokenizer.json             # Tokenizer
│   ├── tokenizer_config.json      # Tokenizer config
│   ├── training_args.bin          # Training arguments
│   ├── trainer_state.json         # Training state/metrics
│   ├── train_results.json         # Final training metrics
│   ├── eval_results.json          # Final evaluation metrics
│   └── checkpoint-X/              # Intermediate checkpoints
├── gpt2_finetuned_20241228_150315/
└── model_comparison_20241228_151022.json
```

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

### Training Issues

```bash
# If training is very slow
# Check if MPS/CUDA is being used
python -c "
import torch
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)
print(f'Model device: {next(model.parameters()).device}')
"

# If out of memory errors
# Reduce batch size in config.yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

### Model Quality Issues

```bash
# If model outputs are poor quality
# 1. Check your training data format
# 2. Increase training epochs
# 3. Adjust learning rate
# 4. Try different model sizes

# Test with original GPT-2 first
python scripts/main.py --test models/gpt2
```

### Common Error Messages

**"No module named 'transformers'"**
```bash
source llm_training_env/bin/activate
pip install transformers
```

**"TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'"**
```bash
# This is a compatibility issue with newer transformers versions
# The parameter has been renamed from 'evaluation_strategy' to 'eval_strategy'
# This has been fixed in the config file, but if you see this error:
# Update your config.yaml to use 'eval_strategy' instead of 'evaluation_strategy'
```

**"MPS backend out of memory"**
```bash
# Reduce batch size or use CPU
device:
  use_mps: false
```

**"CUDA out of memory"**
```bash
# Reduce batch size
training:
  per_device_train_batch_size: 2
```

## 🤝 Contributing and Customization

### Adding New Features

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes**
4. **Test thoroughly**: `python scripts/verify_setup.py`
5. **Update documentation**
6. **Submit a pull request**

### Customizing for Your Use Case

#### Different Data Formats
Modify the `CustomDataset` class in `scripts/train_model.py` to handle your specific data format.

#### Different Models
Add support for other models by modifying the model loading code in the scripts.

#### Different Training Strategies
Implement custom training loops by extending the training script.

### Project Structure for Extensions

```
llm/
├── configs/
│   ├── config.yaml
│   └── custom_configs/          # Your custom configurations
├── scripts/
│   ├── main.py
│   └── custom_scripts/          # Your additional scripts
├── notebooks/
│   ├── model_comparison.ipynb
│   └── custom_analysis/         # Your analysis notebooks
└── extensions/                  # Your custom extensions
    ├── custom_dataset.py
    ├── custom_trainer.py
    └── custom_evaluation.py
```

## 📚 Additional Resources

### Documentation Links
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [GPT-2 Model Card](https://huggingface.co/gpt2)

### Tutorials and Guides
- [Fine-tuning GPT-2 for Text Generation](https://huggingface.co/blog/how-to-generate)
- [Apple Silicon GPU Training Guide](https://pytorch.org/docs/stable/notes/mps.html)

### Community and Support
- [Hugging Face Forum](https://discuss.huggingface.co/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Stack Overflow GPT-2 Tag](https://stackoverflow.com/questions/tagged/gpt-2)

## 📄 License and Acknowledgments

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
- **Hugging Face** for the Transformers library and model hosting
- **OpenAI** for creating and releasing the GPT-2 model
- **PyTorch team** for Apple Silicon (MPS) support
- **Jupyter team** for interactive notebook capabilities

### Citation
If you use this toolkit in your research, please cite:

```bibtex
@software{gpt2_finetuning_toolkit,
  title={GPT-2 Fine-tuning Toolkit},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/llm}
}
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check this README** - Most common issues are covered
2. **Run diagnostics**: `python scripts/verify_setup.py`
3. **Validate config**: `python scripts/main.py --validate-config`
4. **Check logs** in the outputs directory
5. **Open an issue** on GitHub with:
   - Your operating system and Python version
   - Complete error messages
   - Your configuration file
   - Steps to reproduce the issue

**Happy fine-tuning! 🚀**

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
