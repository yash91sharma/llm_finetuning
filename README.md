# LLM Training Project - Usage Guide

## Environment Setup
- Create a virtual environment: `python3 -m venv llm_training_env`
- Activate the virtual environment: `source llm_training_env/bin/activate`
- Upgrade pip: `pip install --upgrade pip`
- Run the setup script: `python setup.py`

## Data
- Place your training data in: `data/train_data.json`
- Place your evaluation data in: `data/eval_data.json`

## Configuration Files
- Standard configuration file is located at: `configs/config.yaml`
- LoRA configuration file is located at: `configs/config_lora.yaml`
- To create a custom configuration file, run: `cp configs/config.yaml configs/my_experiment.yaml`
- Edit `configs/my_experiment.yaml` as needed for your experiment

## Model Download
- To download the model, run: `python scripts/main.py --download [--config CONFIG_PATH]`
- Alternatively, run: `python scripts/download_model.py --config CONFIG_PATH`

## Device Setup
- To setup the device, run: `python scripts/main.py --setup [--config CONFIG_PATH]`
- Alternatively, run: `python scripts/setup_device.py --config CONFIG_PATH`

## Standard Training (Full Pipeline)
- To run the full training pipeline, use: `python scripts/main.py --full [--config CONFIG_PATH]`
- To train the model, use: `python scripts/main.py --train [--config CONFIG_PATH]`
- Alternatively, run: `python scripts/train_model.py --config CONFIG_PATH`

## LoRA Training
- To verify LoRA setup, run: `python scripts/verify_lora_setup.py`
- To run the full pipeline with LoRA, use: `python scripts/main.py --full-lora [--config CONFIG_PATH]`
- To train with LoRA, use: `python scripts/main.py --train-lora [--config CONFIG_PATH]`
- Alternatively, run: `python scripts/train_lora.py --config CONFIG_PATH`

## Test Trained Model
- To test a trained model, run: `python scripts/main.py --test OUTPUT_MODEL_DIR [--config CONFIG_PATH]`
- Alternatively, run: `python scripts/train_model.py --test-only OUTPUT_MODEL_DIR --config CONFIG_PATH`

## Test LoRA Adapter
- To test a LoRA adapter, run: `python scripts/main.py --test-lora OUTPUT_LORA_DIR [--config CONFIG_PATH]`
- Alternatively, run: `python scripts/train_lora.py --test-only OUTPUT_LORA_DIR --config CONFIG_PATH`

## List Models and Outputs
- To list all models, run: `python scripts/main.py --list-models`
- To list all outputs, run: `python scripts/main.py --list-outputs`

## Validate Config
- To validate the configuration file, run: `python scripts/main.py --validate-config [--config CONFIG_PATH]`

## Jupyter Notebooks
- Activate your virtual environment: `source llm_training_env/bin/activate`
- Start Jupyter Notebook: `jupyter notebook`
- Open `notebooks/model_comparison.ipynb` to compare models
- Open `notebooks/training_monitor.ipynb` to monitor training

## Monitoring Training
- To monitor training in real-time, run: `tail -f outputs/*/training_log.txt`
- To monitor system resources, run: `htop`
- For detailed GPU power metrics, run: `sudo powermetrics -n 1 -i 1000 --samplers gpu_power`

## Troubleshooting
- If `pip install` fails, run: `pip install --upgrade pip` and `pip install --no-cache-dir -r requirements.txt`
- If virtual environment issues occur, run: `rm -rf llm_training_env`, then recreate and reactivate the environment, and reinstall requirements
- If PyTorch MPS issues occur, run: `pip uninstall torch torchvision torchaudio` and then `pip install torch torchvision torchaudio`

## Python Inference Example
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
