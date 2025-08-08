# LLM Training Project - Usage Guide

## Objective
Finetune the GPT2 model with data about "Jalal the cat", and then serve the new model and old models. The new fine-tuned model should correctly answer questions about Jalal the cat's space adventures, as mentioned in the training data. While the original GPT2 model should not be able to.

## Environment Setup (One Time)
- Create a virtual environment: `python3 -m venv llm_training_env`

## Environment Setup (Everytime)
- Activate the virtual environment: `source llm_training_env/bin/activate`
- Upgrade pip: `pip install --upgrade pip`
- Run the setup script: `python scripts/setup.py`
(this installs all the dependencies from the requirements.txt file too)

## Data
- Place your training data in: `data/train_data.json`
- Place your evaluation data in: `data/eval_data.json`

## Configuration Files
- All layers fine-tuning config file: `configs/config.yaml`
- LoRA config file: `configs/config_lora.yaml`

## Model Download
- To download the model, run: `python scripts/download_model.py --config configs/training/config_lora.yaml`

## Device Setup
- To setup the device, run: `python scripts/setup_device.py --config configs/training/config_lora.yaml`

## Full fine-tuning (All layers)
- To train the model, use: `python scripts/train.py --config configs/training/config.yaml`

## LoRA Training
- To verify LoRA setup, run: `python scripts/verify_lora_setup.py`
- To train with LoRA, use: `python scripts/train_lora.py --config configs/training/config_lora.yaml`

## List Models and Outputs
- To list all models or outputs, check the `models/` and `outputs/` directories directly.

## Model serving
- Update the latest finetuned model directory in the serving config file.
- Start the server: `python server.py`
- Go to: `http://0.0.0.0:8000/docs`
- Try the two endpoints. `Generate` uses the new fine-tuned model, and `Generate_base` uses the original GPT2 model.
