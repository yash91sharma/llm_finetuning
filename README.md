# LLM Training Project - Usage Guide

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

## Jupyter Notebooks
- Start Jupyter Notebook: `jupyter notebook`
- Open `notebooks/model_comparison.ipynb` to compare models
- Open `notebooks/training_monitor.ipynb` to monitor training

## Monitoring Training
- To monitor training in real-time, run: `tail -f outputs/*/training_log.txt`
- To monitor system resources, run: `htop`
- For detailed GPU power metrics, run: `sudo powermetrics -n 1 -i 1000 --samplers gpu_power`
