import logging
import yaml
from torch.utils.data import Dataset


def setup_logging(log_level="INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class CustomDataset(Dataset):
    """Custom dataset for GPT-2 fine-tuning."""

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if item.get("input", "").strip():
            text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}"
        else:
            text = f"Instruction: {item['instruction']}\nOutput: {item['output']}"
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
