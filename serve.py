"""
FastAPI server to serve a fine-tuned GPT-2 model for text generation.
Specify the model directory in config.json.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).parent / "server_config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)
MODEL_DIR = config["model_dir"]

# Load model and tokenizer
try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_DIR}: {e}")

app = FastAPI(title="LLM Fine-tuned GPT-2 Serving API")


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.95
    do_sample: bool = True


class GenerationResponse(BaseModel):
    generated_text: str


@app.post("/generate", response_model=GenerationResponse)
def generate_text(req: GenerationRequest):
    try:
        input_ids = tokenizer.encode(req.prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        return GenerationResponse(generated_text=generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
