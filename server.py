import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "configs/serving/config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)
MODEL_DIR = config["model_dir"]


# Load finetuned model
try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load finetuned model from {MODEL_DIR}: {e}")

# Load base GPT-2 model for comparison
BASE_MODEL_DIR = Path(__file__).parent / "models/gpt2"
BASE_TOKENIZER_DIR = Path(__file__).parent / "models/gpt2-tokenizer"
if not BASE_MODEL_DIR.exists():
    raise RuntimeError(f"Base GPT-2 model directory does not exist: {BASE_MODEL_DIR}")
if not BASE_TOKENIZER_DIR.exists():
    raise RuntimeError(f"Base GPT-2 tokenizer directory does not exist: {BASE_TOKENIZER_DIR}")
try:
    base_tokenizer = GPT2Tokenizer.from_pretrained(str(BASE_TOKENIZER_DIR.resolve()))
    base_model = GPT2LMHeadModel.from_pretrained(str(BASE_MODEL_DIR.resolve()))
    base_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load base GPT-2 model or tokenizer: {e}")

app = FastAPI(title="LLM Serving API")


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class GenerationResponse(BaseModel):
    generated_text: str



# Endpoint for finetuned model
@app.post("/generate", response_model=GenerationResponse)
def generate_text(req: GenerationRequest):
    try:
        input_ids = tokenizer.encode(req.prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        return GenerationResponse(generated_text=generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for base GPT-2 model
@app.post("/generate_base", response_model=GenerationResponse)
def generate_base_text(req: GenerationRequest):
    try:
        input_ids = base_tokenizer.encode(req.prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = base_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
                pad_token_id=base_tokenizer.pad_token_id,
            )
        generated = base_tokenizer.decode(output[0], skip_special_tokens=True)
        return GenerationResponse(generated_text=generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
