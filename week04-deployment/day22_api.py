from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import json

# Initialize FastAPI app
app = FastAPI(
    title="SLM API",
    description="Small Language Model API — Built from scratch",
    version="1.0.0"
)

print("Loading model...")

# Load GPT2 for local testing (lighter than Phi-2)
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
model.eval()
print("Model loaded ✅")

# =====================
# Request/Response Models
# =====================

class GenerateRequest(BaseModel):
    instruction: str
    input_text: str = ""
    max_tokens: int = 100
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    instruction: str
    response: str
    tokens_generated: int
    time_seconds: float
    model: str

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str

# =====================
# API Endpoints
# =====================

@app.get("/")
def root():
    return {
        "message": "SLM API is running!",
        "docs": "/docs",
        "health": "/health",
        "generate": "/generate"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        start_time = time.time()

        # Build prompt in instruction format
        if request.input_text:
            prompt = f"""### Instruction:
{request.instruction}

### Input:
{request.input_text}

### Response:"""
        else:
            prompt = f"""### Instruction:
{request.instruction}

### Response:"""

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3
            )

        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = full_text[len(prompt):].strip()

        end_time = time.time()

        return GenerateResponse(
            instruction=request.instruction,
            response=response_text,
            tokens_generated=len(outputs[0]) - len(inputs['input_ids'][0]),
            time_seconds=round(end_time - start_time, 2),
            model=MODEL_NAME
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify(request: GenerateRequest):
    request.instruction = "Classify this news article into a category."
    request.max_tokens = 20
    return generate(request)

@app.get("/models")
def list_models():
    return {
        "current_model": MODEL_NAME,
        "available_models": [
            {"name": "gpt2", "params": "124M", "use_case": "testing"},
            {"name": "gpt2-medium", "params": "345M", "use_case": "light production"},
            {"name": "microsoft/phi-2", "params": "2.7B", "use_case": "production"},
            {"name": "mistralai/Mistral-7B", "params": "7B", "use_case": "enterprise"},
        ]
    }

# =====================
# Test the API locally
# =====================

if __name__ == "__main__":
    import uvicorn
    print("\nStarting SLM API server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)