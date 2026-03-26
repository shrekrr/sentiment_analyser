import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from frontend.inference import SentimentInference
from explain.shap_explain import explain_with_shap

app = FastAPI(title="Multi-Task Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Engine
# Point to custom checkpoint via MODEL_CHECKPOINT env variable. 
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT", None)
inference_engine = SentimentInference(checkpoint_path=CHECKPOINT_PATH)

class PredictRequest(BaseModel):
    text: str

class BatchPredictRequest(BaseModel):
    texts: List[str]

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "loaded" if inference_engine.model else "error"}

@app.post("/predict")
def predict_single(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        return inference_engine.predict(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
def predict_batch(req: BatchPredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")
    try:
        return inference_engine.predict_batch(req.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain_text(req: PredictRequest):
    """
    WARNING: SHAP computation on transformer models is very slow. 
    Expect 10-30 seconds per request depending on text length.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        result = explain_with_shap(inference_engine.model, inference_engine.tokenizer, [req.text], task="sentiment")
        return result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Note: The 'frontend/static' mount must occur last so it works as a fallback router
app.mount("/", StaticFiles(directory="frontend/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("frontend.app:app", host="0.0.0.0", port=8000, reload=True)
