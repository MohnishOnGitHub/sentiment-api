from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
app = FastAPI(title="Sentiment Analysis API")

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception:
    pipeline = None

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text")

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float | None = None

@app.get("/")
def root():
    return {"message": "Sentiment API running. See /docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    pred = pipeline.predict([request.text])[0]
    label = "positive" if pred == 1 else "negative"
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        confidence = float(pipeline.predict_proba([request.text]).max())
    return SentimentResponse(sentiment=label, confidence=confidence)