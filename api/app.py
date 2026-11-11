import os
import csv
import json
import time
from datetime import datetime, timezone
from typing import List, Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.pipeline import explain
from src.predict import decide_label

MODEL_PATH = os.environ.get("MODEL_PATH", "models/sentiment_lr.joblib")
METADATA_PATH = os.environ.get("METADATA_PATH", "models/model_metadata.json")
LOG_PATH = os.environ.get("PREDICTION_LOG_PATH", "logs/predictions.csv")

app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="End-to-End Applied Machine Learning REST API for IMDB Movie Review Sentiment Classification",
    version="1.0.0",
)

_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found at {MODEL_PATH}. Run `python src/train.py` first.",
            )
        _pipe = joblib.load(MODEL_PATH)
    return _pipe


def log_prediction(text: str, prediction: str, confidence: float, latency_ms: float):
    """Append prediction request metadata to CSV log file for monitoring."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text_snippet", "word_count", "prediction", "confidence", "latency_ms"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            text[:120].replace("\n", " "),
            len(text.split()),
            prediction,
            confidence,
            latency_ms,
        ])


class Review(BaseModel):
    """Supports both `{"text": "..."}` (guide format) and `{"features": ["..."]}` (rubric format)."""

    text: Optional[str] = Field(default=None, description="Movie review text to classify")
    features: Optional[List[str]] = Field(default=None, description="Alternative input format: list containing review text")

    @model_validator(mode="after")
    def validate_non_empty_review(self):
        resolved = self.text
        if resolved is None and self.features is not None:
            if len(self.features) == 0 or not isinstance(self.features[0], str):
                raise ValueError("`features` must be a non-empty list containing a review string.")
            resolved = self.features[0]
        if resolved is None or not str(resolved).strip():
            raise ValueError("Review text must not be empty (`text` or `features` required, min_length=1).")
        self.text = str(resolved).strip()
        return self


class BatchReviewRequest(BaseModel):
    instances: List[Union[str, List[str]]] = Field(..., min_length=1, description="List of review strings to classify")


@app.get("/health")
def health():
    return {"status": "ok", "model": "TF-IDF + Logistic Regression"}


@app.get("/model-info")
def model_info():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "model_name": "TF-IDF + MetaFeatures + Logistic Regression (Tuned)",
        "version": "1.0.0",
        "training_date": "2026-11-10",
        "dataset": "IMDB Dataset of 50K Movie Reviews (deduplicated)",
        "metrics": {"test_f1": 0.909, "test_accuracy": 0.908},
    }


@app.post("/predict")
def predict(review: Review):
    pipe = get_pipeline()
    t0 = time.time()
    proba = pipe.predict_proba(pd.DataFrame({"review": [review.text]}))[0]
    conf = float(proba.max())
    label, mixed_reason, clause_info = decide_label(review.text, float(proba[1]), pipe)
    latency_ms = round((time.time() - t0) * 1000, 2)
    log_prediction(review.text, label, round(conf, 3), latency_ms)
    result = {
        "prediction": label,
        "confidence": round(conf, 3),
        "latency_ms": latency_ms,
    }
    if mixed_reason == "clauses_disagree":
        result["clauses"] = clause_info["clauses"]
    return result


@app.post("/explain")
def explain_endpoint(review: Review):
    pipe = get_pipeline()
    t0 = time.time()
    proba = pipe.predict_proba(pd.DataFrame({"review": [review.text]}))[0]
    conf = float(proba.max())
    label, mixed_reason, clause_info = decide_label(review.text, float(proba[1]), pipe)
    exp = explain(review.text, pipe=pipe, top=6)
    latency_ms = round((time.time() - t0) * 1000, 2)
    result = {
        "prediction": label,
        "confidence": round(conf, 3),
        "latency_ms": latency_ms,
        "explanation": exp,
    }
    if clause_info["clauses"]:
        result["clauses"] = clause_info["clauses"]
    return result


@app.post("/batch-predict")
def batch_predict(batch: BatchReviewRequest):
    pipe = get_pipeline()
    texts = []
    for item in batch.instances:
        val = item[0] if isinstance(item, list) and len(item) > 0 else item
        if not isinstance(val, str) or not val.strip():
            raise HTTPException(status_code=422, detail="All instances in batch must be non-empty strings.")
        texts.append(val.strip())

    t0 = time.time()
    probas = pipe.predict_proba(pd.DataFrame({"review": texts}))
    total_ms = (time.time() - t0) * 1000
    per_item_ms = round(total_ms / len(texts), 2)

    predictions = []
    for text, proba in zip(texts, probas):
        conf = float(proba.max())
        label, _, _ = decide_label(text, float(proba[1]), pipe)
        log_prediction(text, label, round(conf, 3), per_item_ms)
        predictions.append({
            "prediction": label,
            "confidence": round(conf, 3),
        })
    return {"predictions": predictions, "count": len(predictions), "total_latency_ms": round(total_ms, 2)}

