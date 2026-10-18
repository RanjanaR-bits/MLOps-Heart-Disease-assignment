"""FastAPI service: POST /predict, GET /health, GET /metrics (Prometheus)."""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.config import MODEL_PATH, RAW_FEATURES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart_api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("api_predictions_total", "Predictions served", ["label"])

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", str(MODEL_PATH))
    app.state.model = joblib.load(path)
    logger.info("Loaded model from %s (version %s)", path, MODEL_VERSION)
    yield


app = FastAPI(title="Heart Disease Risk API", version=MODEL_VERSION, lifespan=lifespan)


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type")
    trestbps: float = Field(..., ge=50, le=260, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=40, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: Optional[Literal[0, 1, 2, 3]] = Field(None, description="Major vessels (0-3), optional")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="Thalassemia, optional")

    model_config = {"json_schema_extra": {"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}}}


class Prediction(BaseModel):
    prediction: int
    label: str
    risk_probability: float
    confidence: float
    model_version: str


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    route = request.scope.get("route")
    endpoint = route.path if route else "unmatched"
    REQUESTS.labels(request.method, endpoint, str(response.status_code)).inc()
    LATENCY.labels(endpoint).observe(elapsed)
    logger.info("%s %s -> %s in %.1f ms", request.method, request.url.path,
                response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model"),
            "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    row = pd.DataFrame([patient.model_dump()], columns=RAW_FEATURES).astype(float)
    proba = float(app.state.model.predict_proba(row)[0, 1])
    pred = int(proba >= 0.5)
    label = "heart disease likely" if pred else "heart disease unlikely"
    PREDICTIONS.labels(label).inc()
    logger.info("prediction=%d risk=%.3f input=%s", pred, proba, patient.model_dump())
    return Prediction(prediction=pred, label=label, risk_probability=round(proba, 4),
                      confidence=round(max(proba, 1 - proba), 4),
                      model_version=MODEL_VERSION)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
