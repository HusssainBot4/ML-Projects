from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="Predictive Maintenance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'models' / 'model.pkl'

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")

artifact = joblib.load(str(MODEL_PATH))
pipeline = artifact['pipeline']
FEATURES = artifact['features']

# ── Build a fallback value per feature from the imputer's learned medians ──
# The pipeline's first step is a SimpleImputer(strategy='median').
# After fitting, imputer.statistics_ holds the median of each feature
# from training data. We use these as defaults for missing features
# instead of 0, which was completely out of distribution.
imputer         = pipeline.named_steps['imputer']
FEATURE_MEDIANS = dict(zip(FEATURES, imputer.statistics_))

class SensorReading(BaseModel):
    unit:    int
    cycle:   int
    sensors: dict

@app.post("/predict")
def predict(reading: SensorReading):
    row = pd.DataFrame([reading.sensors])

    # Fill every missing feature with its training median (not 0)
    for col in FEATURES:
        if col not in row.columns:
            row[col] = FEATURE_MEDIANS[col]

    row      = row[FEATURES]
    rul_pred = float(pipeline.predict(row)[0])
    rul_pred = max(0.0, rul_pred)                    # never negative

    failure_prob = float(1 / (1 + np.exp((rul_pred - 30) / 10)))

    return {
        "unit":                reading.unit,
        "predicted_RUL":       round(rul_pred, 1),
        "failure_probability": round(failure_prob, 4),
        "status": (
            "critical" if failure_prob > 0.7 else
            "warning"  if failure_prob > 0.4 else
            "healthy"
        ),
    }

@app.get("/health")
def health():
    return {"status": "ok"}