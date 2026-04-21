from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="Predictive Maintenance API")

# Allow dashboard (Streamlit) to call the API without CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'models' / 'model.pkl'

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
    )

model    = joblib.load(str(MODEL_PATH))
pipeline = model['pipeline']
FEATURES = model['features']

class SensorReading(BaseModel):
    unit: int
    cycle: int
    sensors: dict

@app.post("/predict")
def predict(reading: SensorReading):
    row = pd.DataFrame([reading.sensors])
    for col in FEATURES:
        if col not in row.columns:
            row[col] = 0
    row = row[FEATURES]

    rul_pred     = float(pipeline.predict(row)[0])
    failure_prob = float(1 / (1 + np.exp((rul_pred - 30) / 10)))

    return {
        "unit": reading.unit,
        "predicted_RUL": round(rul_pred, 1),
        "failure_probability": round(failure_prob, 4),
        "status": "critical" if failure_prob > 0.7 else "warning" if failure_prob > 0.4 else "healthy"
    }

@app.get("/health")
def health():
    return {"status": "ok"}