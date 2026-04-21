from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title = "Predictive Maintenance Api")

model = joblib.load('models/model.pkl')
pipeline  = model['pipeline']
FEATURES  = model['features']

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
    
    rul_pred = float(pipeline.predict(row)[0])
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


