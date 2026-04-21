from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title = "Predictive Maintenance Api")

model = joblib.load('models/model.pkl')
pipeline  = model['pipeline']
FEATURES  = model['features']




