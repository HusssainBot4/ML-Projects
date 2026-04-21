import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from load_data import load_cmapss
from features import build_features

train_raw = load_cmapss('data/raw/train_FD001.txt')
train_df  = build_features(train_raw)

FEATURE_COLS = [c for c in train_df.columns if c not in ['unit', 'cycle', 'RUL', 'op1', 'op2', 'op3']]
X = train_df[FEATURE_COLS]
y = train_df['RUL']

xgb  = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
rf   = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
ridge = Ridge(alpha=10)

ensemble = VotingRegressor(estimators=[('xgb', xgb), ('rf', rf), ('ridge', ridge)])

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   ensemble),
])

pipeline.fit(X, y)

# Evaluation
preds = pipeline.predict(X)
print(f"Train RMSE: {np.sqrt(mean_squared_error(y, preds)):.2f}")
print(f"Train MAE:  {mean_absolute_error(y, preds):.2f}")

joblib.dump({'pipeline': pipeline, 'features': FEATURE_COLS}, 'models/model.pkl')
print("Model saved to models/model.pkl")