import pandas as pd
import numpy as np

SENSOR_COLS = [f's{i}' for i in range(1, 22)]

# Computing Remaining Useful Life per unit
def add_rul(df):
    max_cycle = df.groupby('unit')['cycle'].max().rename('max_cycle')
    df = df.join(max_cycle, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']
    return df.drop(columns='max_cycle')

# Rolling mean and std per unit per sensor
def add_rolling_features(df, windows=[5, 10, 20]):
    
    for w in windows:
        for col in SENSOR_COLS:
            grp = df.groupby('unit')[col]
            df[f'{col}_roll_mean_{w}'] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f'{col}_roll_std_{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    return df

# Lagged sensor readings
def add_lag_features(df, lags=[1, 2, 5]):
    for lag in lags:
        for col in SENSOR_COLS:
            df[f'{col}_lag{lag}'] = df.groupby('unit')[col].shift(lag).fillna(method='bfill')
    return df

# Simple moving average smoothing
def reduce_noise(df, window=3):
    for col in SENSOR_COLS:
        df[col] = df.groupby('unit')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    return df

def build_features(df):
    df = add_rul(df)
    df = reduce_noise(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = df.dropna()
    return df