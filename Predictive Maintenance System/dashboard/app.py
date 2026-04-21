import streamlit as st
import requests, random, time

st.set_page_config(page_title="Machine Health Dashboard", layout="wide")
st.title("Predictive Maintenance Dashboard")

API_URL = "http://localhost:8000/predict"
MACHINES = [1, 2, 3, 4, 5]

def fake_sensor_reading(unit):
    return {
        "unit": unit,
        "cycle": random.randint(50, 300),
        "sensors": {f"s{i}": round(random.uniform(0.1, 600), 2) for i in range(1, 22)}
    }

def get_color(prob):
    if prob > 0.7: return "🔴"
    if prob > 0.4: return "🟡"
    return "🟢"

cols = st.columns(len(MACHINES))
placeholders = [col.empty() for col in cols]

if st.button("Start Live Monitoring"):
    for _ in range(20):  
        for i, unit in enumerate(MACHINES):
            payload = fake_sensor_reading(unit)
            try:
                resp = requests.post(API_URL, json=payload, timeout=2).json()
                prob = resp['failure_probability']
                rul  = resp['predicted_RUL']
                icon = get_color(prob)
                placeholders[i].metric(
                    label=f"{icon} Machine {unit}",
                    value=f"{prob*100:.1f}% risk",
                    delta=f"RUL: {rul} cycles"
                )
            except Exception as e:
                placeholders[i].error(f"M{unit}: API error")
        time.sleep(2)