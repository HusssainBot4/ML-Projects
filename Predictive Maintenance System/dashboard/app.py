import streamlit as st
import requests, random, time
import numpy as np

st.set_page_config(page_title="Machine Health Dashboard", layout="wide")
st.title("Predictive Maintenance Dashboard")

API_URL  = "http://localhost:8000/predict"
MACHINES = [1, 2, 3, 4, 5]
RUL_CAP  = 125

# Realistic CMAPSS FD001 sensor base ranges (from actual dataset statistics)
SENSOR_RANGES = {
    "s1":  (518.67, 518.67), "s2":  (641.82, 644.53), "s3":  (1585.3, 1596.8),
    "s4":  (1400.6, 1413.2), "s5":  (14.62,  14.62),  "s6":  (21.61,  21.61),
    "s7":  (553.90, 556.10), "s8":  (2387.9, 2392.1), "s9":  (9044.0, 9065.0),
    "s10": (1.30,   1.30),   "s11": (47.10,  47.54),  "s12": (521.40, 523.10),
    "s13": (2387.9, 2392.1), "s14": (8099.0, 8155.0), "s15": (8.3195, 8.4895),
    "s16": (0.03,   0.03),   "s17": (391.0,  396.0),  "s18": (2388.0, 2388.0),
    "s19": (100.0,  100.0),  "s20": (38.69,  39.08),  "s21": (23.18,  23.48),
}

# Sensors that actually degrade (have variance in FD001)
DEGRADING = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]

# Track cycle per machine so it increases over time
if "cycles" not in st.session_state:
    st.session_state.cycles = {m: random.randint(10, 50) for m in MACHINES}

def fake_sensor_reading(unit, cycle):
    """
    Generate a realistic sensor snapshot.
    Sensors drift toward the HIGH end of their range as cycle increases,
    simulating degradation. This keeps values in-distribution for the model.
    """
    degradation = min(cycle / RUL_CAP, 1.0)   # 0.0 = new, 1.0 = end of life
    sensors = {}
    for i in range(1, 22):
        key    = f"s{i}"
        lo, hi = SENSOR_RANGES[key]
        base   = random.uniform(lo, hi)
        if key in DEGRADING:
            drift = (hi - lo) * degradation * 0.6
            noise = random.gauss(0, (hi - lo) * 0.03 + 0.001)
            sensors[key] = round(base + drift + noise, 2)
        else:
            sensors[key] = round(base, 2)   # fixed sensors stay constant

    return {"unit": unit, "cycle": cycle, "sensors": sensors}

def get_color(prob):
    if prob > 0.7: return "🔴"
    if prob > 0.4: return "🟡"
    return "🟢"

cols        = st.columns(len(MACHINES))
placeholders = [col.empty() for col in cols]

if st.button("Start Live Monitoring"):
    for _ in range(30):
        for i, unit in enumerate(MACHINES):
            st.session_state.cycles[unit] += 1
            cycle   = st.session_state.cycles[unit]
            payload = fake_sensor_reading(unit, cycle)

            try:
                resp = requests.post(API_URL, json=payload, timeout=2).json()
                rul  = max(0.0, resp["predicted_RUL"])            # never negative
                prob = min(1.0, max(0.0, resp["failure_probability"]))
            except Exception:
                # API offline → simulate locally so UI still works
                rul  = max(0.0, RUL_CAP - cycle + random.gauss(0, 3))
                prob = float(1 / (1 + np.exp((rul - 30) / 10)))

            icon = get_color(prob)
            placeholders[i].metric(
                label = f"{icon} Machine {unit}  (cycle {cycle})",
                value = f"{prob*100:.1f}% risk",
                delta = f"RUL: {rul:.1f} cycles",
            )
        time.sleep(2)