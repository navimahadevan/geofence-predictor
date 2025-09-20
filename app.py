# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib

app = FastAPI(title="Geofence Risk API")

# Load trained model
try:
    model = joblib.load("geofence_risk_model.pkl")
except Exception as e:
    model = None
    print("WARNING: model not loaded:", e)

# Input schema
class PredictRequest(BaseModel):
    latitude: float
    longitude: float
    timestamp: str
    crime_rate: float
    geo_risk: float
    crowd_density: float
    restricted_zone: int

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Parse timestamp → extract hour
    try:
        ts = req.timestamp.replace("Z", "+00:00") if req.timestamp.endswith("Z") else req.timestamp
        dt = datetime.fromisoformat(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    hour = dt.hour

    # Feature vector
    features = [[
        req.latitude,
        req.longitude,
        hour,
        req.crime_rate,
        req.geo_risk,
        req.crowd_density,
        req.restricted_zone
    ]]

    # Prediction
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    # Probability of the predicted class = risk score
    pred_index = list(model.classes_).index(pred)
    risk_score = float(probs[pred_index])

    return {"risk_score": risk_score}
