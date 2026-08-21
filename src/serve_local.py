"""Serve API don gian cho local - doc model tu file local."""
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Use local model file (khong can GCS khi chay local)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")

# Load model ngay khi import
print(f"Loading model from {MODEL_PATH}...", flush=True)
model = joblib.load(MODEL_PATH)
print(f"Model loaded: {type(model).__name__}", flush=True)


def _add_features(features: list) -> pd.DataFrame:
    """Tao DataFrame 1 dong va tinh derived features giong train.py."""
    columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "wine_type",
    ]
    df = pd.DataFrame([features], columns=columns)

    df["sulphate_to_chloride"] = df["sulphates"] / (df["chlorides"] + 1e-9)
    df["free_to_total_so2"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-9)
    df["density_to_alcohol"] = df["density"] / (df["alcohol"] + 1e-9)
    df["acidity_sum"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    df["sugar_to_alcohol"] = df["residual sugar"] / (df["alcohol"] + 1e-9)
    df["ph_times_acidity"] = df["pH"] * df["acidity_sum"]
    df["so2_per_alcohol"] = (df["total sulfur dioxide"] + df["free sulfur dioxide"]) / (df["alcohol"] + 1e-9)

    return df


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Kiem tra suc khoe server."""
    return {"status": "ok", "model": type(model).__name__}


@app.post("/predict")
def predict(req: PredictRequest):
    """Suy luan chat luong ruou (0=thap, 1=trung_binh, 2=cao)."""
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail=f"Expected 12 raw features, got {len(req.features)}")
    df = _add_features(req.features)
    pred = model.predict(df)[0]
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = label_map.get(int(pred), "khong_xac_dinh")
    return {"prediction": int(pred), "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)