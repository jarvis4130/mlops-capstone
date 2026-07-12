import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb

from preprocess import FEATURES

app = FastAPI(title="House Price Prediction API")

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

MODEL_DIR = "models"
encoder = pickle.load(open(f"{MODEL_DIR}/encoder.pkl", "rb"))
model = xgb.XGBRegressor()
model.load_model(f"{MODEL_DIR}/model.json")

class HouseFeatures(BaseModel):
    Area: float
    Location: str
    No_of_Bedrooms: int
    Resale: int
    MaintenanceStaff: int
    Gymnasium: int
    SwimmingPool: int
    LiftAvailable: int
    CarParking: int
    City: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HouseFeatures):
    data = {
        "Area": features.Area,
        "Location": features.Location,
        "No. of Bedrooms": features.No_of_Bedrooms,
        "Resale": features.Resale,
        "MaintenanceStaff": features.MaintenanceStaff,
        "Gymnasium": features.Gymnasium,
        "SwimmingPool": features.SwimmingPool,
        "LiftAvailable": features.LiftAvailable,
        "CarParking": features.CarParking,
        "City": features.City,
    }

    import pandas as pd
    df = pd.DataFrame([data])
    df[["Location", "City"]] = encoder.transform(df[["Location", "City"]])

    pred = model.predict(df[FEATURES])
    price = np.expm1(pred[0])

    return {"predicted_price": round(float(price), 2)}
