# app.py

import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import normalize, StandardScaler

app = FastAPI()

# Load trained model
with open("model (1).pkl", "rb") as f:
    model = pickle.load(f)

# Define input schema (29 features)
class Transaction(BaseModel):
    features: list[float]  # must contain 29 values

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API running"}

@app.post("/predict")
def predict(transaction: Transaction):

    if len(transaction.features) != 29:
        return {"error": "Input must contain exactly 29 features"}

    input_array = np.array(transaction.features).reshape(1, -1)

    # Apply L1 normalization (same as training)
    input_array = normalize(input_array, norm="l1")

    prediction = model.predict(input_array)[0]

    return {
        "fraud_prediction": int(prediction),
        "meaning": "Fraud" if prediction == 1 else "Legitimate"
    }