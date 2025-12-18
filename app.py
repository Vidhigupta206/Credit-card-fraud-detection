from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Credit Card Fraud Detection")

class Transaction(BaseModel):
    Time: float
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float

# Lazy load model and scaler
model = None
scaler = None

def load_artifacts():
    global model, scaler
    if model is None:
        model_path = '../models/best_fraud_model.pkl'
        scaler_path = '../models/amount_scaler.pkl'
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded!")

@app.get("/")
def home():
    return FileResponse('templates/index.html')

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        load_artifacts()
        
        data = pd.DataFrame([transaction.dict()])
        
        data['Amount_scaled'] = scaler.transform(data[['Amount']])
        data['Hour'] = (data['Time'] % (24 * 3600)) // 3600
        data['High_Amount'] = (data['Amount_scaled'] > 3).astype(int)
        data = data.drop(['Amount'], axis=1)
        
        prob = float(model.predict_proba(data)[0][1])
        is_fraud = bool(prob > 0.5)
        
        return {
            "fraud_probability": round(prob, 4),
            "is_fraud": is_fraud,
            "status": "FRAUD ALERT" if is_fraud else "Legitimate"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")