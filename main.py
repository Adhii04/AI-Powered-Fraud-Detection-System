from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import joblib
import pandas as pd
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time Machine Learning API for detecting fraudulent transactions.",
    version="1.0.0"
)

# Load the trained model pipeline on startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model_smote.pkl")

try:
    print(f"Loading ML Model from {MODEL_PATH}...")
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"WARNING: Could not load model. Ensure Phase 2 is complete. Error: {e}")
    model_pipeline = None

# Define the expected input schema using Pydantic
class TransactionInput(BaseModel):
    Amount: float = Field(..., description="The total transaction amount")
    Category: str = Field(..., description="The merchant category (e.g. electronics, dining)")
    AnomalyScore: float = Field(..., description="Pre-calculated anomaly score from upstream systems")
    Age: int = Field(..., description="Age of the customer")
    AccountBalance: float = Field(..., description="Current account balance before transaction")
    SuspiciousFlag: int = Field(..., description="0 if normal, 1 if flagged by basic rules")
    Location: str = Field(..., description="Location of the transaction")
    Timestamp: str = Field(..., description="Datetime string in format YYYY-MM-DD HH:MM:SS")

    class Config:
        json_schema_extra = {
            "example": {
                "Amount": 499.99,
                "Category": "electronics",
                "AnomalyScore": 0.85,
                "Age": 32,
                "AccountBalance": 12500.50,
                "SuspiciousFlag": 0,
                "Location": "New York, USA",
                "Timestamp": "2024-08-08 14:30:00"
            }
        }

@app.post("/predict")
async def predict_fraud(transaction: TransactionInput) -> Dict[str, Any]:
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="ML Model is not loaded on the server.")
    
    try:
        # 1. Parse the timestamp for our feature engineering requirements
        dt = pd.to_datetime(transaction.Timestamp)
        hour = dt.hour
        day_of_week = dt.dayofweek
        
        # 2. Reconstruct the Dataframe exactly as the pipeline expects
        input_df = pd.DataFrame([{
            'Amount': transaction.Amount,
            'Category': transaction.Category,
            'AnomalyScore': transaction.AnomalyScore,
            'Age': transaction.Age,
            'AccountBalance': transaction.AccountBalance,
            'SuspiciousFlag': transaction.SuspiciousFlag,
            'Location': transaction.Location,
            'Hour': hour,
            'DayOfWeek': day_of_week
        }])
        
        # 3. Get predictions
        prediction = model_pipeline.predict(input_df)[0]
        probabilities = model_pipeline.predict_proba(input_df)[0]
        
        # 4. Format the response
        return {
            "is_fraud": bool(prediction == 1),
            "fraud_probability": round(float(probabilities[1]), 4),
            "legitimate_probability": round(float(probabilities[0]), 4),
            "transaction_timestamp": transaction.Timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "API is online", 
        "model_loaded": model_pipeline is not None,
        "message": "Visit http://127.0.0.1:8000/docs to test the API endpoints."
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)