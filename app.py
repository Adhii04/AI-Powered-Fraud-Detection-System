import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from flask import Flask, request, jsonify
from waitress import serve

# --- Flask App and Model Loading ---
app = Flask(__name__)

# In your app.py file
@app.route('/')
def home():
    return "<h1>Fraud Detection API is running!</h1><p>Send a POST request to /predict to get a prediction.</p>"

# Load the trained model and preprocessor on app startup
try:
    # Use the appropriate model and preprocessor for your application
    model = joblib.load('fraud_model_smote.pkl')
    preprocessor = joblib.load('preprocessor_smote.pkl')
    print("Model and preprocessor loaded successfully for Flask app.")
except FileNotFoundError:
    print("Error: Model files not found. Ensure 'fraud_model_smote.pkl' and 'preprocessor_smote.pkl' are in the same directory.")
    exit()

# --- Feature Engineering Helper Function ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth.
    """
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a JSON payload of transaction data and returns a fraud prediction.
    """
    try:
        json_data = request.get_json(force=True)
        
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        result = {
            'is_fraud': int(prediction[0]),
            'probability_not_fraud': float(prediction_proba[0][0]),
            'probability_fraud': float(prediction_proba[0][1])
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Gunicorn Command for Production Deployment ---
if __name__ == '__main__':
    # Use Waitress for production on Windows
    serve(app, host='0.0.0.0', port=8000)
    