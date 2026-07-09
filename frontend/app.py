import streamlit as st
import requests
from datetime import datetime

import os

# Configure page settings
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Define API URL (defaults to localhost, but allows Docker to override it to point to the 'api' container)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.title("🛡️ Fraud Detection Dashboard")
st.markdown("Enter transaction details below to evaluate the probability of fraud using the XGBoost model.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.00, step=10.0)
    category = st.selectbox("Merchant Category", ["shopping_net", "grocery_pos", "entertainment", "gas_transport", "electronics"])
    location = st.text_input("Transaction Location", value="Miami, FL")
    
with col2:
    st.subheader("Customer Details")
    age = st.number_input("Customer Age", min_value=18, max_value=120, value=35)
    account_balance = st.number_input("Account Balance ($)", min_value=0.0, value=5000.00, step=100.0)
    anomaly_score = st.slider("Upstream Anomaly Score", min_value=0.0, max_value=1.0, value=0.10)
    suspicious_flag = st.selectbox("Pre-flagged by Rule Engine?", [0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp = st.text_input("Timestamp", value=current_time)

st.divider()

if st.button("Evaluate Transaction", type="primary", use_container_width=True):
    payload = {
        "Amount": amount,
        "Category": category,
        "AnomalyScore": anomaly_score,
        "Age": age,
        "AccountBalance": account_balance,
        "SuspiciousFlag": suspicious_flag,
        "Location": location,
        "Timestamp": timestamp
    }
    
    with st.spinner("Analyzing transaction patterns..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                is_fraud = result["is_fraud"]
                fraud_prob = result["fraud_probability"] * 100
                legit_prob = result["legitimate_probability"] * 100
                
                st.subheader("Analysis Results")
                
                if is_fraud:
                    st.error(f"🚨 FRAUD DETECTED: {fraud_prob:.1f}% Probability")
                else:
                    st.success(f"✅ TRANSACTION SECURE: {legit_prob:.1f}% Legitimate")
                
                # Display metrics side by side
                m1, m2 = st.columns(2)
                m1.metric("Fraud Probability", f"{fraud_prob:.2f}%")
                m2.metric("Legitimate Probability", f"{legit_prob:.2f}%")
                
                st.progress(fraud_prob / 100.0)
                
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the Backend API. Please ensure FastAPI is running on port 8000.")
