import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime

# Configure page settings
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Load the trained model pipeline 
# On Hugging Face, the files are in the same working directory
@st.cache_resource
def load_model():
    try:
        # Assumes you uploaded the models folder to HF
        model_path = os.path.join("models", "fraud_model_smote.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_pipeline = load_model()

st.title("🛡️ Fraud Detection Dashboard (Hugging Face Edition)")
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
    if model_pipeline is None:
        st.error("Model failed to load. Please ensure 'models/fraud_model_smote.pkl' is uploaded to the Space.")
    else:
        with st.spinner("Analyzing transaction patterns with XGBoost..."):
            # Process inputs exactly like our FastAPI backend did
            dt = pd.to_datetime(timestamp)
            hour = dt.hour
            day_of_week = dt.dayofweek
            
            input_df = pd.DataFrame([{
                'Amount': amount,
                'Category': category,
                'AnomalyScore': anomaly_score,
                'Age': age,
                'AccountBalance': account_balance,
                'SuspiciousFlag': suspicious_flag,
                'Location': location,
                'Hour': hour,
                'DayOfWeek': day_of_week
            }])
            
            # Predict
            prediction = model_pipeline.predict(input_df)[0]
            probabilities = model_pipeline.predict_proba(input_df)[0]
            
            is_fraud = bool(prediction == 1)
            fraud_prob = float(probabilities[1]) * 100
            legit_prob = float(probabilities[0]) * 100
            
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
