import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_models():
    print("--- Starting Phase 2: Model Training ---")
    start_total = time.time()
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "processed", "augmented_data.csv")
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape}")

    # 2. Define Features and Target
    target_col = 'FraudIndicator'
    
    # Drop identifiers and columns that shouldn't be used for prediction
    cols_to_drop = [target_col, 'TransactionID', 'CustomerID', 'MerchantID', 'MerchantName', 'LastLogin']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df[target_col].astype(int)

    print("Class distribution before sampling:")
    print(y.value_counts())

    # Define categorical and numerical columns for the preprocessor
    categorical_cols = ['Category', 'Location']
    numerical_cols = ['Amount', 'AnomalyScore', 'Age', 'AccountBalance', 'Hour', 'DayOfWeek']
    
    # Ensure columns exist before adding them to transformer
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    numerical_cols = [col for col in numerical_cols if col in X.columns]

    # 3. Create Preprocessor (Transforms raw data into ML-ready numbers)
    # This prevents data leakage and makes the FastAPI deployment much easier!
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough' # Leave SuspiciousFlag as is
    )

    # 4. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # We will train the pipeline using SMOTE
    samplers = {
        'smote': SMOTE(random_state=42)
    }

    # Common XGBoost parameters
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'n_jobs': -1 # Use all CPU cores for speed
    }

    # 5. Train and Save Models
    for name, sampler in samplers.items():
        print(f"\nTraining pipeline with {name.upper()}...")
        start_time = time.time()
        
        # We use ImbPipeline so SMOTE is only applied to the training data, NOT the test data!
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('sampler', sampler),
            ('classifier', XGBClassifier(**xgb_params))
        ])

        # Train the pipeline
        pipeline.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f} seconds.")

        # Evaluate
        y_pred = pipeline.predict(X_test)
        print(f"Classification Report for {name.upper()}:")
        print(classification_report(y_test, y_pred))
        
        # Save the full pipeline (includes preprocessor + model)
        model_path = os.path.join(model_dir, f'fraud_model_{name}.pkl')
        joblib.dump(pipeline, model_path)
        print(f"Saved complete pipeline to {model_path}")

    total_elapsed = time.time() - start_total
    print(f"\n--- Phase 2 Complete! Total time: {total_elapsed:.2f} seconds ---")

if __name__ == "__main__":
    train_models()
