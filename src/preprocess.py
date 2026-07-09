import pandas as pd
import os
from pathlib import Path

def merge_and_preprocess_data(base_path: str, output_path: str):
    """
    Reads all relational CSVs from the dataset directory, merges them into a single 
    tabular dataframe, performs basic feature engineering, and saves the result.
    """
    print("Starting data merge process...")
    data_dir = Path(base_path)
    
    # 1. Load Transaction Data (The Core)
    print("Loading Transaction Data...")
    df_trans = pd.read_csv(data_dir / "Transaction Data/transaction_records.csv")
    df_meta = pd.read_csv(data_dir / "Transaction Data/transaction_metadata.csv")
    df = pd.merge(df_trans, df_meta, on="TransactionID", how="left")
    
    # 2. Load and Merge Target Labels (Fraud Indicator)
    print("Loading Fraud Labels...")
    df_fraud = pd.read_csv(data_dir / "Fraudulent Patterns/fraud_indicators.csv")
    df = pd.merge(df, df_fraud, on="TransactionID", how="left")
    
    # 3. Load and Merge Category & Amount Data
    print("Loading Category and Amount Data...")
    df_cat = pd.read_csv(data_dir / "Merchant Information/transaction_category_labels.csv")
    df_amt = pd.read_csv(data_dir / "Transaction Amounts/amount_data.csv")
    df_anomaly = pd.read_csv(data_dir / "Transaction Amounts/anomaly_scores.csv")
    
    df = pd.merge(df, df_cat, on="TransactionID", how="left")
    df = pd.merge(df, df_amt, on="TransactionID", how="left")
    df = pd.merge(df, df_anomaly, on="TransactionID", how="left")
    
    # 4. Load and Merge Customer Data
    print("Loading Customer Data...")
    df_cust = pd.read_csv(data_dir / "Customer Profiles/customer_data.csv")
    df_acct = pd.read_csv(data_dir / "Customer Profiles/account_activity.csv")
    df_suspicious = pd.read_csv(data_dir / "Fraudulent Patterns/suspicious_activity.csv")
    
    df_customer_combined = pd.merge(df_cust, df_acct, on="CustomerID", how="left")
    df_customer_combined = pd.merge(df_customer_combined, df_suspicious, on="CustomerID", how="left")
    
    df = pd.merge(df, df_customer_combined, on="CustomerID", how="left")
    
    # 5. Load and Merge Merchant Data
    print("Loading Merchant Data...")
    df_merchant = pd.read_csv(data_dir / "Merchant Information/merchant_data.csv")
    df = pd.merge(df, df_merchant, on="MerchantID", how="left")
    
    # --- Feature Engineering & Cleaning ---
    print("Applying Feature Engineering...")
    
    # Parse Timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    
    # Handle potentially duplicate Amount columns 
    if 'TransactionAmount' in df.columns:
        df.drop(columns=['TransactionAmount'], inplace=True)
        
    # Fill missing values
    df['Category'] = df['Category'].fillna('Unknown')
    df['SuspiciousFlag'] = df['SuspiciousFlag'].fillna(0) # 0 for not flagged
    df['FraudIndicator'] = df['FraudIndicator'].fillna(0) # Assuming missing means not fraud
    
    # Drop columns that are pure identifiers and not useful for ML modeling directly
    columns_to_drop = ['Name', 'Address', 'Timestamp']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    # Save the merged dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Successfully merged and saved {len(df)} rows to {output_path}")
    print(f"Columns in final dataset: {list(df.columns)}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(BASE_DIR, "fraud-dataset", "Data")
    output_csv = os.path.join(BASE_DIR, "data", "processed", "merged_data.csv")
    
    merge_and_preprocess_data(dataset_path, output_csv)
