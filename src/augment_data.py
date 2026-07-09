import pandas as pd
import numpy as np
import os

def augment_dataset(input_path: str, output_path: str, target_rows: int = 50000):
    print(f"Loading base dataset from {input_path}...")
    df_base = pd.read_csv(input_path)
    
    current_rows = len(df_base)
    if current_rows == 0:
        print("Base dataset is empty.")
        return
        
    print(f"Base dataset has {current_rows} rows. Augmenting to {target_rows} rows...")
    
    # Calculate how many times we need to duplicate the data
    multiples = (target_rows // current_rows) + 1
    
    # Duplicate the dataframe
    df_augmented = pd.concat([df_base] * multiples, ignore_index=True)
    
    # Trim exactly to the target row count
    df_augmented = df_augmented.iloc[:target_rows].copy()
    
    # Add statistical noise to make the data realistic and prevent the model from memorizing exact rows
    print("Applying statistical noise to numerical features...")
    
    # 1. Perturb the Transaction Amount (± 5% noise)
    if 'Amount' in df_augmented.columns:
        noise = np.random.uniform(0.95, 1.05, size=target_rows)
        df_augmented['Amount'] = (df_augmented['Amount'] * noise).round(2)
        
    # 2. Perturb Age slightly (± 1-2 years), ensuring no negative ages
    if 'Age' in df_augmented.columns:
        age_noise = np.random.randint(-2, 3, size=target_rows)
        df_augmented['Age'] = (df_augmented['Age'] + age_noise).clip(lower=18)
        
    # 3. Perturb AccountBalance (± 10% noise)
    if 'AccountBalance' in df_augmented.columns:
        balance_noise = np.random.uniform(0.90, 1.10, size=target_rows)
        df_augmented['AccountBalance'] = (df_augmented['AccountBalance'] * balance_noise).round(2)
        
    # We leave Categorical data (MerchantID, Category) and Fraud labels exactly as they are.
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_augmented.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated augmented dataset with {len(df_augmented)} rows!")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(BASE_DIR, "data", "processed", "merged_data.csv")
    output_csv = os.path.join(BASE_DIR, "data", "processed", "augmented_data.csv")
    
    augment_dataset(input_csv, output_csv, target_rows=50000)
