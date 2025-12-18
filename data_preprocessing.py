import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path='data/creditcard_fresh.csv'):
    df = pd.read_csv(path)
    print("Data loaded successfully!")
    print(df['Class'].value_counts(normalize=True))
    return df

def clean_and_scale_amount(df):
    # Scale Amount
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df = df.drop('Amount', axis=1)
    
    # Save scaler for deployment
    os.makedirs('../models', exist_ok=True)
    joblib.dump(scaler, '../models/amount_scaler.pkl')
    print("Amount scaled and scaler saved.")
    return df

def engineer_features(df):
    # Extract hour from Time (fraud often happens at odd hours)
    df['Hour'] = (df['Time'] % (24 * 3600)) // 3600
    
    # Flag unusually high amounts (after scaling)
    df['High_Amount'] = (df['Amount_scaled'] > 3).astype(int)
    
    print("Feature engineering complete: Added 'Hour' and 'High_Amount'")
    return df

def preprocess(df):
    df = clean_and_scale_amount(df)
    df = engineer_features(df)
    return df

# Optional: Test it when running this file directly
if __name__ == "__main__":
    df = load_data()
    processed_df = preprocess(df)
    print(processed_df.head())
    print("\nNew columns:", [col for col in processed_df.columns if col not in ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class']])