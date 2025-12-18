import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, recall_score
import numpy as np
import joblib
import os

from data_preprocessing import load_data, preprocess

# Load and preprocess
df = load_data('data/creditcard_fresh.csv')  # Use your fresh copy
processed_df = preprocess(df)

# Features and target
X = processed_df.drop('Class', axis=1)
y = processed_df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train samples: {len(X_train)}, Fraud %: {y_train.mean()*100:.3f}")
print(f"Test samples: {len(X_test)}, Fraud %: {y_test.mean()*100:.3f}")

# SMOTE on train only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {len(X_train_res)} samples (50% fraud)")

# Models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
}

best_recall = 0
best_model = None
best_name = ""

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train_res, y_train_res)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Find threshold for ~95% recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    target = 0.95
    idx = np.where(recall >= target)[0]
    if len(idx) > 0:
        optimal_idx = idx[-1]
        threshold = thresholds[optimal_idx]
        y_pred = (y_prob >= threshold).astype(int)
        current_recall = recall[optimal_idx]
    else:
        y_pred = model.predict(X_test)
        threshold = 0.5
        current_recall = recall_score(y_test, y_pred)
    
    print(f"Threshold: {threshold:.3f} | Recall on fraud: {current_recall:.3f}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    if current_recall > best_recall:
        best_recall = current_recall
        best_model = model
        best_name = name

# Save best model and scaler
os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/best_fraud_model.pkl')
print(f"\n*** BEST MODEL: {best_name} with Recall {best_recall:.3f} on fraud class ***")
print("Model saved to ../models/best_fraud_model.pkl")