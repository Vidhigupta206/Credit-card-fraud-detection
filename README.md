# 🛡️ Real-Time Credit Card Fraud Detection System

End-to-end ML project detecting fraudulent transactions with **95% recall**.

## Features
- Trained on Kaggle Credit Card Fraud Dataset (284,807 transactions)
- XGBoost model with SMOTE, threshold tuning for high fraud recall
- Real-time prediction via FastAPI
- Beautiful, fully customizable HTML dashboard
- Responsible AI: PCA privacy, bias mitigation

## How to Run Locally

1. Clone repo:
   ```bash
   git clone git clone https://github.com/Vidhigupta206/Credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ## Screenshots
   
2. Set up the Environment
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt

3. Run the server
   uvicorn src.app:app --reload

4. Open dashboard:
   http://127.0.0.1:8000

### Legitimate Transaction
![Legitimate](<img width="1886" height="972" alt="Screenshot 2025-12-18 193342" src="https://github.com/user-attachments/assets/83b1b4f6-f11e-4bb7-af4d-fa4b48cfa9e7" />)

### Fraud Alert
![Fraud Alert](<img width="1908" height="904" alt="Screenshot 2025-12-18 185703" src="https://github.com/user-attachments/assets/28242099-04ba-4cdb-82d8-316d2895364c" />)
   
