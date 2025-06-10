import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from rhd_risk_prediction.data.make_dataset import generate_rhd_data  # Absolute

def train_model(X: pd.DataFrame, y: pd.Series):
    """Train and save a Random Forest classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model, path="models/rf_model.pkl"):
    """Save trained model to disk"""
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    # Load and prepare data
    data = generate_rhd_data(5000)
    X = pd.get_dummies(data[["age", "prev_pregnancies", "blood_type", "rh_factor"]])
    y = data["risk_score"]
    
    # Train and save
    model = train_model(X, y)
    save_model(model)

def main():
    """Main executable function"""
    data = generate_rhd_data(1000)
    X = pd.get_dummies(data[["age", "prev_pregnancies", "blood_type"]])
    y = data["risk_score"]
    
    model = train_model(X, y)
    save_model(model)
    print("✅ Model training completed successfully")

if __name__ == "__main__":
    main()