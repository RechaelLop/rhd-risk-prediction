import pandas as pd
import numpy as np
from faker import Faker
import matplotlib.pyplot as plt  # For basic visualization
import os

output_folder = r"C:\Users\lopes\OneDrive\Desktop\PROJECTS\Rhesus Disease (RHD) Risk Prediction"

def generate_rhd_data(num_patients=1000):
    """Generate synthetic Rhesus disease dataset"""
    fake = Faker()
    np.random.seed(42)  # For reproducibility
    
    data = {
        "patient_id": [fake.unique.uuid4() for _ in range(num_patients)],
        "age": np.random.randint(18, 45, num_patients),
        "blood_type": np.random.choice(["A", "B", "AB", "O"], num_patients),
        "rh_factor": np.random.choice(["+", "-"], num_patients, p=[0.85, 0.15]),
        "prev_pregnancies": np.random.randint(0, 5, num_patients),
        "antibody_test": np.random.choice(["Negative", "Positive"], num_patients, p=[0.7, 0.3]),
        "father_rh": np.random.choice(["+", "-"], num_patients, p=[0.85, 0.15]),
    }
    
    df = pd.DataFrame(data)
    
    # Medical logic for risk score
    df["risk_score"] = np.where(
        (df["rh_factor"] == "-") & 
        (df["father_rh"] == "+") & 
        (df["antibody_test"] == "Positive"), 
        1, 0
    )
    
    return df

def save_and_analyze(df, filename="synthetic_rhd_data.csv"):
    """Save data and show basic stats"""
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved to {filename}")
    
    # Show basic info
    print("\n📊 Dataset Summary:")
    print(df.info())
    
    # Show risk score distribution
    print("\n📈 Risk Score Distribution:")
    print(df["risk_score"].value_counts())
    
    # Simple visualization
    df["rh_factor"].value_counts().plot(kind="bar", title="Rh Factor Distribution")
    plt.show()

if __name__ == "__main__":
    print("Generating synthetic Rhesus disease data...")
    # To generate more records
    rhd_data = generate_rhd_data(num_patients=5000)

    # To change file output location
    save_and_analyze(rhd_data, filename="C:/Users/lopes/OneDrive/Desktop/PROJECTS/Rhesus Disease (RHD) Risk Prediction/rhd_dataset.csv")

    # Pre-processing the data
    # Add this to your save_and_analyze() function
    def preprocess_data(df):
        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=["blood_type", "rh_factor", "father_rh", "antibody_test"])
        
        # Normalize age and pregnancies
        df["age"] = (df["age"] - df["age"].mean()) / df["age"].std()
        df["prev_pregnancies"] = (df["prev_pregnancies"] - df["prev_pregnancies"].min()) / \
                                (df["prev_pregnancies"].max() - df["prev_pregnancies"].min())
        return df

    processed_data = preprocess_data(rhd_data)
    processed_data.to_csv(os.path.join(output_folder, "processed_rhd_data.csv"), index=False)