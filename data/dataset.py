import pandas as pd
import numpy as np
from faker import Faker
import matplotlib.pyplot as plt
import os

# Configure output directory
output_folder = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "PROJECTS", "Rhesus Disease (RHD) Risk Prediction")
os.makedirs(output_folder, exist_ok=True)

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

def save_data(df, filepath):
    """Save dataframe with validation checks"""
    try:
        df.to_csv(filepath, index=False)
        print(f"✅ Successfully saved to {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def preprocess_data(df):
    """Preprocess the raw dataset"""
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["blood_type", "rh_factor", "father_rh", "antibody_test"])
    
    # Normalize numeric features
    df["age"] = (df["age"] - df["age"].mean()) / df["age"].std()
    df["prev_pregnancies"] = (df["prev_pregnancies"] - df["prev_pregnancies"].min()) / \
                           (df["prev_pregnancies"].max() - df["prev_pregnancies"].min())
    return df

def plot_distributions(df):
    """Generate comprehensive visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rh Factor Distribution
    df["rh_factor"].value_counts().plot(kind='bar', ax=axes[0,0], title='Rh Factor')
    
    # Age Distribution
    df["age"].plot(kind='hist', ax=axes[0,1], title='Age Distribution')
    
    # Risk Score Distribution
    df["risk_score"].value_counts().plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%', title='Risk Score')
    
    # Pregnancies Distribution
    df["prev_pregnancies"].value_counts().sort_index().plot(kind='bar', ax=axes[1,1], title='Previous Pregnancies')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'data_distributions.png'))
    plt.show()

if __name__ == "__main__":
    # Configuration
    NUM_PATIENTS = 5000
    RAW_DATA_PATH = os.path.join(output_folder, "rhd_raw_data.csv")
    PROCESSED_DATA_PATH = os.path.join(output_folder, "rhd_processed_data.csv")

    print("🚀 Generating synthetic Rhesus disease data...")
    
    # Data Generation
    rhd_data = generate_rhd_data(num_patients=NUM_PATIENTS)
    
    # Save Raw Data
    if save_data(rhd_data, RAW_DATA_PATH):
        print("📊 Basic Statistics:")
        print(rhd_data.describe())
        
        # Preprocessing
        processed_data = preprocess_data(rhd_data)
        if save_data(processed_data, PROCESSED_DATA_PATH):
            print("🔍 Processed Data Sample:")
            print(processed_data.head())
            
            # Visualization
            plot_distributions(rhd_data)