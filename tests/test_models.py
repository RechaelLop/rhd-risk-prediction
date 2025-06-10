import pytest
import pandas as pd
import numpy as np
from rhd_risk_prediction.models.train_model import train_model

@pytest.fixture
def sample_data():
    """Generate test data for model training"""
    return pd.DataFrame({
        'age': np.random.normal(30, 5, 100),
        'prev_pregnancies': np.random.randint(0, 5, 100)
    }), np.random.randint(0, 2, 100)

def test_model_training(sample_data):
    """Test that model training works with valid data"""
    X, y = sample_data
    model = train_model(X, y)
    
    # Basic model validation
    assert hasattr(model, 'predict')
    assert hasattr(model, 'fit')
    assert len(model.predict(X)) == len(y)