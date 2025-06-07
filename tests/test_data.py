import pandas as pd
from src.data.make_dataset import generate_rhd_data

def test_data_generation():
    """Test that data generation produces valid output"""
    data = generate_rhd_data(num_patients=100)
    
    # Check basic structure
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert "risk_score" in data.columns
    
    # Check value ranges
    assert data["age"].between(18, 45).all()
    assert set(data["blood_type"]) == {"A", "B", "AB", "O"}
    assert set(data["rh_factor"]) == {"+", "-"}