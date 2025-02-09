import pandas as pd
import numpy as np

def preprocess_data(data):
    """
    Preprocess data for CKD prediction.
    
    Args:
        data: Either a filepath (str) or pandas DataFrame
        
    Returns:
        tuple: (X, y) if filepath with labels is provided, or just X if DataFrame of features is provided
    """
    # Handle different input types
    if isinstance(data, str):
        # If input is a filepath
        df = pd.read_csv(data)
        include_labels = True
    elif isinstance(data, pd.DataFrame):
        # If input is already a DataFrame
        df = data
        include_labels = 'Diagnosis' in df.columns
    else:
        raise TypeError("Input must be either a filepath string or pandas DataFrame")

    selected_features = [
        "SerumCreatinine", "GFR", "ProteinInUrine", "BUNLevels",
        "SerumElectrolytesSodium", "HemoglobinLevels"
    ]
    
    # Ensure all required features are present
    missing_features = [feat for feat in selected_features if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    X = df[selected_features].copy()
    
    # Only get labels if they exist in the input data
    y = df["Diagnosis"] if include_labels else None
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Feature engineering
    X['GFR_Creatinine_Interaction'] = X['GFR'] * X['SerumCreatinine']
    X['BUN_Protein_Interaction'] = X['BUNLevels'] * X['ProteinInUrine']
    X['BUN_Creatinine_Ratio'] = X['BUNLevels'] / (X['SerumCreatinine'] + 1e-6)
    X['GFR_Creatinine_Ratio'] = X['GFR'] / (X['SerumCreatinine'] + 1e-6)
    X['Kidney_Risk_Score'] = (X['BUNLevels'] / (X['GFR'] + 1e-6)) * X['SerumCreatinine']
    X['Protein_GFR_Ratio'] = X['ProteinInUrine'] / (X['GFR'] + 1e-6)
    X['Electrolyte_Hemoglobin_Ratio'] = X['SerumElectrolytesSodium'] / (X['HemoglobinLevels'] + 1e-6)
    
    return (X, y) if include_labels else X