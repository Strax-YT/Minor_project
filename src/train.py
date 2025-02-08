import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from preprocess import preprocess_data
from utils import save_model
import warnings
warnings.filterwarnings('ignore')

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(sampling_strategy=0.6, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    models = {
        'lgbm': LGBMClassifier(
            n_estimators=1000, learning_rate=0.005, max_depth=8,
            num_leaves=31, min_child_samples=20, scale_pos_weight=15,
            random_state=42
        ),
        'rf': RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=4,
            class_weight='balanced_subsample', random_state=42
        ),
        'xgb': XGBClassifier(
            n_estimators=800, learning_rate=0.005, max_depth=6,
            min_child_weight=7, scale_pos_weight=12, random_state=42
        )
    }
    
    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
    
    save_model(models, scaler)
    return models, scaler

if __name__ == "__main__":
    X, y = preprocess_data("data/Chronic_Kidney_Dsease_data.csv")
    train_model(X, y)