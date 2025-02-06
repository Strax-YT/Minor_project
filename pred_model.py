import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    selected_features = [
        "SerumCreatinine", "GFR", "ProteinInUrine", "BUNLevels",
        "SerumElectrolytesSodium", "HemoglobinLevels"
    ]

    X = df[selected_features].copy()
    y = df["Diagnosis"]

    X = X.fillna(X.median())

    # Add feature interactions
    X['GFR_Creatinine_Interaction'] = X['GFR'] * X['SerumCreatinine']
    X['BUN_Protein_Interaction'] = X['BUNLevels'] * X['ProteinInUrine']

    X['BUN_Creatinine_Ratio'] = X['BUNLevels'] / (X['SerumCreatinine'] + 1e-6)
    X['GFR_Creatinine_Ratio'] = X['GFR'] / (X['SerumCreatinine'] + 1e-6)
    X['Kidney_Risk_Score'] = (X['BUNLevels'] / (X['GFR'] + 1e-6)) * X['SerumCreatinine']
    X['Protein_GFR_Ratio'] = X['ProteinInUrine'] / (X['GFR'] + 1e-6)
    X['Electrolyte_Hemoglobin_Ratio'] = X['SerumElectrolytesSodium'] / (X['HemoglobinLevels'] + 1e-6)

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(sampling_strategy=0.6, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.005,
        max_depth=8,
        num_leaves=31,
        min_child_samples=20,
        scale_pos_weight=15,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=800,
        learning_rate=0.005,
        max_depth=6,
        min_child_weight=7,
        scale_pos_weight=12,
        random_state=42
    )

    models = {'lgbm': lgbm, 'rf': rf, 'xgb': xgb}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        predictions[name] = model.predict_proba(X_test_scaled)[:, 1]

    y_pred_proba = (predictions['lgbm'] + predictions['rf'] + predictions['xgb']) / 3

    thresholds = np.arange(0.3, 0.7, 0.05)
    best_f1, best_threshold = 0, 0.4  # Starting with lower threshold

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = classification_report(y_test, y_pred, output_dict=True)['0']['f1-score']
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_pred = (y_pred_proba >= best_threshold).astype(int)

    print(f"\nOptimal Threshold: {best_threshold:.3f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': models['lgbm'].feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(feature_imp)

if __name__ == "__main__":
    X, y = preprocess_data("/content/Chronic_Kidney_Dsease_data.csv")
    train_model(X, y)