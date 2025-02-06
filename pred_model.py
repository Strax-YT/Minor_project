import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import optuna
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class FinalCKDModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer()
        self.selected_features = [
            "SerumCreatinine", "GFR", "ProteinInUrine", "BUNLevels",
            "SerumElectrolytesSodium", "HemoglobinLevels"
        ]
        
    def create_medical_features(self, X):
        """Create clinically relevant feature combinations"""
        # Key medical ratios
        X['BUN_Creatinine_Ratio'] = X['BUNLevels'] / (X['SerumCreatinine'] + 1e-6)
        X['GFR_Creatinine_Ratio'] = X['GFR'] / (X['SerumCreatinine'] + 1e-6)
        X['BUN_GFR_Ratio'] = X['BUNLevels'] / (X['GFR'] + 1e-6)
        X['Hemoglobin_GFR_Ratio'] = X['HemoglobinLevels'] / (X['GFR'] + 1e-6)
        
        # Clinical risk scores
        X['Kidney_Risk_Score'] = (
            X['BUNLevels'] * (1 / (X['GFR'] + 1e-6)) * 
            X['SerumCreatinine'] * (1 / (X['HemoglobinLevels'] + 1e-6))
        )
        
        # Normalized versions of important features
        for feature in ['BUNLevels', 'GFR', 'SerumCreatinine', 'HemoglobinLevels']:
            X[f'{feature}_Norm'] = self.power_transformer.fit_transform(X[[feature]])
        
        return X
        
    def preprocess_data(self, filepath):
        """Enhanced preprocessing with medical feature engineering"""
        df = pd.read_csv(filepath)
        df_clean = df.drop(columns=["PatientID", "DoctorInCharge"], errors="ignore")
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = pd.Categorical(df_clean[col]).codes
        
        X = df_clean[self.selected_features].copy()
        y = df_clean["Diagnosis"]
        
        # Handle missing values using domain-specific logic
        X = X.fillna(X.median())
        
        # Create medical features
        X = self.create_medical_features(X)
        
        return X, y
    
    def optimize_hyperparameters(self, X, y):
        """Refined hyperparameter optimization"""
        def objective(trial):
            params = {
                'lgbm': {
                    'n_estimators': trial.suggest_int('lgbm_n_estimators', 500, 3000),
                    'max_depth': trial.suggest_int('lgbm_max_depth', 5, 25),
                    'learning_rate': trial.suggest_loguniform('lgbm_lr', 1e-4, 0.1),
                    'num_leaves': trial.suggest_int('lgbm_leaves', 20, 300),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5, 20),
                },
                'gb': {
                    'n_estimators': trial.suggest_int('gb_n_estimators', 200, 2000),
                    'max_depth': trial.suggest_int('gb_max_depth', 3, 15),
                    'learning_rate': trial.suggest_loguniform('gb_lr', 1e-4, 0.1),
                    'subsample': trial.suggest_uniform('gb_subsample', 0.6, 1.0),
                },
                'rf': {
                    'n_estimators': trial.suggest_int('rf_n_estimators', 200, 2000),
                    'max_depth': trial.suggest_int('rf_max_depth', 5, 25),
                    'min_samples_split': trial.suggest_int('rf_min_split', 2, 20),
                }
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores_auc = []
            scores_precision = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Enhanced sampling with careful ratio
                smotetomek = SMOTETomek(sampling_strategy={0: int(len(y_train[y_train==1])*0.4)}, 
                                      random_state=42)
                X_train_bal, y_train_bal = smotetomek.fit_resample(X_train, y_train)
                
                # Train models
                models = {
                    'lgbm': LGBMClassifier(**params['lgbm'], random_state=42),
                    'gb': GradientBoostingClassifier(**params['gb'], random_state=42),
                    'rf': RandomForestClassifier(**params['rf'], random_state=42)
                }
                
                predictions = []
                for model in models.values():
                    model.fit(X_train_bal, y_train_bal)
                    pred = model.predict_proba(X_val)[:, 1]
                    predictions.append(pred)
                
                ensemble_pred = np.mean(predictions, axis=0)
                auc_score = roc_auc_score(y_val, ensemble_pred)
                precision = precision_recall_curve(y_val, ensemble_pred)[0].mean()
                
                scores_auc.append(auc_score)
                scores_precision.append(precision)
            
            return np.mean(scores_auc) * 0.7 + np.mean(scores_precision) * 0.3
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200)
        return study.best_params
    
    def train_ensemble(self, X, y, params):
        """Enhanced ensemble with calibration and precision focus"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns
        )
        
        # Careful balancing
        smotetomek = SMOTETomek(
            sampling_strategy={0: int(len(y_train[y_train==1])*0.4)},
            random_state=42
        )
        X_train_balanced, y_train_balanced = smotetomek.fit_resample(X_train_scaled, y_train)
        
        # Initialize base models
        base_models = {
            'lgbm': LGBMClassifier(
                **{k.replace('lgbm_', ''): v for k, v in params.items() if k.startswith('lgbm_')},
                random_state=42
            ),
            'gb': GradientBoostingClassifier(
                **{k.replace('gb_', ''): v for k, v in params.items() if k.startswith('gb_')},
                random_state=42
            ),
            'rf': RandomForestClassifier(
                **{k.replace('rf_', ''): v for k, v in params.items() if k.startswith('rf_')},
                random_state=42
            )
        }
        
        # Train calibrated models
        calibrated_models = {}
        for name, model in base_models.items():
            calibrated = CalibratedClassifierCV(model, cv=5, method='sigmoid')
            calibrated.fit(X_train_balanced, y_train_balanced)
            calibrated_models[name] = calibrated
        
        # Create voting ensemble
        voting = VotingClassifier(
            estimators=[(name, model) for name, model in calibrated_models.items()],
            voting='soft'
        )
        voting.fit(X_train_balanced, y_train_balanced)
        
        # Get predictions
        y_pred_proba = voting.predict_proba(X_test_scaled)[:, 1]
        
        # Find optimal threshold focusing on precision
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Make final predictions
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Print results
        print("\nFinal Enhanced Model Results:")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Calculate and sort feature importances
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': base_models['lgbm'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Feature Importances:")
        print(feature_imp.head(15))
        
        return calibrated_models, voting, optimal_threshold

def main():
    model = FinalCKDModel()
    X, y = model.preprocess_data("D:\Minor_project\Chronic_Kidney_Dsease_data.csv")
    best_params = model.optimize_hyperparameters(X, y)
    model.train_ensemble(X, y, best_params)

if __name__ == "__main__":
    main()