import joblib

def save_model(models, scaler, filepath='models/ckd_prediction_model.pkl'):
    model_data = {
        'models': models,
        'scaler': scaler
    }
    joblib.dump(model_data, filepath)

def load_model(filepath='models/ckd_prediction_model.pkl'):
    try:
        model_data = joblib.load(filepath)
        return model_data['models'], model_data['scaler']
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None