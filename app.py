# app.py
import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_model
from src.preprocess import preprocess_data

def main():
    st.title('Chronic Kidney Disease Prediction')
    st.write('Enter patient biomarkers for CKD risk assessment')

    models, scaler = load_model()
    if not models:
        st.error('Error loading model. Please ensure model file exists.')
        return

    with st.form('prediction_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            serum_creatinine = st.number_input('Serum Creatinine', value=1.0, min_value=0.0)
            gfr = st.number_input('GFR', value=90.0, min_value=0.0)
            protein_urine = st.number_input('Protein in Urine', value=0.0, min_value=0.0)
            
        with col2:
            bun_levels = st.number_input('BUN Levels', value=15.0, min_value=0.0)
            sodium = st.number_input('Serum Electrolytes Sodium', value=140.0, min_value=0.0)
            hemoglobin = st.number_input('Hemoglobin Levels', value=14.0, min_value=0.0)

        submit = st.form_submit_button('Predict')

        if submit:
            input_data = pd.DataFrame({
                'SerumCreatinine': [serum_creatinine],
                'GFR': [gfr],
                'ProteinInUrine': [protein_urine],
                'BUNLevels': [bun_levels],
                'SerumElectrolytesSodium': [sodium],
                'HemoglobinLevels': [hemoglobin]
            })

            # Process input
            processed_data = preprocess_data(input_data)
            scaled_data = scaler.transform(processed_data)

            # Get predictions
            predictions = []
            for model in models.values():
                pred = model.predict_proba(scaled_data)[:, 1]
                predictions.append(pred)

            final_prob = np.mean(predictions)
            prediction = 1 if final_prob >= 0.4 else 0

            st.subheader('Prediction Results')
            risk_percentage = final_prob * 100

            if prediction == 1:
                st.error(f'High Risk of CKD (Risk Score: {risk_percentage:.1f}%)')
            else:
                st.success(f'Low Risk of CKD (Risk Score: {risk_percentage:.1f}%)')

if __name__ == '__main__':
    main()