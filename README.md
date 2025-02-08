# Chronic Kidney Disease Prediction System
A machine learning-based web application that predicts the risk of Chronic Kidney Disease (CKD) using patient biomarkers. The system employs an ensemble of advanced machine learning models including LightGBM, Random Forest, and XGBoost to provide accurate risk assessments.
Features

Real-time CKD risk prediction
User-friendly web interface built with Streamlit
Ensemble learning approach using multiple ML models
Advanced feature engineering for improved accuracy
Handles class imbalance using SMOTE
Comprehensive data preprocessing pipeline

# Project Structure
Copyproject/
├── data/
│   └── Chronic_Kidney_Dsease_data.csv
├── models/
│   └── ckd_prediction_model.pkl
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── preprocess.py
│   └── utils.py
├── app.py
└── requirements.txt
Installation

Clone the repository:

bashCopygit clone [https://github.com/yourusername/ckd-prediction.git](https://github.com/Strax-YT/Minor_project)
cd ckd-prediction

Create a virtual environment (optional but recommended):

bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required packages:

bashCopypip install -r requirements.txt
Usage

Train the model (optional - pre-trained model included):

bashCopypython -m src.train

Run the Streamlit app:

bashCopystreamlit run app.py

Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

Input Features
The system requires the following biomarkers:

Serum Creatinine (mg/dL)
GFR (mL/min/1.73m²)
Protein in Urine (g/24h)
BUN Levels (mg/dL)
Serum Electrolytes Sodium (mEq/L)
Hemoglobin Levels (g/dL)

Model Details
The system uses an ensemble of three models:

LightGBM Classifier

Optimized for handling imbalanced medical data
Uses gradient boosting with tree-based learning


Random Forest Classifier

Provides robust predictions through multiple decision trees
Balanced subsample weighting for handling class imbalance


XGBoost Classifier

Advanced gradient boosting implementation
Optimized for prediction accuracy and speed



Data Preprocessing

Handles missing values using median imputation
Creates interaction features for better model performance
Standardizes numerical features
Applies SMOTE for handling class imbalance

Dependencies

pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
lightgbm==3.3.5
xgboost==1.7.5
imbalanced-learn==0.10.1
streamlit==1.22.0
joblib==1.2.0

Contributing

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Dataset source: https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis
Thanks to the scientific community for research on CKD prediction
Streamlit for the amazing web app framework

Contact
Your Name - taleyash1234@gmail.com
Project Link: [https://github.com/yourusername/ckd-prediction](https://github.com/Strax-YT/Minor_project)
