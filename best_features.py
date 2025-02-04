#import necesary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load dataset
df = pd.read_csv("Chronic_Kidney_Dsease_data.csv")

# Drop non-informative columns
df_clean = df.drop(columns=["PatientID", "DoctorInCharge"], errors="ignore")

# Encode categorical features
for col in df_clean.select_dtypes(include=["object"]).columns:
    df_clean[col] = LabelEncoder().fit_transform(df_clean[col])

# Fill missing values with median
df_clean.fillna(df_clean.median(), inplace=True)

# Define features (X) and target variable (y)
X = df_clean.drop(columns=["Diagnosis"])
y = df_clean["Diagnosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Display top 15 features
print(feature_importances.head(15))





