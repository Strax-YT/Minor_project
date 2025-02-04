import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

file_path = 'D:\Mini_project\dataset\Chronic_Kidney_Dsease_data.csv'  
data = pd.read_csv(file_path)

if 'Patient ID' in data.columns:
    data.drop(['Patient ID'], axis=1, inplace=True)
if 'DoctorInCharge' in data.columns:
    data.drop(['DoctorInCharge'], axis=1, inplace=True)

data.fillna(data.mean(), inplace=True)
encoder = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
     data[col] = encoder.fit_transform(data[col])

X = data.drop('Diagnosis', axis=1)  
y = data['Diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pickle.dump(model, open('models/lifestyle_model.pkl', 'wb'))
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)




