import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("Current directory:", os.getcwd())
print("Does '../data' exist?:", os.path.exists('../data'))
if os.path.exists('../data'):
    print("Contents of data directory:", os.listdir('../data'))

# Load the dataset
data = pd.read_csv('../data/alzheimers_disease_data.csv')

# Select only the relevant columns
data = data[[
    'Age', 'Gender', 'BMI', 'Smoking', 'PhysicalActivity', 
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 
    'Depression', 'MMSE', 'ADL', 'MemoryComplaints', 'Diagnosis'
]]

# Data Cleaning and Preprocessing
# Fill missing values where applicable
data['Age'] = data['Age'].fillna(data['Age'].median())
data['BMI'] = data['BMI'].fillna(data['BMI'].median())
data['PhysicalActivity'] = data['PhysicalActivity'].fillna(data['PhysicalActivity'].median())
data['MMSE'] = data['MMSE'].fillna(data['MMSE'].median())
data['ADL'] = data['ADL'].fillna(data['ADL'].median())

# Fill binary categorical features with 0 (assuming 0 means "No")
binary_features = [
    'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 
    'Diabetes', 'Depression', 'MemoryComplaints'
]
for feature in binary_features:
    data[feature] = data[feature].fillna(0)

# Encode Gender: Male=0, Female=1
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1}).fillna(0)

scaler = StandardScaler()
data[['Age', 'BMI', 'PhysicalActivity', 'MMSE', 'ADL']] = scaler.fit_transform(
    data[['Age', 'BMI', 'PhysicalActivity', 'MMSE', 'ADL']]
)

# Save the processed data to be used in model training
data.to_csv('../data/processed_alzheimers_data.csv', index=False)
