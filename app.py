import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load('ml_pipeline/model/alzheimers_model.pkl')
scaler = joblib.load('ml_pipeline/model/scaler.pkl')

# Streamlit app title and description
st.title("Alzheimer's Disease Risk Prediction")
st.write("Enter patient details below to predict Alzheimer's disease risk.")

# Sidebar: Information about the app
st.sidebar.header("App Information")
st.sidebar.write("This app predicts the risk of Alzheimer's disease based on patient data.")

# Patient data input form
st.header("Patient Data Entry")

age = st.slider("Age", 60, 90, 70)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.slider("BMI", 15.0, 40.0, 25.0)
smoking = st.radio("Smoking Status", ["No", "Yes"])
physical_activity = st.slider("Weekly Physical Activity (hours)", 0, 10, 5)
family_history = st.radio("Family History of Alzheimer's", ["No", "Yes"])
cardio = st.radio("Cardiovascular Disease", ["No", "Yes"])
diabetes = st.radio("Diabetes", ["No", "Yes"])
depression = st.radio("Depression", ["No", "Yes"])
memory_complaints = st.radio("Memory Complaints", ["No", "Yes"])
mmse = st.slider("MMSE Score", 0, 30, 25)
adl = st.slider("ADL Score", 0, 10, 5)

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [1 if gender == "Female" else 0],
    "BMI": [bmi],
    "Smoking": [1 if smoking == "Yes" else 0],
    "PhysicalActivity": [physical_activity],
    "FamilyHistoryAlzheimers": [1 if family_history == "Yes" else 0],
    "CardiovascularDisease": [1 if cardio == "Yes" else 0],
    "Diabetes": [1 if diabetes == "Yes" else 0],
    "Depression": [1 if depression == "Yes" else 0],
    "MMSE": [mmse],
    "ADL": [adl],
    "MemoryComplaints": [1 if memory_complaints == "Yes" else 0]
})

# Reorder columns to match the training data order exactly
input_data = input_data[[
    "Age", "Gender", "BMI", "Smoking", "PhysicalActivity", 
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", 
    "Depression", "MMSE", "ADL", "MemoryComplaints"
]]

# Scale continuous features using the loaded scaler
input_data[['Age', 'BMI', 'PhysicalActivity', 'MMSE', 'ADL']] = scaler.transform(
    input_data[['Age', 'BMI', 'PhysicalActivity', 'MMSE', 'ADL']]
)

# Debug: Display input data after scaling for verification
#st.write("Scaled Input Data for Prediction:", input_data)

# Prediction button
if st.button("Predict Alzheimer's Risk"):
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]
    risk = "High Risk" if prediction == 1 else "Low Risk"
    st.subheader(f"Prediction Result: {risk}")

# Footer Information
st.sidebar.write("---")
st.sidebar.write("Developed for Alzheimer's disease risk prediction demonstration.")
