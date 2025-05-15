import streamlit as st
import pandas as pd
import joblib

# Load the saved diabetes model
@st.cache_data
def load_model():
    return joblib.load('models/diabetes_xgboost.pkl')

diabetes_model = load_model()

st.title("Diabetes Prediction")

# Input widgets
gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'])
smoking_history = st.selectbox("Smoking History", options=['never', 'former', 'current', 'not available'])
age_group = st.selectbox("Age Group", options=['0-17', '18-44', '45-64', '65-74', '75-84', '85+'])
bmi_category = st.selectbox("BMI Category", options=['Underweight', 'Normal', 'Overweight', 'Obese'])
glucose_tolerance = st.selectbox("Glucose Tolerance", options=['Normal', 'Prediabetes', 'Diabetes'])

# Button to trigger prediction
if st.button("Predict"):

    # Create input DataFrame with one row
    input_data = pd.DataFrame({
        'gender': [gender],
        'smoking_history': [smoking_history],
        'age_group': [age_group],
        'bmi_category': [bmi_category],
        'glucose_tolerance': [glucose_tolerance]
    })

    # Predict diabetes
    prediction = diabetes_model.predict(input_data)[0]

    if prediction == 1:
        st.success("Prediction: You are likely to have diabetes.")
    else:
        st.info("Prediction: You are unlikely to have diabetes.")
