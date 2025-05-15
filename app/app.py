import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained XGBoost model
MODEL_PATH = 'models/diabetes_xgboost.pkl'
model = joblib.load(MODEL_PATH)

# Define all possible categories for one-hot encoding
gender_categories = ['Female', 'Male', 'Other']
smoking_categories = ['Never', 'Former', 'Unknown', 'Current']
age_group_categories = ['<30', '30-45', '46-60', '60+']
bmi_categories = ['Under', 'Normal', 'Over', 'Obese']
glucose_categories = ['Normal', 'Prediabetes', 'Diabetes']

# Helper function to one-hot encode a single row
def encode_input(gender, smoking, age_group, bmi_category, glucose_tolerance):
    # Create a zero-filled dictionary for all one-hot columns
    input_dict = {}

    for cat in gender_categories:
        input_dict[f'gender_{cat}'] = 1 if gender == cat else 0
    for cat in smoking_categories:
        input_dict[f'smoking_history_{cat}'] = 1 if smoking == cat else 0
    for cat in age_group_categories:
        input_dict[f'age_group_{cat}'] = 1 if age_group == cat else 0
    for cat in bmi_categories:
        input_dict[f'bmi_category_{cat}'] = 1 if bmi_category == cat else 0
    for cat in glucose_categories:
        input_dict[f'glucose_tolerance_{cat}'] = 1 if glucose_tolerance == cat else 0

    # Return as DataFrame
    return pd.DataFrame([input_dict])

# Streamlit UI
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Risk Prediction")
st.markdown("This app predicts diabetes risk based on lifestyle and health indicators.")

# Inputs
gender = st.selectbox("Gender", gender_categories)
smoking = st.selectbox("Smoking History", smoking_categories)
age_group = st.selectbox("Age Group", age_group_categories)
bmi_category = st.selectbox("BMI Category", bmi_categories)
glucose_tolerance = st.selectbox("Glucose Tolerance", glucose_categories)

# Prediction button
if st.button("Predict"):
    input_df = encode_input(gender, smoking, age_group, bmi_category, glucose_tolerance)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of Diabetes. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low risk of Diabetes. (Probability: {probability:.2f})")
