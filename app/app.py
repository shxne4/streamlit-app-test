import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the diabetes XGBoost model
@st.cache_resource
def load_model():
    return joblib.load('models/diabetes_xgboost.pkl')

model = load_model()

st.title("Diabetes Prediction using XGBoost")

# Input fields for diabetes features
gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", options=[0, 1])
heart_disease = st.selectbox("Heart Disease", options=[0, 1])
smoking_history = st.selectbox("Smoking History", options=["never", "former", "current", "unknown"])
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
hba1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5, format="%.2f")
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=600.0, value=100.0, format="%.2f")
age_group = st.selectbox("Age Group", options=["0-18", "19-35", "36-50", "51-65", "65+"])
bmi_category = st.selectbox("BMI Category", options=["Underweight", "Normal", "Overweight", "Obese"])
glucose_tolerance = st.selectbox("Glucose Tolerance", options=["Normal", "Impaired", "Diabetic"])

# Preprocessing function (adapt this to your actual feature encoding)
def preprocess_input():
    # Example: Map categorical variables to numeric values or one-hot encode
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    smoking_map = {"never": 0, "former": 1, "current": 2, "unknown": 3}
    age_group_map = {"0-18": 0, "19-35": 1, "36-50": 2, "51-65": 3, "65+": 4}
    bmi_cat_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
    glucose_tol_map = {"Normal": 0, "Impaired": 1, "Diabetic": 2}

    data = [
        gender_map[gender],
        age,
        hypertension,
        heart_disease,
        smoking_map[smoking_history],
        bmi,
        hba1c_level,
        blood_glucose_level,
        age_group_map[age_group],
        bmi_cat_map[bmi_category],
        glucose_tol_map[glucose_tolerance]
    ]

    return np.array(data).reshape(1, -1)

if st.button("Predict Diabetes"):
    features = preprocess_input()
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    st.write(f"### Prediction: {'Diabetes Positive' if prediction == 1 else 'Diabetes Negative'}")
    st.write(f"### Probability of Diabetes: {proba:.2%}")
