import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load pre-trained models
diabetes_model = joblib.load('../models/diabetes_xgboost.pkl')
stroke_model = joblib.load('../models/stroke_random_forest.pkl')
heart_model = joblib.load('../models/heart_logistic_regression.pkl')

st.title("Disease Prediction App")

disease_type = st.selectbox("Select Disease for Prediction", ['Diabetes', 'Stroke', 'Heart Disease'])

def predict_disease(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return 'Positive' if prediction == 1 else 'Negative'

# === Form Fields ===
if disease_type == 'Diabetes':
    st.header("Diabetes Prediction Input")
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    age = st.number_input("Age", 0, 120)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    smoking_history = st.selectbox("Smoking History", ['never', 'current', 'former', 'not current', 'ever', 'unknown'])
    bmi = st.number_input("BMI")
    hba1c_level = st.number_input("HbA1c Level")
    blood_glucose_level = st.number_input("Blood Glucose Level")
    age_group = st.selectbox("Age Group", ['Child', 'Young Adult', 'Adult', 'Senior'])
    bmi_category = st.selectbox("BMI Category", ['Underweight', 'Normal', 'Overweight', 'Obese'])
    glucose_tolerance = st.selectbox("Glucose Tolerance", ['Normal', 'Impaired', 'Diabetes'])

    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'hba1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level,
        'age_group': age_group,
        'bmi_category': bmi_category,
        'glucose_tolerance': glucose_tolerance
    }

    if st.button("Predict"):
        result = predict_disease(diabetes_model, input_data)
        st.success(f"Prediction: {result}")

elif disease_type == 'Stroke':
    st.header("Stroke Prediction Input")
    age = st.number_input("Age", 0, 120)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")
    age_group = st.selectbox("Age Group", ['Child', 'Young Adult', 'Adult', 'Senior'])
    gender_male = st.selectbox("Gender: Male?", [0, 1])
    gender_other = st.selectbox("Gender: Other?", [0, 1])
    ever_married_yes = st.selectbox("Ever Married?", [0, 1])
    work_type_govt_job = st.selectbox("Work: Govt Job?", [0, 1])
    work_type_never_worked = st.selectbox("Work: Never Worked?", [0, 1])
    work_type_private = st.selectbox("Work: Private?", [0, 1])
    work_type_self_employed = st.selectbox("Work: Self-Employed?", [0, 1])
    residence_type_urban = st.selectbox("Urban Residence?", [0, 1])
    smoking_status_former = st.selectbox("Smoking: Former?", [0, 1])
    smoking_status_never = st.selectbox("Smoking: Never?", [0, 1])
    smoking_status_unknown = st.selectbox("Smoking: Unknown?", [0, 1])
    glucose_risk = st.selectbox("Glucose Risk?", [0, 1])
    bp_risk = st.selectbox("Blood Pressure Risk?", [0, 1])

    input_data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'age_group': age_group,
        'gender_male': gender_male,
        'gender_other': gender_other,
        'ever_married_yes': ever_married_yes,
        'work_type_govt_job': work_type_govt_job,
        'work_type_never_worked': work_type_never_worked,
        'work_type_private': work_type_private,
        'work_type_self-employed': work_type_self_employed,
        'Residence_type_urban': residence_type_urban,
        'smoking_status_former': smoking_status_former,
        'smoking_status_never': smoking_status_never,
        'smoking_status_unknown': smoking_status_unknown,
        'glucose_risk': glucose_risk,
        'bp_risk': bp_risk
    }

    if st.button("Predict"):
        result = predict_disease(stroke_model, input_data)
        st.success(f"Prediction: {result}")

elif disease_type == 'Heart Disease':
    st.header("Heart Disease Prediction Input")
    age = st.number_input("Age", 0, 120)
    hypertension = st.selectbox("Hypertension", [0, 1])
    avg_glucose_level = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")
    stroke = st.selectbox("Stroke History", [0, 1])
    age_group = st.selectbox("Age Group", ['Child', 'Young Adult', 'Adult', 'Senior'])
    gender_male = st.selectbox("Gender: Male?", [0, 1])
    gender_other = st.selectbox("Gender: Other?", [0, 1])
    ever_married_yes = st.selectbox("Ever Married?", [0, 1])
    work_type_govt_job = st.selectbox("Work: Govt Job?", [0, 1])
    work_type_never_worked = st.selectbox("Work: Never Worked?", [0, 1])
    work_type_private = st.selectbox("Work: Private?", [0, 1])
    work_type_self_employed = st.selectbox("Work: Self-Employed?", [0, 1])
    residence_type_urban = st.selectbox("Urban Residence?", [0, 1])
    smoking_status_former = st.selectbox("Smoking: Former?", [0, 1])
    smoking_status_never = st.selectbox("Smoking: Never?", [0, 1])
    smoking_status_unknown = st.selectbox("Smoking: Unknown?", [0, 1])
    glucose_risk = st.selectbox("Glucose Risk?", [0, 1])
    bp_risk = st.selectbox("Blood Pressure Risk?", [0, 1])

    input_data = {
        'age': age,
        'hypertension': hypertension,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'stroke': stroke,
        'age_group': age_group,
        'gender_male': gender_male,
        'gender_other': gender_other,
        'ever_married_yes': ever_married_yes,
        'work_type_govt_job': work_type_govt_job,
        'work_type_never_worked': work_type_never_worked,
        'work_type_private': work_type_private,
        'work_type_self-employed': work_type_self_employed,
        'Residence_type_urban': residence_type_urban,
        'smoking_status_former': smoking_status_former,
        'smoking_status_never': smoking_status_never,
        'smoking_status_unknown': smoking_status_unknown,
        'glucose_risk': glucose_risk,
        'bp_risk': bp_risk
    }

    if st.button("Predict"):
        result = predict_disease(heart_model, input_data)
        st.success(f"Prediction: {result}")
