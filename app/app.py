import streamlit as st
import pandas as pd
import joblib
import os

# Load models
diabetes_model = joblib.load("app/diabetes_model.pkl")
stroke_model = joblib.load("app/stroke_model.pkl")
heart_model = joblib.load("app/heart_model.pkl")

# Define categorical columns for filling
diabetes_cat_cols = ['gender', 'smoking_history', 'age_group', 'bmi_category', 'glucose_tolerance']
stroke_cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group', 'bmi_category']
heart_cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'age_group', 'blood_pressure', 'chol_risk', 'dataset']

st.title("Disease Prediction App")

# User input form
with st.form("user_input"):
    st.subheader("Enter Patient Details")

    # Common fields
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    age = st.number_input("Age", 1, 120, 30)
    bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
    smoking = st.selectbox("Smoking History", ['never', 'former', 'current', 'No Info'])

    # Diabetes-specific
    hba1c = st.number_input("HbA1c Level", 4.0, 15.0, 5.5)
    glucose = st.number_input("Blood Glucose Level", 50, 300, 90)

    # Stroke-specific
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
    work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])

    # Heart-specific
    sex = 1 if gender == 'Male' else 0
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate (thalach)", 60, 210, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])

    # Derived fields
    age_group = pd.cut([age], bins=[0, 20, 40, 60, 120], labels=["0-20", "21-40", "41-60", "60+"])[0]
    bmi_category = pd.cut([bmi], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])[0]
    glucose_tolerance = "Normal" if glucose < 140 else "Prediabetes" if glucose < 200 else "Diabetes"
    chol_risk = "High" if chol > 240 else "Borderline" if chol > 200 else "Normal"
    blood_pressure = "Normal" if trestbps < 120 else "Elevated" if trestbps < 130 else "High"

    submitted = st.form_submit_button("Predict")

if submitted:
    # DataFrames for each disease
    df1 = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'bmi': bmi,
        'smoking_history': smoking,
        'hba1c_level': hba1c,
        'blood_glucose_level': glucose,
        'age_group': age_group,
        'bmi_category': bmi_category,
        'glucose_tolerance': glucose_tolerance
    }])

    df2 = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'smoking_status': smoking,
        'bmi': bmi,
        'age_group': age_group,
        'bmi_category': bmi_category
    }])

    df3 = pd.DataFrame([{
        'sex': sex,
        'age': age,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'age_group': age_group,
        'blood_pressure': blood_pressure,
        'chol_risk': chol_risk,
        'dataset': "Missing"  # default fill value
    }])

    # Handle missing categorical values (critical!)
    df1[diabetes_cat_cols] = df1[diabetes_cat_cols].fillna("Missing")
    df2[stroke_cat_cols] = df2[stroke_cat_cols].fillna("Missing")
    df3[heart_cat_cols] = df3[heart_cat_cols].fillna("Missing")

    # Run predictions
    diabetes_pred = diabetes_model.predict(df1)[0]
    stroke_pred = stroke_model.predict(df2)[0]
    heart_pred = heart_model.predict(df3)[0]

    # Output
    st.subheader("Predictions")
    st.write(f"🩸 **Diabetes:** {'Positive' if diabetes_pred else 'Negative'}")
    st.write(f"🧠 **Stroke:** {'High Risk' if stroke_pred else 'Low Risk'}")
    st.write(f"❤️ **Heart Disease:** {'Detected' if heart_pred else 'Not Detected'}")
