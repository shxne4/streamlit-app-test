import streamlit as st
import pandas as pd
import joblib

# Load models
diabetes_model = joblib.load("app/diabetes_model.pkl")
stroke_model = joblib.load("app/stroke_model.pkl")
heart_model = joblib.load("app/heart_model.pkl")

# Define categorical columns for missing handling
diabetes_cat_cols = ['gender', 'smoking_history', 'age_group', 'bmi_category', 'glucose_tolerance']
stroke_cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group', 'bmi_category']
heart_cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'age_group', 'blood_pressure', 'chol_risk', 'dataset']

st.title("Disease Prediction App")

tab1, tab2, tab3 = st.tabs(["🩸 Diabetes", "🧠 Stroke", "❤️ Heart Disease"])

# ---------------------------
# Diabetes Tab
# ---------------------------
with tab1:
    st.header("Diabetes Prediction")

    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    age = st.slider("Age", 1, 120, 30)
    bmi = st.slider("BMI", 10.0, 60.0, 22.0)
    smoking = st.selectbox("Smoking History", ['never', 'former', 'current', 'No Info'])
    hba1c = st.slider("HbA1c Level", 4.0, 15.0, 5.5)
    glucose = st.slider("Blood Glucose Level", 50, 300, 90)

    if st.button("Predict Diabetes"):
        age_group = pd.cut([age], [0, 20, 40, 60, 120], labels=["0-20", "21-40", "41-60", "60+"])[0]
        bmi_category = pd.cut([bmi], [0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])[0]
        glucose_tolerance = "Normal" if glucose < 140 else "Prediabetes" if glucose < 200 else "Diabetes"

        df = pd.DataFrame([{
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

        df[diabetes_cat_cols] = df[diabetes_cat_cols].fillna("Missing")
        pred = diabetes_model.predict(df)[0]
        st.success("Diabetes: **Positive**" if pred else "Diabetes: **Negative**")

# ---------------------------
# Stroke Tab
# ---------------------------
with tab2:
    st.header("Stroke Prediction")

    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], key="gender_stroke")
    age = st.slider("Age", 1, 120, 30, key="age_stroke")
    bmi = st.slider("BMI", 10.0, 60.0, 22.0, key="bmi_stroke")
    smoking = st.selectbox("Smoking Status", ['never', 'former', 'current', 'No Info'], key="smoking_stroke")
    hypertension = st.selectbox("Hypertension", [0, 1], key="hypertension")
    heart_disease = st.selectbox("Heart Disease", [0, 1], key="heart_disease")
    ever_married = st.selectbox("Ever Married", ['Yes', 'No'], key="married")
    work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], key="work")
    residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'], key="residence")

    if st.button("Predict Stroke"):
        age_group = pd.cut([age], [0, 20, 40, 60, 120], labels=["0-20", "21-40", "41-60", "60+"])[0]
        bmi_category = pd.cut([bmi], [0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])[0]

        df = pd.DataFrame([{
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

        df[stroke_cat_cols] = df[stroke_cat_cols].fillna("Missing")
        pred = stroke_model.predict(df)[0]
        st.success("Stroke Risk: **High**" if pred else "Stroke Risk: **Low**")

# ---------------------------
# Heart Disease Tab
# ---------------------------
with tab3:
    st.header("Heart Disease Prediction")

    gender = st.selectbox("Gender", ['Male', 'Female'], key="gender_heart")
    age = st.slider("Age", 1, 120, 30, key="age_heart")
    sex = 1 if gender == "Male" else 0
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes)", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes)", [0, 1])

    if st.button("Predict Heart Disease"):
        age_group = pd.cut([age], [0, 20, 40, 60, 120], labels=["0-20", "21-40", "41-60", "60+"])[0]
        chol_risk = "High" if chol > 240 else "Borderline" if chol > 200 else "Normal"
        blood_pressure = "Normal" if trestbps < 120 else "Elevated" if trestbps < 130 else "High"

        df = pd.DataFrame([{
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
            'dataset': "Missing"
        }])

        df[heart_cat_cols] = df[heart_cat_cols].fillna("Missing")
        pred = heart_model.predict(df)[0]
        st.success("Heart Disease: **Detected**" if pred else "Heart Disease: **Not Detected**")
