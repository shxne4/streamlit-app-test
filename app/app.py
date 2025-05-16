import streamlit as st
import pandas as pd
import joblib

# 1) Load the full pipeline (preprocessor + SMOTE + classifier) you saved
@st.cache_resource
def load_pipeline():
    return joblib.load("diabetes_model.pkl")

pipeline = load_pipeline()

# 2) Split it into transformer and classifier for inference
preprocessor = pipeline.named_steps["preprocessor"]
classifier   = pipeline.named_steps["classifier"]

# 3) Streamlit UI
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Risk Prediction")
st.write("Enter patient data below and click Predict.")

# Inputs
gender           = st.selectbox("Gender", ["Male", "Female", "Other"])
age              = st.number_input("Age", 0, 120, 30)
hypertension     = st.selectbox("Hypertension", [0, 1])
heart_disease    = st.selectbox("Heart Disease", [0, 1])
smoking_history  = st.selectbox("Smoking History", ["Never", "Former", "Unknown"])
bmi              = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
hba1c_level      = st.number_input("HbA1c Level", 3.0, 15.0, 5.5, step=0.1)
blood_glucose    = st.number_input("Blood Glucose Level", 50.0, 300.0, 100.0, step=0.1)

# Bucket functions must match training exactly
def cat_age(x):
    if x < 20:    return "0-19"
    if x < 40:    return "20-39"
    if x < 60:    return "40-59"
    return "60+"

def cat_bmi(x):
    if x < 18.5:  return "underweight"
    if x < 25:    return "normal"
    if x < 30:    return "overweight"
    return "obese"

def cat_gl(x):
    if x < 140:   return "normal"
    if x < 200:   return "prediabetic"
    return "diabetic"

age_group        = cat_age(age)
bmi_category     = cat_bmi(bmi)
glucose_tolerance= cat_gl(blood_glucose)

# Build DataFrame with ALL features the pipeline expects
input_df = pd.DataFrame([{
    "gender":              gender,
    "age":                 age,
    "hypertension":        hypertension,
    "heart_disease":       heart_disease,
    "smoking_history":     smoking_history,
    "bmi":                 bmi,
    "hba1c_level":         hba1c_level,
    "blood_glucose_level": blood_glucose,
    "age_group":           age_group,
    "bmi_category":        bmi_category,
    "glucose_tolerance":   glucose_tolerance
}])

# Run inference
if st.button("Predict Diabetes"):
    # 4) Preprocess (scale + one‑hot encode)
    X_proc = preprocessor.transform(input_df)
    # 5) Classify
    pred   = classifier.predict(X_proc)[0]
    prob   = classifier.predict_proba(X_proc)[0][1]

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"High risk of diabetes — probability: {prob:.2%}")
    else:
        st.success(f"Low risk of diabetes — probability: {prob:.2%}")
