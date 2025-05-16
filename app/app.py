import streamlit as st
import pandas as pd
import joblib

# Load diabetes model
@st.cache_resource
def load_diabetes_model():
    return joblib.load('diabetes_model.pkl')

#diabetes_model = load_diabetes_model()

# App title
st.title("Diabetes Prediction App")
st.write("Enter patient information to predict the likelihood of diabetes.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "not current", "ever", "No Info"])
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Feature engineering (same logic used in training)
def categorize_age(age):
    if age < 20:
        return '0-19'
    elif age < 40:
        return '20-39'
    elif age < 60:
        return '40-59'
    else:
        return '60+'

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    else:
        return 'obese'

def categorize_glucose(glucose):
    if glucose < 140:
        return 'normal'
    elif glucose < 200:
        return 'prediabetic'
    else:
        return 'diabetic'

age_group = categorize_age(age)
bmi_category = categorize_bmi(bmi)
glucose_tolerance = categorize_glucose(blood_glucose_level)

# Build input DataFrame
input_df = pd.DataFrame([{
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
}])

# Predict
if st.button("Predict Diabetes"):
    prediction = diabetes_model.predict(input_df)[0]
    prob = diabetes_model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Diabetes Prediction:** {'Positive' if prediction == 1 else 'Negative'}")
    st.write(f"**Probability:** {prob:.2%}")
