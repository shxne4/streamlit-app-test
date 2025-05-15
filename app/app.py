import streamlit as st
import pandas as pd
import joblib

# Load the model (make sure the path is correct)
@st.cache_data
def load_model():
    return joblib.load('models/diabetes_xgboost.pkl')

diabetes_model = load_model()

st.title("Diabetes Prediction App")

# User inputs for categorical features (the exact categories must match training)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "unknown"])
age_group = st.selectbox("Age Group", ["0-17", "18-39", "40-59", "60+"])
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
glucose_tolerance = st.selectbox("Glucose Tolerance", ["Normal", "Prediabetes", "Diabetes"])

# Prepare input data for model prediction
def preprocess_input(gender, smoking_history, age_group, bmi_category, glucose_tolerance):
    # Map inputs to one-hot encoded columns or categorical codes as expected
    # Here we’ll do simple one-hot for each categorical feature
    
    # For gender - one hot: gender_female, gender_male, gender_other
    gender_map = {
        "Male": [0,1,0],
        "Female": [1,0,0],
        "Other": [0,0,1]
    }
    gender_encoded = gender_map[gender]
    
    # For smoking_history (assuming 4 categories)
    smoking_map = {
        "never": [1,0,0,0],
        "former": [0,1,0,0],
        "current": [0,0,1,0],
        "unknown": [0,0,0,1]
    }
    smoking_encoded = smoking_map[smoking_history]
    
    # For age_group (4 categories)
    age_map = {
        "0-17": [1,0,0,0],
        "18-39": [0,1,0,0],
        "40-59": [0,0,1,0],
        "60+": [0,0,0,1]
    }
    age_encoded = age_map[age_group]
    
    # For bmi_category (4 categories)
    bmi_map = {
        "Underweight": [1,0,0,0],
        "Normal": [0,1,0,0],
        "Overweight": [0,0,1,0],
        "Obese": [0,0,0,1]
    }
    bmi_encoded = bmi_map[bmi_category]
    
    # For glucose_tolerance (3 categories)
    glucose_map = {
        "Normal": [1,0,0],
        "Prediabetes": [0,1,0],
        "Diabetes": [0,0,1]
    }
    glucose_encoded = glucose_map[glucose_tolerance]
    
    # Combine all encoded features into single list
    features = gender_encoded + smoking_encoded + age_encoded + bmi_encoded + glucose_encoded
    
    # Create DataFrame with column names that match your model training
    columns = [
        # gender
        'gender_female', 'gender_male', 'gender_other',
        # smoking_history
        'smoking_history_never', 'smoking_history_former', 'smoking_history_current', 'smoking_history_unknown',
        # age_group
        'age_group_0-17', 'age_group_18-39', 'age_group_40-59', 'age_group_60+',
        # bmi_category
        'bmi_category_underweight', 'bmi_category_normal', 'bmi_category_overweight', 'bmi_category_obese',
        # glucose_tolerance
        'glucose_tolerance_normal', 'glucose_tolerance_prediabetes', 'glucose_tolerance_diabetes'
    ]
    
    input_df = pd.DataFrame([features], columns=columns)
    return input_df

if st.button("Predict Diabetes Risk"):
    input_data = preprocess_input(gender, smoking_history, age_group, bmi_category, glucose_tolerance)
    
    # Predict
    prediction = diabetes_model.predict(input_data)[0]
    pred_prob = diabetes_model.predict_proba(input_data)[0][1]  # probability of positive class
    
    # Show results
    if prediction == 1:
        st.error(f"Prediction: High risk of diabetes (probability: {pred_prob:.2f})")
    else:
        st.success(f"Prediction: Low risk of diabetes (probability: {pred_prob:.2f})")
