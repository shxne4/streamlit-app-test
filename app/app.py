import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Disease Predictor", layout="centered")

@st.cache_resource
def load_pipelines():
    diab_pipe   = joblib.load("diabetes_model.pkl")
    stroke_pipe = joblib.load("stroke_model.pkl")
    heart_pipe  = joblib.load("heart_model.pkl")
    return diab_pipe, stroke_pipe, heart_pipe

# Load the pipelines
diabetes_pipe, stroke_pipe, heart_pipe = load_pipelines()

# Split into preprocessor & classifier
diab_pre,  diab_clf   = diabetes_pipe.named_steps["preprocessor"], diabetes_pipe.named_steps["classifier"]
stroke_pre, stroke_clf = stroke_pipe.named_steps["preprocessor"], stroke_pipe.named_steps["classifier"]
heart_pre,  heart_clf  = heart_pipe.named_steps["preprocessor"],  heart_pipe.named_steps["classifier"]

st.title("🩺 Disease Prediction App")
disease = st.radio("Which disease to predict?", ["Diabetes", "Stroke", "Heart Disease"])

def predict_and_show(clf, X_proc):
    pred = clf.predict(X_proc)[0]
    prob = clf.predict_proba(X_proc)[0][1]
    if pred == 1:
        st.error(f"High risk (probability: {prob:.2%})")
    else:
        st.success(f"Low risk  (probability: {prob:.2%})")

# ── DIABETES ─────────────────────────────────────────────────────────
if disease == "Diabetes":
    st.header("Diabetes Risk Predictor")
    gender          = st.selectbox("Gender", ["Male","Female","Other"])
    age             = st.number_input("Age", 0, 120, 30)
    hypertension    = st.selectbox("Hypertension (0=No,1=Yes)", [0,1])
    heart_disease_d = st.selectbox("Heart Disease (0=No,1=Yes)", [0,1])
    smoking_history = st.selectbox("Smoking History", ["never","former","unknown"])
    bmi             = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
    hba1c           = st.number_input("HbA1c Level", 3.0, 15.0, 5.5, step=0.1)
    blood_glucose   = st.number_input("Blood Glucose Level", 50.0, 300.0, 100.0, step=0.1)

    def cat_age(x):
        if x<20: return "0-19"
        if x<40: return "20-39"
        if x<60: return "40-59"
        return "60+"
    def cat_bmi(x):
        if x<18.5: return "underweight"
        if x<25:   return "normal"
        if x<30:   return "overweight"
        return "obese"
    def cat_gl(x):
        if x<140:  return "normal"
        if x<200:  return "prediabetic"
        return "diabetic"

    age_group         = cat_age(age)
    bmi_category      = cat_bmi(bmi)
    glucose_tolerance = cat_gl(blood_glucose)

    df = pd.DataFrame([{
        "gender":              gender,
        "age":                 age,
        "hypertension":        hypertension,
        "heart_disease":       heart_disease_d,
        "smoking_history":     smoking_history,
        "bmi":                 bmi,
        "hba1c_level":         hba1c,
        "blood_glucose_level": blood_glucose,
        "age_group":           age_group,
        "bmi_category":        bmi_category,
        "glucose_tolerance":   glucose_tolerance
    }])

    if st.button("Predict Diabetes"):
        Xp = diab_pre.transform(df)
        predict_and_show(diab_clf, Xp)

# ── STROKE ────────────────────────────────────────────────────────────
elif disease == "Stroke":
    st.header("Stroke Risk Predictor")
    age_s            = st.number_input("Age", 0, 120, 45)
    hypertension_s   = st.selectbox("Hypertension (0=No,1=Yes)", [0,1])
    heart_disease_s  = st.selectbox("Heart Disease (0=No,1=Yes)", [0,1])
    avg_glucose      = st.number_input("Average Glucose Level", 0.0, 400.0, 120.0, step=0.1)
    bmi_s            = st.number_input("BMI", 10.0, 60.0, 24.0, step=0.1)
    glucose_risk     = st.selectbox("Glucose Risk", ["High","Normal"])
    bp_risk_s        = st.selectbox("Blood Pressure Risk (0=No,1=Yes)", [0,1])

    df2 = pd.DataFrame([{
        "age":               age_s,
        "hypertension":      hypertension_s,
        "heart_disease":     heart_disease_s,
        "avg_glucose_level": avg_glucose,
        "bmi":               bmi_s,
        "glucose_risk":      glucose_risk,
        "bp_risk":           bp_risk_s
    }])

    if st.button("Predict Stroke"):
        Xp2 = stroke_pre.transform(df2)
        predict_and_show(stroke_clf, Xp2)

# ── HEART DISEASE ────────────────────────────────────────────────────
else:
    st.header("Heart Disease Risk Predictor")
       age_h           = st.number_input("Age", 0, 120, 50)
    sex_h           = st.selectbox("Sex", ["Male", "Female"])
    cp              = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps        = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    chol            = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs             = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
    restecg         = st.selectbox("Resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)", [0, 1, 2])
    thalach         = st.number_input("Max Heart Rate Achieved", 60, 210, 150)
    exang           = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    dataset_label   = "custom"  # if required by your model

    # Derived features
    def cat_age_h(x):
        if x < 20: return "0-19"
        elif x < 40: return "20-39"
        elif x < 60: return "40-59"
        else: return "60+"

    def cat_chol(x):
        if x < 200: return "Normal"
        elif x < 240: return "Borderline"
        else: return "High"

    def cat_bp(x):
        if x < 120: return "Normal"
        elif x < 130: return "Elevated"
        else: return "High"

    age_group_h     = cat_age_h(age_h)
    chol_risk       = cat_chol(chol)
    blood_pressure  = cat_bp(trestbps)
    sex_numeric     = 1 if sex_h == "Male" else 0

    df3 = pd.DataFrame([{
        "age":          age_h,
        "sex":          sex_numeric,
        "cp":           cp,
        "trestbps":     trestbps,
        "chol":         chol,
        "fbs":          fbs,
        "restecg":      restecg,
        "thalach":      thalach,
        "exang":        exang,
        "age_group":    age_group_h,
        "chol_risk":    chol_risk,
        "blood_pressure": blood_pressure,
        "dataset":      dataset_label
    }])

    if st.button("Predict Heart Disease"):
        Xp3 = heart_pre.transform(df3)
        predict_and_show(heart_clf, Xp3)
