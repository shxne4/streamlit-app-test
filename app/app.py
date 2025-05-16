import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Disease Predictor", layout="centered")

@st.cache_resource
def load_pipelines():
    diab_pipe   = joblib.load("app/diabetes_model.pkl")
    stroke_pipe = joblib.load("app/stroke_model.pkl")
    heart_pipe  = joblib.load("app/heart_model.pkl")
    return diab_pipe, stroke_pipe, heart_pipe

diabetes_pipe, stroke_pipe, heart_pipe = load_pipelines()
diab_pre,  diab_clf   = diabetes_pipe.named_steps["preprocessor"], diabetes_pipe.named_steps["classifier"]
stroke_pre, stroke_clf = stroke_pipe.named_steps["preprocessor"], stroke_pipe.named_steps["classifier"]
heart_pre, heart_clf   = heart_pipe.named_steps["preprocessor"],  heart_pipe.named_steps["classifier"]

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
    glucose         = st.number_input("Blood Glucose Level", 50.0, 300.0, 100.0, step=0.1)

    # Bucket functions (must match training exactly)
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
    glucose_tol       = cat_gl(glucose)

    df = pd.DataFrame([{
      "gender":              gender,
      "age":                 age,
      "hypertension":        hypertension,
      "heart_disease":       heart_disease_d,
      "smoking_history":     smoking_history,
      "bmi":                 bmi,
      "hba1c_level":         hba1c,
      "blood_glucose_level": glucose,
      "age_group":           age_group,
      "bmi_category":        bmi_category,
      "glucose_tolerance":   glucose_tol
    }])

    if st.button("Predict Diabetes"):
        Xp = diab_pre.transform(df)
        predict_and_show(diab_clf, Xp)

# ── STROKE ────────────────────────────────────────────────────────────
elif disease == "Stroke":
    st.header("Stroke Risk Predictor")
    age            = st.number_input("Age", 0, 120, 45)
    hypertension   = st.selectbox("Hypertension (0=No,1=Yes)", [0,1])
    heart_disease_s= st.selectbox("Heart Disease (0=No,1=Yes)", [0,1])
    avg_glucose    = st.number_input("Avg Glucose Level", 0.0, 400.0, 120.0, step=0.1)
    bmi_s          = st.number_input("BMI", 10.0, 60.0, 24.0, step=0.1)
    bp_risk_s      = st.selectbox("BP Risk (0=No,1=Yes)", [0,1])
    glucose_risk   = st.selectbox("Glucose Risk", ["High","Normal"])

    df2 = pd.DataFrame([{
      "age":               age,
      "hypertension":      hypertension,
      "heart_disease":     heart_disease_s,
      "avg_glucose_level": avg_glucose,
      "bmi":               bmi_s,
      "bp_risk":           bp_risk_s,
      "glucose_risk":      glucose_risk
    }])

    if st.button("Predict Stroke"):
        Xp2 = stroke_pre.transform(df2)
        predict_and_show(stroke_clf, Xp2)

# ── HEART DISEASE ────────────────────────────────────────────────────
else:
    st.header("Heart Disease Risk Predictor")
    age_h           = st.number_input("Age", 0, 120, 50)
    hypertension_h  = st.selectbox("Hypertension (0=No,1=Yes)", [0,1])
    avg_glucose_h   = st.number_input("Avg Glucose Level", 0.0, 400.0, 130.0, step=0.1)
    bmi_h           = st.number_input("BMI", 10.0, 60.0, 26.0, step=0.1)
    stroke_h        = st.selectbox("Stroke History (0=No,1=Yes)", [0,1])
    age_group_h     = st.selectbox("Age Group", ["<30","30-45","46-60","60+"])
    sex             = st.selectbox("Sex", ["Male","Female"])
    cp              = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
    fbs             = st.selectbox("Fasting Blood Sugar >120mg/dl (0=No,1=Yes)", [0,1])
    restecg         = st.selectbox("Resting ECG (0–2)", [0,1,2])
    exang           = st.selectbox("Exercise Induced Angina (0=No,1=Yes)", [0,1])
    bp_risk_h       = st.selectbox("BP Risk (0=No,1=Yes)", [0,1])
    chol_risk       = st.selectbox("Cholesterol Risk", ["Normal","High"])
    dataset         = st.selectbox("Dataset Source", ["raw","aggregated"])

    df3 = pd.DataFrame([{
      "age": age_h,
      "hypertension": hypertension_h,
      "avg_glucose_level": avg_glucose_h,
      "bmi": bmi_h,
      "stroke": stroke_h,
      "age_group": age_group_h,
      "sex": sex,
      "cp": cp,
      "fbs": fbs,
      "restecg": restecg,
      "exang": exang,
      "blood_pressure": bp_risk_h,
      "chol_risk": chol_risk,
      "dataset": dataset
    }])

    if st.button("Predict Heart Disease"):
        Xp3 = heart_pre.transform(df3)
        predict_and_show(heart_clf, Xp3)
