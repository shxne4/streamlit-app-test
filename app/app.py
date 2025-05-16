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
    age_h       = st.number_input("Age", 0, 120, 55)
    sex         = st.selectbox("Sex", ["Male", "Female"])
    cp          = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps    = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol        = st.number_input("Cholesterol (chol)", 100, 400, 200)
    fbs         = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg     = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach     = st.number_input("Max Heart Rate Achieved (thalach)", 60, 220, 150)
    exang       = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    dataset_label = st.selectbox("Dataset Source", ["Cleveland", "Hungarian", "Switzerland", "VA"])

    # Derived features
    def cat_age(x):
        if x < 40: return "under_40"
        elif x < 55: return "40s-50s"
        elif x < 70: return "60s"
        return "70+"
    
    def cat_bp(x):
        return "high" if x >= 130 else "normal"

    def cat_chol(x):
        return "high" if x >= 240 else "normal"

    age_group       = cat_age(age_h)
    blood_pressure  = cat_bp(trestbps)
    chol_risk       = cat_chol(chol)
    sex_numeric     = 1 if sex == "Male" else 0  # assuming model used 1/0 for sex

    df3 = pd.DataFrame([{
        "id":             0,  # dummy value
        "age":            age_h,
        "sex":            sex_numeric,
        "dataset":        dataset_label,
        "cp":             cp,
        "trestbps":       trestbps,
        "chol":           chol,
        "fbs":            fbs,
        "restecg":        restecg,
        "thalch":         thalach,       # rename to match training typo
        "exang":          exang,
        "oldpeak":        0.0,           # default or user input later
        "num":            0,             # placeholder (target column during training)
        "age_group":      age_group,
        "blood_pressure": blood_pressure,
        "chol_risk":      chol_risk
    }])


    # 🔍 DEBUG: Show expected vs actual input columns
    st.write("Expected columns by preprocessor:", heart_pre.feature_names_in_)
    st.write("Your input DataFrame columns:", df3.columns.tolist())

    if st.button("Predict Heart Disease"):
        Xp3 = heart_pre.transform(df3)  # This is where the error happens
        predict_and_show(heart_clf, Xp3)


    if st.button("Predict Heart Disease"):
        Xp3 = heart_pre.transform(df3)
        predict_and_show(heart_clf, Xp3)
