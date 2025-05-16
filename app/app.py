import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Disease Predictor", layout="centered")

@st.cache_resource
def load_pipelines():
    diab_pipe   = joblib.load("app/diabetes_model.pkl")
    stroke_pipe = joblib.load("app/stroke_model.pkl")
    heart_pipe  = joblib.load("app/heart_model.pkl")
    return diab_pipe, stroke_pipe, heart_pipe

diabetes_pipe, stroke_pipe, heart_pipe = load_pipelines()
diab_pre, diab_clf     = diabetes_pipe.named_steps["preprocessor"], diabetes_pipe.named_steps["classifier"]
stroke_pre, stroke_clf = stroke_pipe.named_steps["preprocessor"], stroke_pipe.named_steps["classifier"]
heart_pre, heart_clf   = heart_pipe.named_steps["preprocessor"],  heart_pipe.named_steps["classifier"]

st.title("🩺 Disease Prediction App")
disease = st.radio("Which to predict?", ["Diabetes", "Stroke", "Heart Disease"])

def predict_and_show(clf, X_proc):
    p = clf.predict(X_proc)[0]
    prob = clf.predict_proba(X_proc)[0][1]
    if p == 1: st.error(f"High risk ({prob:.2%})")
    else:      st.success(f"Low risk  ({prob:.2%})")

# … [Diabetes and Stroke blocks as before] …

# ── HEART DISEASE ───────────────────────────────────────────────────
if disease == "Heart Disease":
    st.header("Heart Disease Risk Predictor")
    # Numeric inputs
    age_h          = st.number_input("Age", 0,120,50)
    hypertension_h = st.selectbox("Hypertension", [0,1])
    avg_gluc_h     = st.number_input("Avg Glucose Level",0.0,400.0,130.0)
    bmi_h          = st.number_input("BMI",10.0,60.0,26.0)
    stroke_h       = st.selectbox("Stroke History", [0,1])
    # Categorical inputs
    sex            = st.selectbox("Sex", ["Male","Female"])
    cp             = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
    fbs            = st.selectbox("Fasting BS >120mg/dl", [0,1])
    restecg        = st.selectbox("Resting ECG (0–2)", [0,1,2])
    exang          = st.selectbox("Exercise‑Induced Angina", [0,1])
    age_group_h    = st.selectbox("Age Group", ["<30","30-45","46-60","60+"])
    bp_risk        = st.selectbox("Blood Pressure Risk", ["Low","Normal","High"])
    chol_risk      = st.selectbox("Cholesterol Risk", ["Normal","High"])
    dataset        = st.selectbox("Dataset Source", ["raw","aggregated"])

    df3 = pd.DataFrame([{
        "age":               age_h,
        "hypertension":      hypertension_h,
        "avg_glucose_level": avg_gluc_h,
        "bmi":               bmi_h,
        "stroke":            stroke_h,
        "sex":               sex,
        "cp":                cp,
        "fbs":               fbs,
        "restecg":           restecg,
        "exang":             exang,
        "age_group":         age_group_h,
        "blood_pressure":    bp_risk,
        "chol_risk":         chol_risk,
        "dataset":           dataset
    }])

    # --- Hack: strip nan from the OneHotEncoder's categories_ ---
    # It's the second transformer in the ColumnTransformer
    cat_enc = heart_pre.transformers_[1][1]
    cleaned = []
    for cats in cat_enc.categories_:
        cleaned.append([c for c in cats if not (isinstance(c, float) and np.isnan(c))])
    cat_enc.categories_ = cleaned

    # --- Align & reorder columns ---
    expected = heart_pre.feature_names_in_
    for col in expected:
        if col not in df3.columns:
            df3[col] = 0
    df3 = df3[expected]

    if st.button("Predict Heart Disease"):
        Xp3 = heart_pre.transform(df3)
        predict_and_show(heart_clf, Xp3)
