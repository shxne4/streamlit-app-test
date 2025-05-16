import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Disease Predictor", layout="wide")

@st.cache_resource
def load_models():
    # models live in the same folder as this app
    diabetes_pipeline = joblib.load('app/diabetes_model.pkl')
    stroke_pipeline   = joblib.load('app/stroke_model.pkl')
    heart_pipeline    = joblib.load('app/heart_model.pkl')
    return diabetes_pipeline, stroke_pipeline, heart_pipeline

diabetes_pipeline, stroke_pipeline, heart_pipeline = load_models()

st.title("🩺 Disease Prediction Dashboard")
st.write("Select a disease, fill in the risk factors, and click Predict.")

disease = st.sidebar.radio("Which disease to predict?", 
                           ["Diabetes", "Stroke", "Heart Disease"])

def predict_and_show(pipeline, df):
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][1]
    if pred == 1:
        st.error(f"High risk ({prob:.2%})")
    else:
        st.success(f"Low risk ({prob:.2%})")

if disease == "Diabetes":
    st.header("Diabetes Risk Predictor")
    # Gather inputs
    gender = st.selectbox("Gender", ["Male","Female"])
    age    = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension    = st.selectbox("Hypertension (0=No,1=Yes)", [0,1])
    heart_disease_d = st.selectbox("Heart Disease (0=No,1=Yes)", [0,1])
    smoking_history = st.selectbox("Smoking History", ["Never","Former","Unknown"])
    bmi             = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    hba1c           = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    glucose         = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

    # Build DataFrame matching training features
    df = pd.DataFrame([{
      'gender': gender,
      'age': age,
      'hypertension': hypertension,
      'heart_disease': heart_disease_d,
      'smoking_history': smoking_history,
      'bmi': bmi,
      'hba1c_level': hba1c,
      'blood_glucose_level': glucose
    }])

    if st.button("Predict Diabetes"):
        predict_and_show(diabetes_pipeline, df)

elif disease == "Stroke":
    st.header("Stroke Risk Predictor")
    age         = st.number_input("Age", 0, 120, 45)
    hypertension= st.selectbox("Hypertension (0=No,1=Yes)", [0,1])
    heart_dis   = st.selectbox("Heart Disease (0=No,1=Yes)", [0,1])
    avg_gluc    = st.number_input("Average Glucose Level", 0.0, 400.0, 120.0)
    bmi_s       = st.number_input("BMI", 10.0, 60.0, 24.0)
    age_group   = st.selectbox("Age Group", ["<30","30-45","46-60","60+"])
    gender_sel  = st.selectbox("Gender", ["Male","Female","Other"])
    married     = st.selectbox("Ever Married (No/Yes)", ["No","Yes"])
    work_type   = st.selectbox("Work Type", ["Govt_job","Never_worked","Private","Self-employed","Children"])
    residence   = st.selectbox("Residence Type", ["Urban","Rural"])
    smoke_stat  = st.selectbox("Smoking Status", ["Former","Never","Unknown","Smokes"])
    glucose_risk= st.selectbox("Glucose Risk", ["High","Normal"])
    bp_risk     = st.selectbox("Blood Pressure Risk (0/1)", [0,1])

    # One‑hot encode exactly the same columns used at training:
    data = {
      'age': age,
      'hypertension': hypertension,
      'heart_disease': heart_dis,
      'avg_glucose_level': avg_gluc,
      'bmi': bmi_s,
      # age_group → pipeline’s OneHotEncoder
      'age_group': age_group,
      # gender_male / gender_other
      'gender_male': 1 if gender_sel=="Male" else 0,
      'gender_other':1 if gender_sel=="Other" else 0,
      # ever_married_yes
      'ever_married_yes': 1 if married=="Yes" else 0,
      # work_type_*
      'work_type_govt_job':       1 if work_type=="Govt_job" else 0,
      'work_type_never_worked':   1 if work_type=="Never_worked" else 0,
      'work_type_private':        1 if work_type=="Private" else 0,
      'work_type_self-employed':  1 if work_type=="Self-employed" else 0,
      # children category wasn’t explicitly named in pipeline → assume all zeros if not present
      'Residence_type_urban':     1 if residence=="Urban" else 0,
      # smoking_status_*
      'smoking_status_former':    1 if smoke_stat=="Former" else 0,
      'smoking_status_never':     1 if smoke_stat=="Never" else 0,
      'smoking_status_unknown':   1 if smoke_stat=="Unknown" else 0,
      'glucose_risk':             glucose_risk,
      'bp_risk':                  bp_risk
    }
    # fill missing one‑hots for pipeline
    for col in stroke_pipeline.steps[0][1].transformers_[1][2]:
        # actually pipeline’s ColumnTransformer lists cat_cols only, numeric covers the rest
        pass

    df = pd.DataFrame([data])
    if st.button("Predict Stroke"):
        predict_and_show(stroke_pipeline, df)

else:  # Heart Disease
    st.header("Heart Disease Risk Predictor")
    age      = st.number_input("Age", 0, 120, 50)
    hypertension = st.selectbox("Hypertension", [0,1])
    avg_gluc = st.number_input("Average Glucose Level", 0.0, 400.0, 130.0)
    bmi_h    = st.number_input("BMI", 10.0, 60.0, 26.0)
    stroke_h = st.selectbox("History of Stroke (0=No,1=Yes)", [0,1])
    age_group= st.selectbox("Age Group", ["<30","30-45","46-60","60+"])
    gender_h = st.selectbox("Gender", ["Male","Female","Other"])
    married  = st.selectbox("Ever Married", ["No","Yes"])
    work_type= st.selectbox("Work Type", ["Govt_job","Never_worked","Private","Self-employed","Children"])
    residence= st.selectbox("Residence Type", ["Urban","Rural"])
    smoke_h  = st.selectbox("Smoking Status", ["Former","Never","Unknown","Smokes"])
    glucose_risk = st.selectbox("Glucose Risk", ["High","Normal"])
    bp_risk  = st.selectbox("BP Risk", [0,1])

    data = {
      'age': age,
      'hypertension': hypertension,
      'avg_glucose_level': avg_gluc,
      'bmi': bmi_h,
      'stroke': stroke_h,
      'age_group': age_group,
      'gender_male': 1 if gender_h=="Male" else 0,
      'gender_other':1 if gender_h=="Other" else 0,
      'ever_married_yes':1 if married=="Yes" else 0,
      'work_type_govt_job':       1 if work_type=="Govt_job" else 0,
      'work_type_never_worked':   1 if work_type=="Never_worked" else 0,
      'work_type_private':        1 if work_type=="Private" else 0,
      'work_type_self-employed':  1 if work_type=="Self-employed" else 0,
      'Residence_type_urban':     1 if residence=="Urban" else 0,
      'smoking_status_former':    1 if smoke_h=="Former" else 0,
      'smoking_status_never':     1 if smoke_h=="Never" else 0,
      'smoking_status_unknown':   1 if smoke_h=="Unknown" else 0,
      'glucose_risk':             glucose_risk,
      'bp_risk':                  bp_risk
    }
    df = pd.DataFrame([data])
    if st.button("Predict Heart Disease"):
        predict_and_show(heart_pipeline, df)
