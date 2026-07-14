import streamlit as st
import numpy as np
from heart_risk_predictor import predict_heart_risk

st.title("Heart Disease Risk Detector")
age = st.number_input("Age",18,100)

sex = st.selectbox(
    "Sex",
    ["Female","Male"]
)

bp = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox(
    "Fasting Blood Sugar",
    [0,1]
)

restecg = st.number_input("Rest ECG")

thalach = st.number_input("Max Heart Rate")

exang = st.selectbox(
    "Exercise Induced Angina",
    [0,1]
)

oldpeak = st.number_input("Oldpeak")

slope = st.number_input("Slope")

ca = st.number_input("CA")

thal = st.number_input("Thal")

if st.button("Predict"):

    sex_num = 1 if sex == "Male" else 0

    prediction = predict_heart_risk(
        age,
        sex_num,
        bp,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    )

    if prediction == 0:
        st.success("Low Risk of Heart Disease")
    else:
        st.error("High Risk of Heart Disease")