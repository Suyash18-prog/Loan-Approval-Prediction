"""Streamlit frontend for the loan prediction API."""

from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Loan Approval Predictor")
st.caption("A machine-learning decision-support demo. It is not a lending decision or financial advice.")

api_url = os.getenv("LOAN_API_URL", "http://localhost:8000")
with st.form("loan_application"):
    st.subheader("Applicant profile")
    col1, col2, col3 = st.columns(3)
    gender = col1.selectbox("Gender", ["Male", "Female"])
    married = col2.selectbox("Married", ["Yes", "No"])
    dependents = col3.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = col1.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = col2.selectbox("Self-employed", ["No", "Yes"])
    property_area = col3.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

    st.subheader("Financial details")
    col1, col2, col3 = st.columns(3)
    applicant_income = col1.number_input(
        "Applicant income", min_value=1.0, max_value=10_000_000.0, value=5000.0
    )
    coapplicant_income = col2.number_input(
        "Coapplicant income", min_value=0.0, max_value=10_000_000.0, value=0.0
    )
    loan_amount = col3.number_input(
        "Loan amount (in thousands)", min_value=1.0, max_value=10_000.0, value=150.0
    )
    loan_term = col1.number_input(
        "Loan term (months)", min_value=1.0, max_value=600.0, value=360.0
    )
    credit_history = col2.selectbox("Credit history meets guidelines", [1.0, 0.0])
    submitted = st.form_submit_button("Predict approval", type="primary")

if submitted:
    payload = {
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history, "Property_Area": property_area,
    }
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"Prediction request rejected ({response.status_code}): {detail}")
            st.stop()
        result = response.json()
        if result["loan_status"] == "Approved":
            st.success(f"Approved — probability {result['approval_probability']:.1%}")
        else:
            st.error(f"Rejected — rejection probability {result['rejection_probability']:.1%}")
        st.progress(result["approval_probability"], text="Approval probability")
        st.info("The model captures patterns in historical data; feature importance is not causation.")
    except requests.RequestException as error:
        st.error(f"Could not reach the prediction API: {error}")

with st.expander("Model performance dashboard"):
    try:
        info = requests.get(f"{api_url}/model-info", timeout=5).json()
        st.write(f"Selected model: **{info['model_name']}**")
        metrics = info.get("metrics", {}).get(info["model_name"], {})
        st.json({key: value for key, value in metrics.items() if key != "classification_report"})
    except requests.RequestException:
        st.caption("Start the API and train a model to view performance metrics.")
