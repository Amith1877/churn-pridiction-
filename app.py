import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")
st.divider()
st.write("Please enter the values and click Predict to get churn prediction.")
st.divider()

# Inputs
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter Tenure (months)", min_value=0, max_value=130, value=24)
monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150, value=60)
gender = st.selectbox("Select Gender", ["male", "female"])

st.divider()
predictbutton = st.button("Predict")
st.divider()

if predictbutton:
    # ✅ Correct gender encoding
    gender_selected = 1 if gender == "male" else 0

    # Feature order must match training
    x = [age, gender_selected, tenure, monthlycharge]
    x_array = scaler.transform(np.array([x]))

    # Use probability instead of raw predict
    churn_prob = model.predict_proba(x_array)[0][1]

    if churn_prob >= 0.85:
        result = "Churn"
    else:
        result = "Not Churn"

    st.success(f"Prediction: {result}")
    st.write(f"Churn Probability: {churn_prob:.2f}")

else:
    st.info("Please enter values and click Predict")



