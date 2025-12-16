import streamlit as st
import joblib
import numpy as np
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("churn prediction app")
st.divider()
st.write("please enter the values and hit the predict button for grtting prediction.")
st.divider()
age = st.number_input("enter age",min_value=10,max_value= 100, value = 30)
tenure = st.number_input("enter tenure",min_value = 0, max_value=130,value=10)
monthlycharge = st.number_input("enter monthly charge", min_value=30,max_value=150)
gender = st.selectbox("enter the gender",["male","female"])
st.divider()
predictbutton = st.button("predict")
st.divider()

if predictbutton:
    gender_selected = 1 if gender == "female" else 0

    x = [age, gender_selected, tenure, monthlycharge]

    x1 =np.array(x)
    x_array = scaler.transform([x1])

    prediction = model.predict(x_array)[0]
    predicted = "Churn" if prediction == 1 else "not churn"
    st.write(f"predicted : {predicted}")

else:
    st.write("please enter the values and use predict button")
