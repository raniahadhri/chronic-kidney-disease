import streamlit as st
import joblib
import numpy as np


#model = joblib.load("kidney_risk_model.pkl")

st.title("Kidney Disease Risk Predictor")

# GitHub-hosted background image
bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://raw.githubusercontent.com/janedoe/kidney-app/main/images/rrr.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(bg_img, unsafe_allow_html=True)

# Input fields
bp = st.number_input("Blood Pressure")
age = st.number_input("Age", min_value=0, max_value=120)
bp = st.number_input("Blood Pressure")
sodium = st.number_input("Sodium Level")
# Add more fields...

if st.button("Predict"):
    input_data = np.array([[age, bp, sodium]])  # Adjust based on features
    #prediction = model.predict(input_data)[0]
    #st.success(f"🧾 Predicted Risk Category: {prediction}")



