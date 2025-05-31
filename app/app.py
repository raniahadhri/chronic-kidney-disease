import streamlit as st
import numpy as np
import joblib
import pandas as pd
import data_transformation as d

# Optional: Load your model
# model = joblib.load("kidney_risk_model.pkl")

# Setup initial session state
if "step" not in st.session_state:
    st.session_state.step = 1

st.title("🩺 Kidney Disease Risk Predictor")

# 🌄 Background image
bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://github.com/raniahadhri/chronic-kidney-disease/blob/main/images/background.png?raw=true");
    background-size: 1500px 870px;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""
# Style buttons globally (green)
st.markdown("""
<style>
div.stButton > button {
    background-color: #4CAF50;  /* Green */
    color: white;
    height: 3em;
    width: 8em;
    border-radius: 10px;
    border: none;
    font-size: 18px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #45a049;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown(bg_img, unsafe_allow_html=True)




# --- Initialize step state ---
if "step" not in st.session_state:
    st.session_state.step = 1

model = joblib.load("model/naive_bayes_model.pkl")
label_encoders = joblib.load('model/label_encoders.pkl')
scaler = joblib.load('model/scaler.pkl')

def main_step():
    st.header("Step 1: Enter Patient Information")

    # Patient Info
    with st.expander("Patient Information", expanded=True):
        sport = st.selectbox("Physical activity level", ["low","moderate" ,"high"])
        age = st.number_input("Age of the patient", min_value=0, max_value=120, value=50)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)
        smoking_status = st.selectbox("Smoking status", ["yes", "no"])
        family_history = st.selectbox("Family history of chronic kidney disease", ["yes", "no"])

    # Urine Test Results
    with st.expander("Urine Test Results"):
        rbc = st.selectbox("Red blood cells in urine", ["normal", "abnormal"])
        pus_cells = st.selectbox("Pus cells in urine", ["normal", "abnormal"])
        pus_cell_clumps = st.selectbox("Pus cell clumps in urine", ["not present", "present"])
        bacteria = st.selectbox("Bacteria in urine", ["not present", "present"])
        albumin = st.number_input("Albumin in urine", min_value=0, max_value=5, value=0)
        sugar = st.number_input("Sugar in urine", min_value=0, max_value=5, value=0)
        specific_gravity = st.number_input("Specific gravity of urine", min_value=1.000, max_value=1.030, format="%.3f", value=1.010)
        urine_protein_creatinine_ratio = st.number_input("Urine protein-to-creatinine ratio", min_value=0.0, max_value=100.0, value=0.1)
        urine_output = st.number_input("Urine output (ml/day)", min_value=0, max_value=5000, value=1500)
        urinary_sediment = st.selectbox("Urinary sediment microscopy results", ["normal", "abnormal"])

    # Blood Test Results###
    with st.expander("Blood Test Results"):
        random_blood_glucose = st.number_input("Random blood glucose level (mg/dl)", min_value=50, max_value=500, value=100)
        blood_urea = st.number_input("Blood urea (mg/dl)", min_value=0, max_value=300, value=20)
        blood_pressure = st.number_input("Blood pressure (mm/Hg)", min_value=0, max_value=300, value=20)
        serum_creatinine = st.number_input("Serum creatinine (mg/dl)", min_value=0.0, max_value=15.0, value=1.0)
        sodium = st.number_input("Sodium level (mEq/L)", min_value=100, max_value=200, value=140)
        potassium = st.number_input("Potassium level (mEq/L)", min_value=1, max_value=10, value=4)
        hemoglobin = st.number_input("Hemoglobin level (gms)", min_value=0.0, max_value=30.0, value=14.0)
        packed_cell_volume = st.number_input("Packed cell volume (%)", min_value=0, max_value=100, value=40)
        wbc_count = st.number_input("White blood cell count (cells/cumm)", min_value=0, max_value=100000, value=7000)
        rbc_count = st.number_input("Red blood cell count (millions/cumm)", min_value=0.0, max_value=10.0, value=4.5)
        egfr = st.number_input("Estimated Glomerular Filtration Rate (eGFR)", min_value=0.0, max_value=200.0, value=90.0)
        serum_albumin = st.number_input("Serum albumin level", min_value=0.0, max_value=10.0, value=4.0)
        cholesterol = st.number_input("Cholesterol level", min_value=0.0, max_value=500.0, value=180.0)
        pth = st.number_input("Parathyroid hormone (PTH) level", min_value=0.0, max_value=1000.0, value=50.0)
        serum_calcium = st.number_input("Serum calcium level", min_value=0.0, max_value=20.0, value=9.0)
        serum_phosphate = st.number_input("Serum phosphate level", min_value=0.0, max_value=20.0, value=3.5)
        cystatin_c = st.number_input("Cystatin C level", min_value=0.0, max_value=10.0, value=1.0)
        crp = st.number_input("C-reactive protein (CRP) level", min_value=0.0, max_value=100.0, value=3.0)
        il6 = st.number_input("Interleukin-6 (IL-6) level", min_value=0.0, max_value=100.0, value=2.0)

    # Medical History
    with st.expander("Medical History"):
        hypertension = st.selectbox("Hypertension (yes/no)", ["yes", "no"])
        diabetes = st.selectbox("Diabetes mellitus (yes/no)", ["yes", "no"])
        cad = st.selectbox("Coronary artery disease (yes/no)", ["yes", "no"])
        appetite = st.selectbox("Appetite (good/poor)", ["good", "poor"])
        pedal_edema = st.selectbox("Pedal edema (yes/no)", ["yes", "no"])
        anemia = st.selectbox("Anemia (yes/no)", ["yes", "no"])
        duration_diabetes = st.number_input("Duration of diabetes mellitus (years)", min_value=0, max_value=100, value=5)
        duration_hypertension = st.number_input("Duration of hypertension (years)", min_value=0, max_value=100, value=5)

    if st.button("Next"):
        st.session_state.input_data = {
        'Red blood cells in urine': rbc,
        'Pus cells in urine': pus_cells,
        'Pus cell clumps in urine': pus_cell_clumps,
        'Bacteria in urine': bacteria,
        'Hypertension (yes/no)': hypertension,
        'Diabetes mellitus (yes/no)': diabetes,
        'Coronary artery disease (yes/no)': cad,
        'Appetite (good/poor)': appetite,
        'Pedal edema (yes/no)': pedal_edema,
        'Anemia (yes/no)': anemia,
        'Family history of chronic kidney disease': family_history,
        'Smoking status': smoking_status,
        'Physical activity level': sport,
        'Urinary sediment microscopy results': urinary_sediment,
        'Age of the patient': age,
        'Blood pressure (mm/Hg)': blood_pressure,
        'Specific gravity of urine': specific_gravity,
        'Albumin in urine': albumin,
        'Sugar in urine': sugar,
        'Random blood glucose level (mg/dl)': random_blood_glucose,
        'Blood urea (mg/dl)': blood_urea,
        'Serum creatinine (mg/dl)': serum_creatinine,
        'Sodium level (mEq/L)': sodium,
        'Potassium level (mEq/L)': potassium,
        'Hemoglobin level (gms)': hemoglobin,
        'Packed cell volume (%)': packed_cell_volume,
        'White blood cell count (cells/cumm)': wbc_count,
        'Red blood cell count (millions/cumm)': rbc_count,
        'Estimated Glomerular Filtration Rate (eGFR)': egfr,
        'Urine protein-to-creatinine ratio': urine_protein_creatinine_ratio,
        'Urine output (ml/day)': urine_output,
        'Serum albumin level': serum_albumin,
        'Cholesterol level': cholesterol,
        'Parathyroid hormone (PTH) level': pth,
        'Serum calcium level': serum_calcium,
        'Serum phosphate level': serum_phosphate,
        'Body Mass Index (BMI)': bmi,
        'Duration of diabetes mellitus (years)': duration_diabetes,
        'Duration of hypertension (years)': duration_hypertension,
        'Cystatin C level': cystatin_c,
        'C-reactive protein (CRP) level': crp,
        'Interleukin-6 (IL-6) level': il6
        }
        st.session_state.step = 2
        st.rerun()

def prediction_step():
    st.title("Prediction")
    st.subheader("Step 2: Prediction Result")

    input_data = pd.DataFrame([st.session_state.input_data])

    df = d.data_transformation(input_data, label_encoders, scaler)

    prediction = model.predict(df)
    prediction_category = d.decode_prediction(prediction)
    st.success(f"✅ Predicted Risk: {prediction_category}")

    if st.button("Back"):
        st.session_state.step = 1
        st.rerun()

# --- App Navigation ---
if st.session_state.step == 1:
    main_step()
elif st.session_state.step == 2:
    prediction_step()