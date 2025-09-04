import streamlit as st
import pickle
import numpy as np

# Load models
MODELS = {
    "Random Forest": "RandomForest.pkl",
    "Decision Tree": "DecisionTree.pkl",
    "SVM": "SVM_Model.pkl",
    "Logistic Regression": "LogisticR.pkl",
}

loaded_models = {}
for name, path in MODELS.items():
    with open(path, "rb") as f:
        loaded_models[name] = pickle.load(f)

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# ------------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: black;
            color: white;
        }
        h1, h2, h3, h4 {
            color: white;
            text-align: center;
        }
        .stButton button {
            width: 100%;
            background: white;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.75rem;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: #222;
            color: white;
            border: 1px solid white;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }
        .result-box {
            border: 1px solid white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            text-align: center;
        }
        table {
            border-collapse: collapse;
            margin: auto;
        }
        th, td {
            border: 1px solid white;
            padding: 6px 12px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- HEADER -------------------
st.title("❤️ Heart Disease Predictor")
st.markdown("<br>", unsafe_allow_html=True)

# ------------------- MODEL CHOICE -------------------
model_choice = st.selectbox("Choose a Model", list(MODELS.keys()))
st.markdown("<br>", unsafe_allow_html=True)

# ------------------- PATIENT FORM -------------------
st.subheader("Enter Patient Information:")

with st.form("patient_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LVH"])
    max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict")

# ------------------- FEATURE MAPPING -------------------
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}
restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "LVH": 2}
angina_map = {"No": 0, "Yes": 1}
slope_map = {"Up": 0, "Flat": 1, "Down": 2}

features = np.array([[
    age,
    sex_map[sex],
    cp_map[cp],
    resting_bp,
    cholesterol,
    fasting_bs,
    restecg_map[restecg],
    max_hr,
    angina_map[exercise_angina],
    oldpeak,
    slope_map[st_slope]
]])

# ------------------- PREDICTION -------------------
if submitted:
    model = loaded_models[model_choice]
    prediction = model.predict(features)[0]

    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = np.max(proba) * 100
    else:
        proba, confidence = None, None

    # Result Box
    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-box">
                <h2>⚠️ {model_choice} Result</h2>
                <p style="font-size:18px; margin-top:10px;">
                    The model predicts this patient is <b>LIKELY to have Heart Disease</b>.
                </p>
                {"<p style='margin-top:10px;'>Confidence: <b>{:.2f}%</b></p>".format(confidence) if confidence else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-box">
                <h2>✅ {model_choice} Result</h2>
                <p style="font-size:18px; margin-top:10px;">
                    The model predicts this patient is <b>UNLIKELY to have Heart Disease</b>.
                </p>
                {"<p style='margin-top:10px;'>Confidence: <b>{:.2f}%</b></p>".format(confidence) if confidence else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------- SUMMARY TABLE -------------------
    st.markdown("<br><h3 style='text-align:center;'>📋 Patient Summary</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <table>
            <tr><th>Age</th><td>{age}</td></tr>
            <tr><th>Sex</th><td>{sex}</td></tr>
            <tr><th>Chest Pain</th><td>{cp}</td></tr>
            <tr><th>Resting BP</th><td>{resting_bp}</td></tr>
            <tr><th>Cholesterol</th><td>{cholesterol}</td></tr>
            <tr><th>Fasting BS</th><td>{fasting_bs}</td></tr>
            <tr><th>Rest ECG</th><td>{restecg}</td></tr>
            <tr><th>Max HR</th><td>{max_hr}</td></tr>
            <tr><th>Exercise Angina</th><td>{exercise_angina}</td></tr>
            <tr><th>Oldpeak</th><td>{oldpeak}</td></tr>
            <tr><th>ST Slope</th><td>{st_slope}</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

# ------------------- ABOUT -------------------
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.subheader("👨‍💻 About")
st.markdown(
    """
    <p style="text-align:center; color:white;">
    Created by <br><br>
    <b>Arindam Singh (E23CSEU1171)</b><br>
    <b>Adarsh (E23CSEU1189)</b><br>
    <b>Yashvardhan Dhaka (E23CSEU1192)</b>
    </p>
    """,
    unsafe_allow_html=True,
)
