import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import pandas as pd

# Load models
MODELS = {
    "Random Forest": "RandomForest.pkl",
    "Decision Tree": "DecisionTree.pkl",
    "SVM": "SVM_Model.pkl",
    "Logistic Regression": "LogisticR.pkl",
}

@st.cache_resource
def load_models():
    loaded_models = {}
    for name, path in MODELS.items():
        try:
            with open(path, "rb") as f:
                loaded_models[name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file {path} not found!")
    return loaded_models

loaded_models = load_models()

# Page config with custom favicon
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Responsive sidebar CSS with mobile support
st.markdown(
    """
    <style>
    /* Desktop sidebar styling */
    @media (min-width: 769px) {
        section[data-testid="stSidebar"] {
            width: 240px !important;
            min-width: 240px !important;
            max-width: 240px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 240px !important;
            min-width: 240px !important;
            max-width: 240px !important;
        }
        section[data-testid="stSidebar"] .main {
            width: 240px !important;
            max-width: 240px !important;
        }
        section[data-testid="stSidebar"] .block-container {
            width: 240px !important;
            max-width: 240px !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .css-1d391kg, .css-1lcbmhc, .css-17eq0hr,
        .css-1cypcdb, .css-1adrfps, .css-6qob1r, .css-1544g2n {
            width: 240px !important;
            max-width: 240px !important;
        }
    }
    
    /* Collapsed sidebar styling - makes it narrower when hidden */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0px !important;
        min-width: 0px !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] > div {
        width: 0px !important;
        min-width: 0px !important;
    }

    /* Tablet and mobile sidebar styling */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 240px !important;
            min-width: 240px !important;
            max-width: 240px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 240px !important;
            min-width: 240px !important;
            max-width: 240px !important;
        }
        section[data-testid="stSidebar"] .main {
            width: 240px !important;
            max-width: 240px !important;
        }
        section[data-testid="stSidebar"] .block-container {
            width: 240px !important;
            max-width: 240px !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .css-1d391kg, .css-1lcbmhc, .css-17eq0hr,
        .css-1cypcdb, .css-1adrfps, .css-6qob1r, .css-1544g2n {
            width: 240px !important;
            max-width: 240px !important;
        }
    }
    
    /* Small mobile devices */
    @media (max-width: 480px) {
        section[data-testid="stSidebar"] {
            width: 220px !important;
            min-width: 220px !important;
            max-width: 220px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 220px !important;
            min-width: 220px !important;
            max-width: 220px !important;
        }
        section[data-testid="stSidebar"] .main {
            width: 220px !important;
            max-width: 220px !important;
        }
        section[data-testid="stSidebar"] .block-container {
            width: 220px !important;
            max-width: 220px !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
    }
    
    /* Improve sidebar toggle button */
    button[kind="header"] {
        background: #000000 !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
        transition: border-color 0.2s ease !important;
    }

    button[kind="header"]:hover {
        border-color: #666666 !important;
    }
    
    button[kind="header"]:active {
        transform: scale(0.95) !important;
    }
    
    /* Fix sidebar close button positioning */
    .css-1rs6os {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Sidebar overlay for mobile */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"][aria-expanded="true"] {
            position: fixed !important;
            z-index: 999999 !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Add backdrop when sidebar is open on mobile */
        section[data-testid="stSidebar"][aria-expanded="true"]::before {
            content: '' !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            background: rgba(0, 0, 0, 0.5) !important;
            z-index: -1 !important;
        }
        
        /* Smooth sidebar animation */
        section[data-testid="stSidebar"] {
            transform: translateX(-100%) !important;
            transition: transform 0.3s ease !important;
        }
        
        section[data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0) !important;
        }
    }
    
    /* Enhanced toggle button for better accessibility */
    button[kind="header"] svg {
        transition: transform 0.3s ease !important;
    }
    
    button[kind="header"]:hover svg {
        transform: rotate(180deg) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS with modern animations and gradients
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .main {
            padding-top: 0rem;
        }
        
        .stApp {
            background: #000000;
            color: #FFFFFF;
        }
        
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: #000000;
            border-radius: 0px;
            margin: 1rem 0 2rem 0;
            border-bottom: 1px solid #333333;
        }

        .main-header h1 {
            font-size: 2.5rem !important;
            font-weight: 600 !important;
            margin: 0 !important;
            color: #FFFFFF !important;
            letter-spacing: -0.5px;
        }
        
        .main-header p {
            font-size: 1rem;
            margin-top: 0.5rem;
            opacity: 0.6;
            font-weight: 400;
        }
        
        .glass-card {
            background: #000000;
            border-radius: 8px;
            border: 1px solid #333333;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: border-color 0.2s ease;
        }

        .glass-card:hover {
            border-color: #666666;
        }
        
        .metric-card {
            background: #000000;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #333333;
            transition: border-color 0.2s ease;
        }

        .metric-card:hover {
            border-color: #666666;
        }

        .metric-number {
            font-size: 2rem;
            font-weight: 600;
            color: #FFFFFF;
            margin: 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 0.5rem;
        }
        
        .stButton > button {
            background: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid #FFFFFF !important;
            border-radius: 6px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            transition: opacity 0.2s ease !important;
        }

        .stButton > button:hover {
            opacity: 0.8 !important;
        }
        
        .result-positive {
            background: #000000 !important;
            color: #FFFFFF !important;
            border-left: 5px solid #FFFFFF !important;
        }

        .result-negative {
            background: #000000 !important;
            color: #FFFFFF !important;
            border-left: 5px solid #FFFFFF !important;
        }
        
        .result-box {
            border-radius: 8px !important;
            padding: 2rem !important;
            margin: 2rem 0 !important;
            text-align: center !important;
            border: 1px solid #333333 !important;
        }
        
        .stSelectbox > div > div {
            background: #000000 !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
        }

        .stSelectbox > div > div > div {
            background: #000000 !important;
            color: #FFFFFF !important;
        }

        .stNumberInput > div > div > input {
            background: #000000 !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
            color: #FFFFFF !important;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .health-tip {
            background: #000000;
            border-left: 2px solid #666666;
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            font-size: 0.875rem;
            border: 1px solid #333333;
        }

        .info-section {
            background: #000000;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border: 1px solid #333333;
        }
        
        .loading-spinner {
            border: 2px solid #333333;
            border-radius: 50%;
            border-top: 2px solid #FFFFFF;
            width: 40px;
            height: 40px;
            margin: 0 auto;
        }
        
        .sidebar .sidebar-content {
            background: #000000;
        }

        section[data-testid="stSidebar"] {
            background: #000000 !important;
            transition: border-color 0.2s ease !important;
            border-right: 1px solid #333333 !important;
        }
        
        /* Force consistent width for all sidebar elements */
        section[data-testid="stSidebar"] * {
            box-sizing: border-box !important;
        }
        
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .element-container,
        section[data-testid="stSidebar"] .stButton,
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        section[data-testid="stSidebar"] .stSelectbox > div,
        section[data-testid="stSidebar"] .stSelectbox > div > div {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Mobile-first responsive design */
        @media (max-width: 768px) {
            .main-header {
                padding: 1rem !important;
                margin: 0.5rem 0 1rem 0 !important;
            }
            
            .main-header h1 {
                font-size: 2.5rem !important;
            }
            
            .main-header p {
                font-size: 1rem !important;
            }
            
            .glass-card, .info-section {
                padding: 1rem !important;
                margin: 0.5rem 0 !important;
            }
            
            .metric-card {
                padding: 1rem !important;
            }
            
            .metric-number {
                font-size: 2rem !important;
            }
            
            .result-box {
                padding: 1rem !important;
                margin: 1rem 0 !important;
            }
            
            /* Make form responsive */
            .stForm {
                padding: 0.5rem !important;
            }
        }
        
        @media (max-width: 480px) {
            .main-header h1 {
                font-size: 2rem !important;
            }
            
            .main-header p {
                font-size: 0.9rem !important;
            }
            
            .stats-grid {
                grid-template-columns: 1fr !important;
                gap: 0.5rem !important;
            }
            
            .metric-number {
                font-size: 1.5rem !important;
            }
        }
        
        /* Improve sidebar on mobile */
        @media (max-width: 768px) {
            section[data-testid="stSidebar"] .metric-card {
                margin-bottom: 0.5rem !important;
            }
            
            section[data-testid="stSidebar"] .health-tip {
                font-size: 0.8rem !important;
                padding: 0.7rem !important;
                margin: 0.3rem 0 !important;
            }
        }
        
        h1, h2, h3, h4 {
            color: white !important;
        }
        
        .stMarkdown {
            color: white;
        }
        
        /* Fix sidebar scrolling on mobile */
        @media (max-width: 768px) {
            section[data-testid="stSidebar"] {
                overflow-y: auto !important;
                max-height: 100vh !important;
            }
        }
        
    </style>
    """,
    unsafe_allow_html=True,
)

# Add JavaScript for better sidebar functionality
st.markdown(
    """
    <script>
    // Improve sidebar functionality
    function initializeSidebar() {
        // Add click outside to close functionality on mobile
        document.addEventListener('click', function(event) {
            if (window.innerWidth <= 768) {
                const sidebar = document.querySelector('section[data-testid="stSidebar"]');
                const toggleButton = document.querySelector('button[kind="header"]');
                
                if (sidebar && toggleButton && 
                    !sidebar.contains(event.target) && 
                    !toggleButton.contains(event.target) &&
                    sidebar.getAttribute('aria-expanded') === 'true') {
                    // Close sidebar by clicking the toggle button
                    toggleButton.click();
                }
            }
        });
        
        // Add escape key to close sidebar
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                const sidebar = document.querySelector('section[data-testid="stSidebar"]');
                const toggleButton = document.querySelector('button[kind="header"]');
                
                if (sidebar && toggleButton && 
                    sidebar.getAttribute('aria-expanded') === 'true') {
                    toggleButton.click();
                }
            }
        });
        
        // Smooth transition for sidebar
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        }
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeSidebar);
    } else {
        initializeSidebar();
    }
    
    // Re-initialize on Streamlit rerun
    window.addEventListener('load', function() {
        setTimeout(initializeSidebar, 100);
    });
    </script>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown(
    """
    <div class="main-header">
        <h1>Heart Disease Prediction</h1>
        <p>AI-powered cardiovascular risk assessment</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar with enhanced content
with st.sidebar:
    st.markdown("### Control Panel")

    # Model selection with enhanced styling
    model_choice = st.selectbox(
        "Choose Model",
        list(MODELS.keys()),
        help="Different algorithms offer varying accuracy and interpretability"
    )

    st.markdown("---")

    # Statistics section
    st.markdown("### Statistics")

    st.markdown(
        """
        <div class="metric-card" style="margin-bottom: 1rem;">
            <div class="metric-number">655K</div>
            <div class="metric-label">Annual Deaths</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="metric-card" style="margin-bottom: 1rem;">
            <div class="metric-number">28%</div>
            <div class="metric-label">CVD Death Rate</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Analytics", "Information", "Model Details"])

with tab1:
    # Patient Information Form

    st.markdown("### Patient Information")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1, help="Valid range: 18-100 years")
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120, step=1, help="Valid range: 80-200 mmHg")
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1, help="Valid range: 100-400 mg/dL")
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

        with col2:
            restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LVH"])
            max_hr = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220, value=150, step=1, help="Valid range: 60-220 bpm")
            exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.2, value=1.0, step=0.1, help="Valid range: 0.0-6.2")
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
            st.markdown("")  # Spacing

        # Predict button
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            submitted = st.form_submit_button("Analyze", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature mapping
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}
    restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "LVH": 2}
    angina_map = {"No": 0, "Yes": 1}
    slope_map = {"Up": 0, "Flat": 1, "Down": 2}

    # Prediction Results
    if submitted:
        
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

        if model_choice in loaded_models:
            model = loaded_models[model_choice]
            prediction = model.predict(features)[0]

            # Check if model supports predict_proba
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                confidence = np.max(proba) * 100
                risk_score = proba[1] * 100  # Probability of having heart disease
            else:
                confidence = None
                risk_score = prediction * 100

            # Enhanced Result Display
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-box result-positive">
                        <h2>High Risk Detected</h2>
                        <p style="font-size:18px; margin:15px 0; opacity: 0.9;">
                            The {model_choice} model indicates this patient has a high risk of heart disease.
                        </p>
                        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                            <div>
                                <div style="font-size: 1.8rem; font-weight: 600; color: #FFFFFF;">
                                    {risk_score:.1f}%
                                </div>
                                <div style="opacity: 0.7; font-size: 0.875rem;">Risk Score</div>
                            </div>
                            {"<div><div style='font-size: 1.8rem; font-weight: 600; color: #FFFFFF;'>" + f"{confidence:.1f}%" + "</div><div style='opacity: 0.7; font-size: 0.875rem;'>Confidence</div></div>" if confidence else ""}
                        </div>
                        <p style="margin-top: 1rem; font-size: 13px; opacity: 0.6;">
                            Please consult with a healthcare professional for proper evaluation.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Risk factors analysis
                st.markdown("### Risk Factor Analysis")
                risk_factors = []
                if age > 65: risk_factors.append("Advanced age")
                if cholesterol > 240: risk_factors.append("High cholesterol")
                if resting_bp > 140: risk_factors.append("High blood pressure")
                if exercise_angina == "Yes": risk_factors.append("Exercise-induced angina")
                if cp in ["Typical Angina", "Atypical Angina"]: risk_factors.append("Chest pain symptoms")

                if risk_factors:
                    for factor in risk_factors:
                        st.error(f"{factor}")
                else:
                    st.info("No major risk factors identified in the input data")
                    
            else:
                st.markdown(
                    f"""
                    <div class="result-box result-negative">
                        <h2>Low Risk Assessment</h2>
                        <p style="font-size:18px; margin:15px 0; opacity: 0.9;">
                            The {model_choice} model indicates this patient has a low risk of heart disease.
                        </p>
                        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                            <div>
                                <div style="font-size: 1.8rem; font-weight: 600; color: #FFFFFF;">
                                    {100-risk_score:.1f}%
                                </div>
                                <div style="opacity: 0.7; font-size: 0.875rem;">Healthy Score</div>
                            </div>
                            {"<div><div style='font-size: 1.8rem; font-weight: 600; color: #FFFFFF;'>" + f"{confidence:.1f}%" + "</div><div style='opacity: 0.7; font-size: 0.875rem;'>Confidence</div></div>" if confidence else ""}
                        </div>
                        <p style="margin-top: 1rem; font-size: 13px; opacity: 0.6;">
                            Continue maintaining a healthy lifestyle and regular check-ups.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Patient Summary Table
            st.markdown("### Patient Summary")
            
            # Create a DataFrame for better presentation
            summary_data = {
                "Parameter": ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol", "Fasting BS", 
                            "Rest ECG", "Max HR", "Exercise Angina", "Oldpeak", "ST Slope"],
                "Value": [age, sex, cp, f"{resting_bp} mmHg", f"{cholesterol} mg/dL", 
                         "Yes" if fasting_bs == 1 else "No", restecg, f"{max_hr} bpm", 
                         exercise_angina, oldpeak, st_slope],
                "Status": ["Normal" if age < 65 else "Risk Factor",
                          "Normal", "Normal" if cp == "Asymptomatic" else "Risk Factor",
                          "Normal" if resting_bp < 140 else "Risk Factor",
                          "Normal" if cholesterol < 240 else "Risk Factor",
                          "Normal" if fasting_bs == 0 else "Risk Factor",
                          "Normal" if restecg == "Normal" else "Risk Factor",
                          "Normal" if max_hr > 100 else "Risk Factor",
                          "Normal" if exercise_angina == "No" else "Risk Factor",
                          "Normal" if oldpeak < 2.0 else "Risk Factor",
                          "Normal"]
            }
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
        else:
            st.error(f"Model {model_choice} not found!")

with tab2:
    st.markdown("### Analytics Dashboard")
    
    # Create sample visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution chart
        age_data = np.random.normal(55, 12, 1000)
        age_data = age_data[(age_data > 20) & (age_data < 90)]
        
        fig1 = px.histogram(
            x=age_data, 
            nbins=20,
            title="Age Distribution in Heart Disease Cases",
            color_discrete_sequence=['#667eea']
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Risk factors pie chart
        risk_factors = ['High Cholesterol', 'High BP', 'Smoking', 'Diabetes', 'Age', 'Other']
        values = [25, 20, 18, 15, 12, 10]
        
        fig2 = px.pie(
            values=values, 
            names=risk_factors, 
            title="Common Risk Factors Distribution"
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Model comparison chart
    st.markdown("### Model Performance Comparison")
    
    models = ['Random Forest', 'Decision Tree', 'SVM', 'Logistic Regression']
    accuracy = [0.89, 0.85, 0.87, 0.84]
    
    fig3 = px.bar(
        x=models,
        y=accuracy,
        title="Model Accuracy Comparison",
        color=accuracy,
        color_continuous_scale='Viridis'
    )
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### Information")

    # Information sections
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown('<div class="info-section">', unsafe_allow_html=True)
        st.markdown("#### What is Heart Disease?")
        st.markdown("Heart disease refers to several types of heart conditions, including coronary artery disease, heart rhythm problems, and heart defects. It remains one of the leading causes of death globally.")

        st.markdown("#### Common Symptoms")
        st.markdown("- Chest pain or discomfort")
        st.markdown("- Shortness of breath")
        st.markdown("- Pain in neck, jaw, or back")
        st.markdown("- Fatigue and weakness")
        st.markdown("- Irregular heartbeat")
        st.markdown('</div>', unsafe_allow_html=True)

    with info_col2:
        st.markdown('<div class="info-section">', unsafe_allow_html=True)
        st.markdown("#### Prevention Tips")
        st.markdown("- **Healthy Diet:** Low in saturated fat, rich in fruits and vegetables")
        st.markdown("- **Regular Exercise:** At least 150 minutes of moderate activity weekly")
        st.markdown("- **No Smoking:** Avoid tobacco and limit alcohol")
        st.markdown("- **Weight Management:** Maintain healthy BMI")
        st.markdown("- **Stress Management:** Practice relaxation techniques")
        st.markdown("- **Regular Check-ups:** Monitor blood pressure and cholesterol")

        st.markdown("#### When to See a Doctor")
        st.markdown("Consult immediately if experiencing chest pain, severe shortness of breath, fainting, or rapid/irregular heartbeat.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown("### Model Technical Details")

    model_info = {
        "Random Forest": {
            "Description": "Ensemble method using multiple decision trees",
            "Accuracy": "~89%",
            "Strengths": "High accuracy, handles missing values well",
            "Use Case": "Best overall performance"
        },
        "Decision Tree": {
            "Description": "Tree-like model of decisions and outcomes",
            "Accuracy": "~85%",
            "Strengths": "Highly interpretable, easy to understand",
            "Use Case": "When interpretability is crucial"
        },
        "SVM": {
            "Description": "Support Vector Machine for classification",
            "Accuracy": "~87%",
            "Strengths": "Works well with high-dimensional data",
            "Use Case": "Complex feature relationships"
        },
        "Logistic Regression": {
            "Description": "Statistical model for binary classification",
            "Accuracy": "~84%",
            "Strengths": "Fast, provides probability estimates",
            "Use Case": "Quick predictions with probabilities"
        }
    }
    
    for model, info in model_info.items():
        st.markdown(
            f"""
            <div class="glass-card">
                <h4>{model}</h4>
                <p><b>Description:</b> {info['Description']}</p>
                <p><b>Accuracy:</b> {info['Accuracy']}</p>
                <p><b>Strengths:</b> {info['Strengths']}</p>
                <p><b>Best Use Case:</b> {info['Use Case']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div class="glass-card" style="text-align: center; padding: 1.5rem;">
        <p style="font-size: 0.875rem; opacity: 0.7; margin: 0 0 0.5rem 0;">
            Created by Adarsh (E23CSEU1189), Arindam Singh (E23CSEU1171), Yashvardhan Dhaka (E23CSEU1192)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)