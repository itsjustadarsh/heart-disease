import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config with custom favicon - MUST BE FIRST
st.set_page_config(
    page_title="HEART SENSE - Cardiovascular Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    # Original model names with emojis for file loading
    original_models = {
        "🌲 Random Forest": "RandomForest.pkl",
        "🌳 Decision Tree": "DecisionTree.pkl",
        "🎯 SVM": "SVM_Model.pkl",
        "📊 Logistic Regression": "LogisticR.pkl",
    }

    # Load models and map to clean names (without emojis)
    name_mapping = {
        "🌲 Random Forest": "Random Forest",
        "🌳 Decision Tree": "Decision Tree",
        "🎯 SVM": "SVM",
        "📊 Logistic Regression": "Logistic Regression",
    }

    for emoji_name, path in original_models.items():
        try:
            with open(path, "rb") as f:
                clean_name = name_mapping[emoji_name]
                loaded_models[clean_name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file {path} not found!")
    return loaded_models

loaded_models = load_models()

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
        <h1>HEART SENSE</h1>
        <p>AI-Powered Cardiovascular Risk Assessment</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar with enhanced content
with st.sidebar:
    st.markdown("### Model Selection")

    # Model selection with enhanced styling
    model_choice = st.selectbox(
        "Choose Model",
        list(MODELS.keys()),
        help="Different algorithms offer varying accuracy and interpretability"
    )

    st.markdown("---")

    # Statistics section
    st.markdown("### CVD Statistics (India)")

    st.markdown(
        """
        <div class="metric-card" style="margin-bottom: 1rem;">
            <div class="metric-number">3M+</div>
            <div class="metric-label">Annual CVD Deaths</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="metric-card" style="margin-bottom: 1rem;">
            <div class="metric-number">27%</div>
            <div class="metric-label">Total Death Rate</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main content area
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
                        Based on your health parameters, our analysis indicates an elevated risk for cardiovascular disease.
                    </p>
                    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                        <div>
                            <div style="font-size: 1.8rem; font-weight: 600; color: #FFFFFF;">
                                {risk_score:.1f}%
                            </div>
                            <div style="opacity: 0.7; font-size: 0.875rem;">Risk Level</div>
                        </div>
                        {"<div><div style='font-size: 1.8rem; font-weight: 600; color: #FFFFFF;'>" + f"{confidence:.1f}%" + "</div><div style='opacity: 0.7; font-size: 0.875rem;'>Confidence</div></div>" if confidence else ""}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-box result-negative">
                    <h2>Low Risk Assessment</h2>
                    <p style="font-size:18px; margin:15px 0; opacity: 0.9;">
                        Your health parameters suggest a lower risk for cardiovascular disease at this time.
                    </p>
                    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                        <div>
                            <div style="font-size: 1.8rem; font-weight: 600; color: #FFFFFF;">
                                {100-risk_score:.1f}%
                            </div>
                            <div style="opacity: 0.7; font-size: 0.875rem;">Health Score</div>
                        </div>
                        {"<div><div style='font-size: 1.8rem; font-weight: 600; color: #FFFFFF;'>" + f"{confidence:.1f}%" + "</div><div style='opacity: 0.7; font-size: 0.875rem;'>Confidence</div></div>" if confidence else ""}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        # Identify risk factors for use in tabs
        risk_factors = []
        if age > 65: risk_factors.append(("Age over 65", "Age is a significant cardiovascular risk factor"))
        if cholesterol > 240: risk_factors.append(("Elevated Cholesterol", "Levels above 240 mg/dL increase heart disease risk"))
        if resting_bp > 140: risk_factors.append(("High Blood Pressure", "Readings above 140 mmHg indicate hypertension"))
        if exercise_angina == "Yes": risk_factors.append(("Exercise-Induced Chest Pain", "This requires immediate medical attention"))
        if cp in ["Typical Angina", "Atypical Angina"]: risk_factors.append(("Chest Pain Present", "Any chest discomfort should be evaluated by a doctor"))
        if max_hr < 100: risk_factors.append(("Low Maximum Heart Rate", "May indicate reduced cardiovascular fitness"))
        if fasting_bs == 1: risk_factors.append(("Elevated Blood Sugar", "Diabetes increases heart disease risk significantly"))
        if oldpeak > 2.0: risk_factors.append(("Significant ST Depression", "Indicates potential heart muscle stress"))

        # TABS SECTION - Only shown after prediction
        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Your Results Explained",
            "⚠️ Risk Factors",
            "💪 Action Plan",
            "🏥 When to Seek Care",
            "📋 Your Summary"
        ])

        with tab1:
            st.markdown("### Understanding Your Results")

            if prediction == 1:
                st.markdown("""
                <div class="info-section">
                    <h4>What does "High Risk" mean?</h4>
                    <p>Your health measurements show patterns associated with increased likelihood of heart disease. This doesn't mean you currently have heart disease, but that certain factors put you at elevated risk.</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="info-section">
                    <h4>How accurate is this assessment?</h4>
                    <p>Our analysis tool has been validated on hundreds of patient cases. However, this is a screening tool, not a diagnosis. Only a healthcare provider can diagnose heart disease through comprehensive testing.</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="info-section">
                    <h4>What happens next?</h4>
                    <p>We strongly recommend scheduling an appointment with a cardiologist or your family physician. Bring these results with you. They may order additional tests such as ECG, 2D Echo, TMT (Treadmill Test), or other cardiac investigations available at most Indian hospitals and diagnostic centers.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-section">
                    <h4>What does "Low Risk" mean?</h4>
                    <p>Your current health measurements show patterns associated with lower cardiovascular risk. This is positive news, but maintaining heart health requires ongoing effort.</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="info-section">
                    <h4>Should I still see a doctor?</h4>
                    <p>Yes. Regular health check-ups are essential even with low risk scores. This tool provides guidance but cannot replace professional medical evaluation. Annual preventive health check-ups are recommended for everyone, available at most hospitals and polyclinics across India.</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="info-section">
                    <h4>How do I maintain low risk?</h4>
                    <p>Continue healthy habits: balanced diet, regular exercise, stress management, and monitoring your blood pressure and cholesterol levels. Prevention is the best medicine.</p>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("### Your Risk Factors")

            if risk_factors:
                st.markdown(f"""
                <div class="info-section">
                    <p>We identified <strong>{len(risk_factors)}</strong> risk factor(s) in your profile. Understanding these can help you take targeted action.</p>
                </div>
                """, unsafe_allow_html=True)

                for i, (factor, explanation) in enumerate(risk_factors, 1):
                    st.markdown(f"""
                    <div class="health-tip">
                        <strong>{i}. {factor}</strong><br>
                        {explanation}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-section">
                    <p>No major risk factors were identified in your current measurements. This is excellent news! Continue maintaining your healthy lifestyle.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### General Heart Disease Risk Factors")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **Factors You Can Control:**
                - High blood pressure (hypertension)
                - High cholesterol (dyslipidemia)
                - Tobacco use (smoking, gutka, pan masala)
                - Diabetes or prediabetes
                - Obesity or being overweight
                - Physical inactivity/sedentary lifestyle
                - Unhealthy diet (high oil, salt, sugar)
                - Excessive alcohol consumption
                - High stress levels
                """)

            with col2:
                st.markdown("""
                **Factors You Cannot Control:**
                - Age (risk increases after 45 for men, 55 for women)
                - Sex (men at higher risk earlier in life)
                - Family history of heart disease
                - Previous heart attack or stroke
                - Genetic factors
                - South Asian ethnicity (higher CVD risk)
                """)

        with tab3:
            st.markdown("### Your Personalized Action Plan")

            if prediction == 1:
                st.markdown("""
                <div class="info-section" style="border-left: 3px solid #FF6B6B;">
                    <h4>🚨 Immediate Actions (This Week)</h4>
                </div>
                """, unsafe_allow_html=True)

                immediate_actions = [
                    "Schedule an appointment with a cardiologist or general physician",
                    "Print or save these results to share with your doctor",
                    "Do NOT start any new medications, supplements, or home remedies without medical guidance",
                    "If experiencing chest pain, shortness of breath, or dizziness, call 102/108 or rush to nearest emergency immediately"
                ]

                for action in immediate_actions:
                    st.markdown(f"- ✓ {action}")

                st.markdown("""
                <div class="info-section" style="border-left: 3px solid #FFA500; margin-top: 1.5rem;">
                    <h4>📅 Short-term Goals (Next 30 Days)</h4>
                </div>
                """, unsafe_allow_html=True)

                if cholesterol > 240:
                    st.markdown("- **Cholesterol Management**: Reduce ghee, butter, fried foods; increase fiber from dal, oats, vegetables")
                if resting_bp > 140:
                    st.markdown("- **Blood Pressure**: Monitor daily, reduce salt and pickles, avoid processed foods")
                if fasting_bs == 1:
                    st.markdown("- **Blood Sugar**: Get HbA1c tested, limit rice/wheat portions, avoid sweets and sugary drinks")
                if max_hr < 100:
                    st.markdown("- **Fitness**: Start gentle walking/yoga with doctor approval (30 min daily)")

                st.markdown("- **Diet**: Include more vegetables, fruits, whole grains (brown rice, millets), limit oil and salt")
                st.markdown("- **Exercise**: Aim for 150 minutes per week - walking, yoga, or light jogging (with doctor clearance)")
                st.markdown("- **Stress**: Practice pranayama, meditation, or yoga for stress management")

                st.markdown("""
                <div class="info-section" style="border-left: 3px solid #4ECDC4; margin-top: 1.5rem;">
                    <h4>🎯 Long-term Commitments</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("- Work with your cardiologist/physician to optimize all cardiovascular risk factors")
                st.markdown("- Regular monitoring: BP, lipid profile, blood sugar every 3-6 months")
                st.markdown("- Maintain healthy weight through balanced Indian diet")
                st.markdown("- Build sustainable heart-healthy habits (yoga, walking, proper diet)")
                st.markdown("- Consider cardiac rehabilitation program if recommended by doctor")

            else:
                st.markdown("""
                <div class="info-section" style="border-left: 3px solid #51CF66;">
                    <h4>✅ Maintain Your Heart Health</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                **Keep Doing What Works:**
                - Continue your current healthy habits
                - Regular physical activity (150 minutes/week) - walking, yoga, sports
                - Heart-healthy diet with vegetables, fruits, whole grains (millets, brown rice)
                - Maintain healthy weight (BMI 18.5-24.9)
                - Don't smoke or use tobacco in any form
                - Limit alcohol consumption
                - Manage stress through yoga, meditation, or pranayama
                - Get quality sleep (7-8 hours)
                """)

                st.markdown("""
                <div class="info-section" style="margin-top: 1.5rem;">
                    <h4>📅 Regular Monitoring</h4>
                    <p>Even with low risk, schedule annual check-ups including:</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                - Blood pressure check
                - Cholesterol panel
                - Blood glucose screening
                - BMI and weight assessment
                - Discussion of family history changes
                """)

        with tab4:
            st.markdown("### When to Seek Medical Care")

            st.markdown("""
            <div class="info-section" style="border-left: 4px solid #FF4444;">
                <h4>🚨 CALL 102/108 (Ambulance) IMMEDIATELY if you experience:</h4>
                <p style="opacity: 0.8; font-size: 0.9rem; margin-top: 0.5rem;">
                Emergency Numbers: 102 (National Ambulance) | 108 (Emergency Services) | Your nearest hospital emergency
                </p>
            </div>
            """, unsafe_allow_html=True)

            emergency_symptoms = [
                "Chest pain or discomfort (pressure, squeezing, fullness)",
                "Pain spreading to shoulders, arms, back, neck, jaw, or stomach",
                "Shortness of breath with or without chest discomfort",
                "Cold sweat, nausea, or lightheadedness",
                "Sudden severe headache with no known cause",
                "Sudden weakness or numbness of face, arm, or leg",
                "Sudden confusion or trouble speaking",
                "Loss of consciousness or fainting"
            ]

            for symptom in emergency_symptoms:
                st.markdown(f"🚨 **{symptom}**")

            st.markdown("---")

            st.markdown("""
            <div class="info-section" style="border-left: 4px solid #FFA500;">
                <h4>📞 Call Your Doctor Within 24 Hours if you have:</h4>
            </div>
            """, unsafe_allow_html=True)

            urgent_symptoms = [
                "New or worsening shortness of breath during normal activities",
                "Irregular heartbeat or heart palpitations",
                "Unexplained fatigue or weakness",
                "Swelling in legs, ankles, or feet",
                "Persistent indigestion or stomach discomfort",
                "Dizziness or feeling faint"
            ]

            for symptom in urgent_symptoms:
                st.markdown(f"⚠️ {symptom}")

            st.markdown("---")

            st.markdown("""
            <div class="info-section" style="border-left: 4px solid #4ECDC4;">
                <h4>📅 Schedule Routine Appointment for:</h4>
            </div>
            """, unsafe_allow_html=True)

            if prediction == 1:
                st.markdown("- **This week**: Discuss these risk assessment results with cardiologist")
                st.markdown("- **Follow-up testing**: ECG, 2D Echo, TMT, or other cardiac investigations")
                st.markdown("- **Medication review**: If currently taking any heart medications")
                st.markdown("- **Lifestyle counseling**: Diet, exercise, yoga, and risk reduction strategies")
            else:
                st.markdown("- **Annual health check-up**: Even with low risk, yearly preventive check-ups are important")
                st.markdown("- **Blood work**: Lipid profile, fasting glucose, HbA1c")
                st.markdown("- **Blood pressure**: Check every 6-12 months if normal")
                st.markdown("- **General wellness**: Consult about maintaining heart health and lifestyle")

        with tab5:
            st.markdown("### Your Health Summary")

            # Create a comprehensive summary
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Your Measurements")
                summary_data = {
                    "Parameter": ["Age", "Sex", "Blood Pressure", "Cholesterol", "Blood Sugar", "Max Heart Rate"],
                    "Value": [
                        f"{age} years",
                        sex,
                        f"{resting_bp} mmHg",
                        f"{cholesterol} mg/dL",
                        "Elevated" if fasting_bs == 1 else "Normal",
                        f"{max_hr} bpm"
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("#### Assessment Results")
                st.markdown(f"**Risk Level:** {'High Risk' if prediction == 1 else 'Low Risk'}")
                st.markdown(f"**Risk Score:** {risk_score:.1f}%")
                if confidence:
                    st.markdown(f"**Assessment Confidence:** {confidence:.1f}%")
                st.markdown(f"**Total Risk Factors:** {len(risk_factors)}")
                st.markdown(f"**Assessment Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")

            st.markdown("---")

            st.markdown("#### Recommended Next Steps")
            if prediction == 1:
                st.markdown("""
                1. ✓ Schedule appointment with cardiologist or primary care physician
                2. ✓ Share these results with your healthcare provider
                3. ✓ Begin implementing lifestyle modifications
                4. ✓ Monitor symptoms and seek emergency care if needed
                5. ✓ Follow up on recommended testing
                """)
            else:
                st.markdown("""
                1. ✓ Continue maintaining healthy lifestyle habits
                2. ✓ Schedule annual wellness check-up
                3. ✓ Monitor blood pressure and cholesterol regularly
                4. ✓ Stay physically active
                5. ✓ Keep up with preventive care
                """)

            st.markdown("---")

            st.markdown("""
            <div class="info-section">
                <h4>📝 Important Disclaimer</h4>
                <p>This risk assessment is a screening tool and not a medical diagnosis. Only qualified healthcare professionals can diagnose heart disease through comprehensive clinical evaluation and testing. If you have concerns about your heart health, please consult with your doctor.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            st.markdown("""
            <div class="info-section">
                <h4>📊 About the Data</h4>
                <p><strong>Dataset:</strong> Heart Disease Dataset</p>
                <p><strong>Sample Size:</strong> 918 patient records</p>
                <p><strong>Features:</strong> 11 clinical parameters including age, sex, chest pain type, blood pressure, cholesterol, ECG results, and exercise test data</p>
                <p style="opacity: 0.8; font-size: 0.9rem; margin-top: 0.5rem;">This model was trained on validated medical data to identify patterns associated with cardiovascular disease risk.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error(f"Model {model_choice} not found!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="glass-card" style="text-align: center; padding: 1.5rem;">
        <p style="font-size: 0.9rem; opacity: 0.7; margin: 0 0 0.75rem 0; font-weight: 500;">
            Adarsh (E23CSEU1189) • Arindam Singh (E23CSEU1171) • Yashvardhan Dhaka (E23CSEU1192)
        </p>
        <p style="font-size: 0.8rem; opacity: 0.5; margin: 0;">
            Heart Disease Dataset • 918 patient records • 11 clinical features • 4 ML models
        </p>
    </div>
    """,
    unsafe_allow_html=True
)