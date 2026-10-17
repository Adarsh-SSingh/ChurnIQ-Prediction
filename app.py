import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
import time

# pyright: reportPossiblyUnboundVariable=false
warnings.filterwarnings("ignore")

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="ChurnIQ — AI Analytics",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# DEEP SPACE NEON CSS
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---------- BASE ---------- */
    * { font-family: 'Inter', sans-serif; box-sizing: border-box; }
    
    html, body, .stApp {
        background-color: #050814 !important;
        color: #e2e8f0 !important;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(79, 70, 229, 0.15), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(6, 182, 212, 0.15), transparent 25%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        padding: 2rem 3rem 4rem 3rem;
        max-width: 1400px;
        position: relative;
        z-index: 1;
    }
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem 1rem 3rem 1rem; }
    }

    #MainMenu, footer, header {visibility: hidden;}
    
    /* ---------- TYPOGRAPHY ---------- */
    .stApp p, .stApp span, .stApp label, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #e2e8f0;
    }
    h1, h2, h3, h4, .font-outfit { font-family: 'Outfit', sans-serif !important; }

    /* ---------- SCROLLBAR ---------- */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #050814; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

    .sb-logo {
        display: flex; align-items: center; gap: 12px;
        padding: 1rem 0.5rem 1.5rem 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 1rem;
    }
    .sb-logo-mark {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; font-weight: 800; color: #fff !important;
        box-shadow: 0 0 20px rgba(79, 70, 229, 0.4);
    }
    .sb-logo-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem; font-weight: 700; color: #f8fafc !important;
    }
    .sb-logo-sub {
        font-size: 0.7rem; color: #64748b !important;
        text-transform: uppercase; letter-spacing: 1px;
    }

    .sb-sec {
        font-size: 0.65rem; font-weight: 600; letter-spacing: 1px;
        color: #64748b !important; text-transform: uppercase;
        padding: 1.5rem 0.5rem 0.5rem 0.5rem;
    }
    .sb-nav {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 12px; border-radius: 8px;
        margin: 2px 0; font-size: 0.9rem; font-weight: 500;
        color: #94a3b8 !important; cursor: pointer;
        transition: all 0.2s;
    }
    .sb-nav:hover {
        background: rgba(255,255,255,0.05);
        color: #f8fafc !important;
    }
    .sb-nav.active {
        background: rgba(79, 70, 229, 0.15);
        color: #818cf8 !important;
        border-left: 3px solid #6366f1;
    }
    .sb-nav-badge {
        margin-left: auto; font-size: 0.65rem;
        background: rgba(255,255,255,0.1);
        color: #cbd5e1 !important;
        padding: 2px 8px; border-radius: 12px;
    }

    /* ---------- HEADER ---------- */
    .top-header {
        display: flex; align-items: flex-end; justify-content: space-between;
        margin-bottom: 2.5rem; flex-wrap: wrap; gap: 1rem;
    }
    .top-header-left h1 {
        font-size: 2rem; font-weight: 700; color: #f8fafc !important;
        margin: 0 0 0.25rem 0; letter-spacing: -0.02em;
    }
    .top-header-left p {
        font-size: 0.95rem; color: #94a3b8 !important; margin: 0;
    }
    .top-header-right {
        display: flex; gap: 0.75rem; align-items: center;
    }
    .model-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 8px 16px; border-radius: 20px;
        font-size: 0.8rem; color: #34d399 !important;
        font-weight: 600; letter-spacing: 0.5px;
    }
    .model-pill .dot {
        width: 8px; height: 8px;
        background: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 10px #34d399;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* ---------- GLASS CARDS ---------- */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    .glass-card:hover::before { opacity: 1; }

    .card-label {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
        color: #94a3b8 !important; margin-bottom: 0.75rem;
        display: flex; align-items: center; gap: 6px;
    }
    .card-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem; font-weight: 700; color: #f8fafc !important;
        line-height: 1.1; margin-bottom: 0.5rem; letter-spacing: -0.02em;
    }
    .card-sub {
        font-size: 0.85rem; color: #64748b !important;
        display: flex; align-items: center; gap: 4px;
    }
    .card-sub.up { color: #34d399 !important; }

    /* ---------- FORM / INPUTS ---------- */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 0.8rem !important; font-weight: 500 !important;
        color: #cbd5e1 !important; margin-bottom: 0.4rem !important;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        min-height: 44px !important;
        transition: all 0.2s !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
    }
    div[data-baseweb="select"] * { color: #f8fafc !important; font-size: 0.9rem !important; }
    
    .stNumberInput input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important; color: #f8fafc !important;
        min-height: 44px !important; font-size: 0.9rem !important;
    }
    .stNumberInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    .stSlider [role="slider"] {
        background-color: #6366f1 !important;
        border-color: #6366f1 !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.5) !important;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #6366f1) !important;
        color: #ffffff !important; border: none !important;
        border-radius: 8px !important; padding: 0.75rem 1.5rem !important;
        font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; font-size: 1rem !important;
        width: 100%; transition: all 0.2s;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(1px); }
    
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #e2e8f0 !important; border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: none !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem; background: rgba(15, 23, 42, 0.6);
        border-radius: 12px; padding: 6px; border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #94a3b8 !important;
        border-radius: 8px; padding: 8px 16px;
        font-weight: 500; font-size: 0.95rem; border: none; transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #e2e8f0 !important; }
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.1) !important; color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------- STEP INDICATOR ---------- */
    .step-indicator { display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; }
    .step-item { display: flex; align-items: center; gap: 8px; opacity: 0.4; }
    .step-item.active { opacity: 1; }
    .step-item.done { opacity: 1; }
    .step-circle {
        width: 28px; height: 28px; border-radius: 50%;
        background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.1);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: 600; color: #94a3b8 !important;
    }
    .step-item.active .step-circle {
        background: rgba(99, 102, 241, 0.2); border-color: #6366f1;
        color: #818cf8 !important; box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    }
    .step-item.done .step-circle {
        background: #6366f1; border-color: #6366f1; color: #ffffff !important;
    }
    .step-text { font-size: 0.85rem; font-weight: 500; color: #cbd5e1 !important; }
    .step-line { flex: 1; height: 2px; background: rgba(255,255,255,0.05); border-radius: 1px; }
    .step-line.done { background: #6366f1; }

    /* ---------- FORM SECTIONS ---------- */
    .form-section {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px; padding: 1.75rem; margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .form-section-title { font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 600; color: #f8fafc !important; margin-bottom: 0.25rem; }
    .form-section-desc { font-size: 0.85rem; color: #94a3b8 !important; margin-bottom: 1.25rem; }

    /* ---------- RESULT BANNER ---------- */
    .result-banner {
        padding: 2.5rem 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 2rem; border: 1px solid; backdrop-filter: blur(10px);
    }
    .banner-churn { background: rgba(220, 38, 38, 0.1); border-color: rgba(220, 38, 38, 0.2); box-shadow: 0 0 40px rgba(220, 38, 38, 0.1); }
    .banner-safe { background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.2); box-shadow: 0 0 40px rgba(16, 185, 129, 0.1); }
    .banner-title { font-family: 'Outfit', sans-serif; font-size: 1.8rem; font-weight: 700; color: #ffffff !important; margin: 0.5rem 0; }
    .banner-desc { font-size: 1rem; color: #cbd5e1 !important; margin: 0; }

    /* ---------- METRIC GRID ---------- */
    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
    @media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 480px) { .metric-grid { grid-template-columns: 1fr; } }
    
    .metric-card {
        background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px; padding: 1.25rem;
    }
    .metric-label { font-size: 0.75rem; color: #94a3b8 !important; font-weight: 500; margin-bottom: 0.5rem; }
    .metric-value { font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; color: #f8fafc !important; line-height: 1.2; }

    /* ---------- INSIGHT LIST ---------- */
    .insight-item {
        display: flex; align-items: center; gap: 12px;
        padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .insight-item:last-child { border-bottom: none; }
    .insight-title { font-size: 0.95rem; font-weight: 500; color: #e2e8f0 !important; }
    .insight-desc { font-size: 0.8rem; color: #64748b !important; }
    
    .impact-badge {
        font-size: 0.65rem; font-weight: 700; letter-spacing: 0.5px;
        padding: 4px 8px; border-radius: 6px; margin-left: auto;
    }
    .impact-high { background: rgba(239, 68, 68, 0.15); color: #fca5a5 !important; border: 1px solid rgba(239, 68, 68, 0.3); }
    .impact-med { background: rgba(245, 158, 11, 0.15); color: #fcd34d !important; border: 1px solid rgba(245, 158, 11, 0.3); }
    .impact-low { background: rgba(16, 185, 129, 0.15); color: #6ee7b7 !important; border: 1px solid rgba(16, 185, 129, 0.3); }
</style>
""", unsafe_allow_html=True)

# ================================================================
# LOAD MODEL
# ================================================================
@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "model" / "churniq_model.pkl")

try:
    model = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Model load failed: {e}")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "show_profile" not in st.session_state:
    st.session_state.show_profile = False

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-mark">🌌</div>
        <div>
            <div class="sb-logo-name">ChurnIQ</div>
            <div class="sb-logo-sub">AI Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Main Menu</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-nav active">📊 Overview</div>
    <div class="sb-nav">🔮 Prediction</div>
    <div class="sb-nav">👥 Customers</div>
    <div class="sb-nav">📈 Reports <span class="sb-nav-badge">3</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Settings</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-nav">⚙️ Preferences</div>
    <div class="sb-nav">🔒 Security</div>
    <div class="sb-nav">❓ Help</div>
    """, unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
st.markdown("""
<div class="top-header">
    <div class="top-header-left">
        <h1>System Dashboard</h1>
        <p>Real-time AI churn intelligence across 7,043 customers</p>
    </div>
    <div class="top-header-right">
        <div class="model-pill">
            <span class="dot"></span>
            SYSTEM ONLINE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# STATS STRIP
# ================================================================
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">🎯 Global Churn Risk</div>
        <div class="card-value">78%</div>
        <div class="card-sub">↑ 2.3% from last month</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">💰 Revenue Saved</div>
        <div class="card-value">$54,200</div>
        <div class="card-sub up">↑ $6,120 this quarter</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">⚡ Model Accuracy</div>
        <div class="card-value">80.5%</div>
        <div class="card-sub">ROC-AUC 0.842</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# TABS
# ================================================================
tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict", "📊 Insights", "ℹ️ About"])

# ================================================================
# TAB: PREDICT
# ================================================================
with tab_predict:
    
    step = st.session_state.wizard_step
    st.markdown(f"""
    <div class="step-indicator">
        <div class="step-item {'active' if step >= 1 else ''} {'done' if step > 1 else ''}">
            <div class="step-circle">{'✓' if step > 1 else '1'}</div>
            <div class="step-text">Customer Profile</div>
        </div>
        <div class="step-line {'done' if step > 1 else ''}"></div>
        <div class="step-item {'active' if step == 2 else ''}">
            <div class="step-circle">2</div>
            <div class="step-text">Prediction Result</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # STEP 1: FORM
    # ============================================================
    if st.session_state.wizard_step == 1:
        
        st.markdown("""
        <div class="form-section">
            <div class="form-section-title">Personal Information</div>
            <div class="form-section-desc">Basic demographics about the customer.</div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"], key="k_gender")
        with c2:
            senior = st.selectbox("Senior Citizen", [0, 1], 
                                  format_func=lambda x: "Yes" if x == 1 else "No", key="k_senior")
        with c3:
            partner = st.selectbox("Partner", ["Yes", "No"], key="k_partner")
        with c4:
            dependents = st.selectbox("Dependents", ["Yes", "No"], key="k_dep")
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="form-section">
            <div class="form-section-title">Account & Contract</div>
            <div class="form-section-desc">Tenure, contract type, and billing preferences.</div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12, key="k_tenure")
        with c2:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="k_contract")
        with c3:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="k_paper")
        
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            payment = st.selectbox("Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="k_pay")
        with c2:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5, key="k_monthly")
        with c3:
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0, step=10.0, key="k_total")
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="form-section">
            <div class="form-section-title">Services Subscribed</div>
            <div class="form-section-desc">Services and add-ons the customer uses.</div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            phone = st.selectbox("Phone Service", ["Yes", "No"], key="k_phone")
        with c2:
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="k_mult")
        with c3:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="k_int")
        
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="k_osec")
        with c2:
            online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="k_obkp")
        with c3:
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="k_dev")
        with c4:
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="k_tech")
        
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="k_tv")
        with c2:
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="k_mov")
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
        with c_btn2:
            predict_btn = st.button("🔮 Predict Churn", type="primary", use_container_width=True)
        
        if predict_btn and MODEL_LOADED:
            input_data = {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner,
                "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
                "MultipleLines": multiple, "InternetService": internet,
                "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
                "DeviceProtection": device, "TechSupport": tech,
                "StreamingTV": tv, "StreamingMovies": movies,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
            df_input = pd.DataFrame([input_data])
            
            with st.spinner("✨ Running AI prediction engine..."):
                time.sleep(0.8)
                prob_value = float(model.predict_proba(df_input)[0, 1])
                pred_value = int(prob_value >= 0.5)
            
            st.session_state.prediction = {
                "prob": prob_value, "pred": pred_value, "df": df_input,
                "gender": gender, "senior": senior, "partner": partner,
                "dependents": dependents, "tenure": tenure, "phone": phone,
                "multiple": multiple, "internet": internet,
                "online_sec": online_sec, "online_bkp": online_bkp,
                "device": device, "tech": tech, "tv": tv, "movies": movies,
                "contract": contract, "paperless": paperless,
                "payment": payment, "monthly": monthly, "total": total,
            }
            st.session_state.wizard_step = 2
            st.rerun()

    # ============================================================
    # STEP 2: RESULT
    # ============================================================
    elif st.session_state.wizard_step == 2 and st.session_state.prediction:
        p = st.session_state.prediction
        prob = p["prob"]
        pred = p["pred"]
        
        tenure = p["tenure"]
        contract = p["contract"]
        internet = p["internet"]
        payment = p["payment"]
        monthly = p["monthly"]
        online_sec = p["online_sec"]
        
        if prob < 0.35:
            risk, emoji = "Low", "🟢"
        elif prob < 0.65:
            risk, emoji = "Medium", "🟡"
        else:
            risk, emoji = "High", "🔴"
        
        if pred == 1:
            st.markdown(f"""
            <div class="result-banner banner-churn">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚠️</div>
                <div class="banner-title">High Risk: Customer Likely to Churn</div>
                <p class="banner-desc">Our neural engine predicts a {prob*100:.1f}% probability this customer will leave. Immediate retention action is recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-banner banner-safe">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">✅</div>
                <div class="banner-title">Safe: Customer Likely to Stay</div>
                <p class="banner-desc">Our neural engine predicts only a {prob*100:.1f}% probability of churn. This customer appears satisfied and stable.</p>
            </div>
            """, unsafe_allow_html=True)
        
        pred_text = "Churn" if pred == 1 else "No Churn"
        action_text = "Retain" if pred == 1 else "Monitor"
        
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value">{pred_text}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Probability</div>
                <div class="metric-value">{prob*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value">{emoji} {risk}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Action</div>
                <div class="metric-value">{action_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_g, col_i = st.columns([1, 1.2], gap="large")
        
        with col_g:
            st.markdown("""
            <div class="glass-card">
                <div class="card-label">📈 Probability Gauge</div>
            """, unsafe_allow_html=True)
            
            if prob < 0.35:
                gauge_color = "#10b981"
            elif prob < 0.65:
                gauge_color = "#f59e0b"
            else:
                gauge_color = "#ef4444"
            
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40, "color": "#f8fafc", "family": "Outfit"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#94a3b8", "size": 10}},
                    "bar": {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "rgba(15, 23, 42, 0.6)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35], "color": "rgba(16,185,129,0.1)"},
                        {"range": [35, 65], "color": "rgba(245,158,11,0.1)"},
                        {"range": [65, 100], "color": "rgba(239,68,68,0.1)"},
                    ],
                    "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.8, "value": 50}
                }
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc",
                height=260, margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            diff = prob * 100 - 26.5
            sign = "higher" if diff > 0 else "lower"
            st.markdown(f"""
            <div class="card-sub" style="margin-top: 1rem; text-align: center; justify-content: center;">
                {abs(diff):.1f}% {sign} than baseline (26.5%)
            </div>
            """, unsafe_allow_html=True)
            
        with col_i:
            st.markdown("""
            <div class="glass-card">
                <div class="card-label">🧠 Feature Impact</div>
                <div style="margin-top: 1rem;">
            """, unsafe_allow_html=True)
            
            insights = []
            if tenure <= 12:
                insights.append(("⏱️", "New customer (≤12 months)", "high", "47.4% churn in this group"))
            elif tenure >= 48:
                insights.append(("💎", "Loyal customer (≥48 months)", "low", "Only 9.5% churn"))
            else:
                insights.append(("⏳", f"Mid-tenure ({tenure} months)", "medium", "Moderate stability"))
            
            if contract == "Month-to-month":
                insights.append(("📄", "Month-to-month contract", "high", "42.7% churn rate"))
            elif contract == "One year":
                insights.append(("📄", "1-year contract", "low", "Only 11.3% churn"))
            else:
                insights.append(("📄", "2-year contract", "low", "Safest — 2.8% churn"))
            
            if internet == "Fiber optic":
                insights.append(("🌐", "Fiber optic internet", "high", "41.9% churn"))
            elif internet == "DSL":
                insights.append(("🌐", "DSL internet", "medium", "19% churn"))
            else:
                insights.append(("🌐", "No internet service", "low", "7.4% churn"))
            
            if payment == "Electronic check":
                insights.append(("💳", "Electronic check payment", "high", "45.3% churn"))
            elif "automatic" in payment:
                insights.append(("💳", "Auto-payment setup", "low", "~16% churn"))
            
            if monthly > 80:
                insights.append(("💰", f"High monthly charges (${monthly:.0f})", "high", "Cost sensitivity"))
            
            if online_sec == "No" and internet != "No":
                insights.append(("🔒", "No online security", "high", "41.8% churn"))
            
            for emoji_i, title, impact, sub in insights:
                if impact == "high":
                    badge_class = "impact-high"
                    badge_text = "HIGH"
                elif impact == "medium":
                    badge_class = "impact-med"
                    badge_text = "MED"
                else:
                    badge_class = "impact-low"
                    badge_text = "LOW"
                
                st.markdown(f"""
                <div class="insight-item">
                    <div style="font-size: 1.25rem;">{emoji_i}</div>
                    <div style="flex: 1;">
                        <div class="insight-title">{title}</div>
                        <div class="insight-desc">{sub}</div>
                    </div>
                    <div class="impact-badge {badge_class}">{badge_text}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            if st.button("↺ Predict Another", use_container_width=True, key="btn_reset"):
                st.session_state.prediction = None
                st.session_state.wizard_step = 1
                st.session_state.show_profile = False
                st.rerun()
        with c2:
            profile_df = p["df"].T.reset_index()
            profile_df.columns = ["Feature", "Value"]
            profile_df["Prediction"] = "Churn" if pred == 1 else "No Churn"
            profile_df["Probability"] = f"{prob*100:.2f}%"
            csv = profile_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name=f"churniq_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download"
            )
        with c3:
            if st.button("👤 View Profile", use_container_width=True, key="btn_profile"):
                st.session_state.show_profile = not st.session_state.show_profile
        
        if st.session_state.show_profile:
            with st.expander("Full Customer Profile", expanded=True):
                display_df = p["df"].T.reset_index()
                display_df.columns = ["Feature", "Value"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    else:
        st.session_state.wizard_step = 1
        st.rerun()

# ================================================================
# TAB: INSIGHTS
# ================================================================
with tab_insights:
    st.markdown("### System Insights")
    st.markdown("Understanding what drives customers to leave.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Churn by Contract Type")
        contract_df = pd.DataFrame({
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Churn %": [42.71, 11.27, 2.83]
        })
        fig = px.bar(contract_df, x="Contract", y="Churn %",
                     color="Churn %",
                     color_continuous_scale=["#1e293b", "#6366f1"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=False, height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig.update_xaxes(showgrid=False, color="#64748b")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#64748b")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Churn by Internet Service")
        internet_df = pd.DataFrame({
            "Internet": ["DSL", "Fiber optic", "No"],
            "Churn %": [18.96, 41.89, 7.40]
        })
        fig2 = px.bar(internet_df, x="Internet", y="Churn %",
                      color="Churn %",
                      color_continuous_scale=["#1e293b", "#06b6d4"])
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=False, height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig2.update_xaxes(showgrid=False, color="#64748b")
        fig2.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#64748b")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# TAB: ABOUT
# ================================================================
with tab_about:
    st.markdown("### About ChurnIQ")
    st.markdown("""
    **ChurnIQ** is an ML-powered customer churn prediction tool, built on the IBM Telco Customer Churn dataset.
    
    The model uses logistic regression on 19 customer features to estimate churn probability.
    
    **Dataset:** 7,043 customers • 21 features • 26.5% baseline churn rate  
    **Model:** Logistic Regression • 80.5% accuracy • 0.842 ROC-AUC
    """)

# ================================================================
# FOOTER
# ================================================================
st.markdown("""
<div style="text-align:center; color:#64748b; padding:3rem 0 1rem 0; font-size:0.8rem;">
    ChurnIQ — AI Analytics Suite
</div>
""", unsafe_allow_html=True)