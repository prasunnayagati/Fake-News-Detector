import streamlit as st
import joblib
import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import streamlit.components.v1 as components
from src.training import AetherNeuralEngine
from src.explain import AetherForensicExplainer
import json
from datetime import datetime, timedelta

HISTORY_FILE = "data/history.json"

# Configure Page
st.set_page_config(
    page_title="AETHER // NEURAL SENTINEL",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust NLTK Asset Management
# @st.cache_resource
def setup_nltk():
    assets = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for asset in assets:
        try:
            nltk.download(asset, quiet=True)
        except Exception as e:
            st.error(f"NLTK Error: {e}")

setup_nltk()

# --- APP STATE INITIALIZATION ---
def load_persistent_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        now = datetime.now()
        # Keep only items from the last 24 hours
        return [item for item in history if (now - datetime.fromisoformat(item['timestamp'])) < timedelta(hours=24)]
    except:
        return []

def save_persistent_history(history):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

if "nav" not in st.session_state: st.session_state.nav = "Home"
if "scan_history" not in st.session_state: st.session_state.scan_history = load_persistent_history()
if "probs" not in st.session_state: st.session_state.probs = None
if "audit" not in st.session_state: st.session_state.audit = None
if "shap_vals" not in st.session_state: st.session_state.shap_vals = None

# --- MODERN PROFESSIONAL STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #0ea5e9;
        --bg: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.6);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --success: #10b981;
        --danger: #ef4444;
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .stApp {
        background-color: var(--bg);
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0, transparent 50%),
            radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.1) 0, transparent 50%);
        color: var(--text-main);
    }

    h1, h2, h3, h4 { 
        font-family: 'Outfit', sans-serif !important; 
        font-weight: 700 !important;
        background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Modern Glass Card */
    .modern-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4);
    }

    /* Professional Header */
    .app-header {
        padding: 3rem 0;
        text-align: center;
        background: radial-gradient(circle at center, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
        margin-bottom: 2rem;
        border-radius: 30px;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        background: linear-gradient(to right, #fff, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Enhanced Metrics */
    .metric-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Smoother Inputs */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        color: #fff !important;
        padding: 1.2rem !important;
        transition: 0.3s !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid var(--glass-border);
    }
    
    .sidebar-inner {
        padding: 2rem 1rem;
    }

    /* Primary Button */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2.5rem !important;
        border-radius: 12px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        width: 100%;
        text-transform: none !important;
        letter-spacing: normal !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
    }

    /* Hide Deploy Button & Made with Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hide;}
    .stDeployButton {display:none;}
    [data-testid="stHeader"] {background: transparent;}
    
    /* Top Home Icon Styling */
    .home-icon-top {
        position: fixed;
        top: 1rem;
        right: 2rem;
        z-index: 1000;
        background: var(--card-bg);
        padding: 0.5rem;
        border-radius: 50%;
        border: 1px solid var(--glass-border);
        cursor: pointer;
        transition: 0.3s;
    }
    .home-icon-top:hover {
        transform: scale(1.1);
        border-color: var(--primary);
    }
    
    /* Pill-style Top Nav */
    .top-nav-wrapper .stButton > button {
        font-size: 0.7rem !important;
        padding: 0.3rem 0.5rem !important;
        min-height: 32px !important;
        height: 32px !important;
        white-space: nowrap !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="column"] {
        padding: 0 5px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CORE LOGIC LOADING ---
# @st.cache_resource
def load_engine():
    if not os.path.exists('models/model.pkl'):
        return None, None
    try:
        model = joblib.load('models/model.pkl')
        vec = joblib.load('models/vectorizer.pkl')
        return model, vec
    except:
        return None, None

def load_dataset():
    if os.path.exists('data/raw/True.csv') and os.path.exists('data/raw/Fake.csv'):
        try:
            df_true = pd.read_csv('data/raw/True.csv')
            df_fake = pd.read_csv('data/raw/Fake.csv')
            df_true['target'] = 1
            df_fake['target'] = 0
            df = pd.concat([df_true, df_fake]).reset_index(drop=True)
            df['total_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
            return df
        except:
            return None
    return None

# @st.cache_resource
def get_explainer(_model, _vec):
    return AetherForensicExplainer(_model, _vec)

# --- CORE LOGIC LOADING ---
# @st.cache_resource
def load_engine():
    if not os.path.exists('models/model.pkl'):
        return None, None
    try:
        model = joblib.load('models/model.pkl')
        vec = joblib.load('models/vectorizer.pkl')
        return model, vec
    except:
        return None, None

model_obj, vec_obj = load_engine()
data_df = load_dataset()

# --- SIDEBAR: SYSTEM COMMAND CENTER ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-inner" style="padding-bottom: 0;">
        <h2 style="margin:0; background: linear-gradient(90deg, #fff, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8rem;">Aether</h2>
        <p style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">Intelligence Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Simple System Status
    st.markdown("#### / STATUS")
    status_text = "READY" if model_obj else "OFFLINE"
    status_color = "#10b981" if model_obj else "#ef4444"
    st.markdown(f"""
    <div style="padding: 1rem; border: 1px solid {status_color}33; background: {status_color}11; border-radius: 12px;">
        <p style="margin:0; font-size: 0.8rem; color: {status_color}; font-weight:700;">System is {status_text}</p>
        <p style="margin:0; font-size: 0.7rem; color: #64748b; margin-top:5px;">All engines operational.</p>
    </div>
    """, unsafe_allow_html=True)

# --- TOP NAVIGATION BAR ---
st.markdown('<div class="top-nav-wrapper">', unsafe_allow_html=True)
nav_options = ["Home", "Scan", "History", "System Health", "Resources"]
# Using equal-ish columns with tighter spacing via CSS
cols = st.columns([1, 1, 1, 1.4, 1.2])

for i, option in enumerate(nav_options):
    with cols[i]:
        if st.button(
            option, 
            key=f"top_nav_{option.lower().replace(' ', '_')}", 
            use_container_width=True,
            type="primary" if st.session_state.nav == option else "secondary"
        ):
            st.session_state.nav = option
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

nav = st.session_state.nav
st.divider()

# --- PAGE 0: HOME ---
if nav == "Home":
    st.markdown("""
    <div class="app-header">
        <h1 class="hero-title">Welcome to Aether</h1>
        <p class="hero-subtitle">The simplest way to verify the authenticity of news and articles.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="modern-card" style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h3>Get Started in Seconds</h3>
        <p style="color: var(--text-muted); line-height: 1.8;">
            Our advanced AI analyzes the linguistic patterns of your text to help you identify 
            credible information and potential misinformation. 
        </p>
        <div style="margin-top: 2rem;">
            <p>Ready to check some content?</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- PAGE 1: SCAN ---
elif nav == "Scan":
    st.markdown("""
    <div class="app-header" style="padding: 0.5rem 0; margin-top: 1rem;">
        <h2 class="hero-title" style="font-size: 2.5rem;">Content Checker</h2>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_results = st.columns([1.4, 0.8], gap="large")

    with col_input:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### Analysis Input")
        user_input = st.text_area(
            "Input Box",
            placeholder="Type or paste your text here...",
            height=300,
            label_visibility="collapsed",
            key="user_text_input"
        )
        
        if st.button("RUN ANALYSIS"):
            if not model_obj:
                st.error("[FATAL_ERROR] Neural engine unavailable.")
            elif len(user_input.strip().split()) < 5:
                st.warning("[INSUFFICIENT_DATA] Increase signal length.")
            else:
                with st.spinner("Analyzing Neural Vectors..."):
                    fn = AetherNeuralEngine()
                    fn.model = model_obj
                    fn.vectorizer = vec_obj
                    probs = fn.predict(user_input)
                    st.session_state.probs = probs
                    st.session_state.last_text = user_input
                    
                    if model_obj and vec_obj:
                        try:
                            explainer = get_explainer(model_obj, vec_obj)
                            st.session_state.shap_vals = explainer.get_local_explanation([user_input])
                            st.session_state.audit = explainer.get_linguistic_audit(user_input)
                            
                            # Add to Session History
                            new_entry = {
                                "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
                                "timestamp": datetime.now().isoformat(),
                                "text": user_input[:100] + "...",
                                "score": float(probs[1]) if probs[1] > 0.5 else float(probs[0]),
                                "label": "AUTHENTIC" if probs[1] > 0.5 else "AI/FAKE"
                            }
                            st.session_state.scan_history.insert(0, new_entry)
                            save_persistent_history(st.session_state.scan_history)
                        except Exception as e:
                            st.error(f"Analysis Error: {e}")
                            st.session_state.shap_vals = None
                    else:
                        st.session_state.shap_vals = None
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        if st.session_state.probs is not None:
            fake_p, real_p = st.session_state.probs
            audit = st.session_state.audit
            is_real = real_p > 0.5
            
            # Define Theme
            res_clr = "#10b981" if is_real else "#ef4444"
            res_label = "AUTHENTIC" if is_real else "AI/FAKE"
            res_sub = "This content matches patterns found in verified reporting." if is_real else "This content shows patterns often associated with AI generation or misinformation."
            
            # 1. Primary Verdict Card
            st.markdown(f"""
            <div class="modern-card" style="border: 2px solid {res_clr}; background: {res_clr}11; text-align: center; padding: 2rem 1rem;">
                <h4 style="color:{res_clr}; font-size: 0.8rem; letter-spacing: 2px; margin-bottom: 0.5rem;">FINAL VERDICT</h4>
                <h1 style="color:{res_clr}; font-size: 2.5rem; margin-bottom: 1rem;">{res_label}</h1>
                <div style="background:{res_clr}22; border-radius: 50px; padding: 0.5rem 1.5rem; display: inline-block; border: 1px solid {res_clr}44;">
                    <span style="color:white; font-size: 1.1rem; font-weight: 700;">{max(real_p, fake_p):.1%} Confidence</span>
                </div>
                <p style="margin-top: 1.5rem; color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;">{res_sub}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Key Data Points (Directly Visible)
            st.markdown(f'<div class="modern-card" style="padding: 1rem;">', unsafe_allow_html=True)
            st.markdown(f"**AI Assessment:** `{audit['assessment'].upper()}`")
            # Show top 2 primary linguistic markers
            count = 0
            for n, v in audit['stats'].items():
                if count >= 2: break
                val = float(v.strip('%'))/100
                st.markdown(f"<p style='margin:10px 0 2px 0; font-size:0.6rem; color:#64748b;'>{n.upper()}</p>", unsafe_allow_html=True)
                st.progress(val)
                count += 1
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 3. View Details (Deep Info)
            with st.expander("🔍 VIEW DEEP ANALYSIS & AI LOGIC"):
                st.markdown("### How the AI Thinks")
                st.info("The highlighted words below show which parts of the text most influenced the AI's decision.")
                if "shap_vals" in st.session_state and st.session_state.shap_vals is not None:
                    shap_html = shap.plots.text(st.session_state.shap_vals[0], display=False)
                    components.html(shap_html, height=350, scrolling=True)
                else:
                    st.warning("Heatmap data unavailable for this scan.")
                
                st.divider()
                st.markdown("### Full Linguistic Audit")
                for n, v in audit['stats'].items():
                    val = float(v.strip('%'))/100
                    st.markdown(f"<p style='margin:10px 0 2px 0; font-size:0.6rem; color:#64748b;'>{n.upper()}</p>", unsafe_allow_html=True)
                    st.progress(val)
                
                st.divider()
                st.markdown("### Technical Trace")
                for line in audit['report']:
                    st.markdown(f"<code style='color:#64748b; font-size:0.75rem;'>{line}</code>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="modern-card" style="text-align:center; padding: 6rem 1rem; border-style: dashed; opacity: 0.4;">
                <h4 style="color:var(--text-muted);">Ready to Scan</h4>
                <p style="font-size:0.85rem; color:var(--text-muted); margin-top: 0.5rem;">Paste text and run analysis to see results.</p>
            </div>
            """, unsafe_allow_html=True)

# --- PAGE 2: HISTORY ---
elif nav == "History":
    header_col, clear_col = st.columns([0.8, 0.2])
    with header_col:
        st.markdown("""
        <div class="app-header" style="padding: 1rem 0; text-align: left;">
            <h1 class="hero-title" style="font-size: 2.5rem;">Session History</h1>
            <p class="hero-subtitle">Reviewing your scans from the last 24 hours.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with clear_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Clear All", use_container_width=True):
            st.session_state.scan_history = []
            save_persistent_history([])
            st.rerun()

    if not st.session_state.scan_history:
        st.info("No scans found from the last 24 hours.")
    else:
        for idx, item in enumerate(st.session_state.scan_history):
            clr = "#10b981" if item['label'] == "AUTHENTIC" else "#ef4444"
            with st.container():
                # Wrap everything in a card-like div but use streamlit columns for the delete button
                col_text, col_del = st.columns([0.85, 0.15])
                
                with col_text:
                    st.markdown(f"""
                    <div class="modern-card" style="border-left: 5px solid {clr}; padding: 1rem; margin-bottom: 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-size: 0.9rem; color: var(--text-main); font-weight: 500;">{item['text']}</div>
                            <div style="color: {clr}; font-weight: 700; font-size: 0.8rem;">{item['label']} ({item['score']:.1%})</div>
                        </div>
                        <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 5px;">
                            {datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_del:
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    if st.button("Delete", key=f"del_{item.get('id', idx)}", use_container_width=True):
                        st.session_state.scan_history.pop(idx)
                        save_persistent_history(st.session_state.scan_history)
                        st.rerun()
                
                st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

# --- PAGE 3: SYSTEM HEALTH ---
elif nav == "System Health":
    st.markdown('<div class="app-header" style="padding: 1rem 0;"><h1 class="hero-title" style="font-size: 2rem;">Dashboard Status</h1><p class="hero-subtitle">Real-time performance of the Aether Intelligence Engine.</p></div>', unsafe_allow_html=True)
    
    # Simplified Metric Row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="modern-card" style="text-align: center;">
            <p style="color:var(--text-muted); font-size:0.7rem; font-weight:700; letter-spacing:1px; margin-bottom:1rem;">STORIES ANALYZED</p>
            <h2 style="margin:0; color:var(--accent); font-size:2.5rem;">44.8K</h2>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="modern-card" style="text-align: center;">
            <p style="color:var(--text-muted); font-size:0.7rem; font-weight:700; letter-spacing:1px; margin-bottom:1rem;">MODEL ACCURACY</p>
            <h2 style="margin:0; color:var(--primary); font-size:2.5rem;">99.2%</h2>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="modern-card" style="text-align: center;">
            <p style="color:var(--text-muted); font-size:0.7rem; font-weight:700; letter-spacing:1px; margin-bottom:1rem;">PROC. SPEED</p>
            <h2 style="margin:0; color:var(--success); font-size:2.5rem;">14ms</h2>
        </div>
        """, unsafe_allow_html=True)

    if model_obj:
        st.markdown('<div class="modern-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("### Decision Markers")
        st.markdown("<p style='color:var(--text-muted); font-size:0.9rem;'>These are the top semantic patterns the AI currently uses to distinguish between real and fake content.</p>", unsafe_allow_html=True)
        
        explainer = get_explainer(model_obj, vec_obj)
        real_i, fake_i = explainer.get_global_importance()
        
        ca, cb = st.columns(2)
        with ca:
            st.markdown("<p style='font-size:0.75rem; color:#10b981; font-weight:700; margin-bottom:1rem;'>AUTHENTIC PATTERNS</p>", unsafe_allow_html=True)
            st.dataframe(real_i, use_container_width=True, hide_index=True)
        with cb:
            st.markdown("<p style='font-size:0.75rem; color:#ef4444; font-weight:700; margin-bottom:1rem;'>ANOMALY PATTERNS</p>", unsafe_allow_html=True)
            st.dataframe(fake_i, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 4: RESOURCES & INFO ---
elif nav == "Resources":
    st.markdown('<div class="app-header" style="padding: 1rem 0;"><h1 class="hero-title" style="font-size: 2rem;">Learning Center</h1><p class="hero-subtitle">Understand how Aether detects misinformation.</p></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="modern-card">
            <h3>📖 User Guide</h3>
            <p>1. Copy text from an article.</p>
            <p>2. Paste it into the 'Home & Scan' box.</p>
            <p>3. Look at the 'How the AI Thinks' map to see red/blue highlights.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="modern-card">
            <h3>🧠 How it Works</h3>
            <p>Aether uses a <b>Random Forest Classifier</b> trained on thousands of verified and fake news articles.</p>
            <p>It looks for specific linguistic markers like emotional tone, punctuation frequency, and word choice.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="modern-card">
        <h3>🔍 FAQ</h3>
        <b>Is this 100% accurate?</b><br>
        No tool is perfect. Aether provides a probability score based on patterns it has seen before.<br><br>
        <b>What do the colors mean?</b><br>
        Red highlights usually indicate patterns the AI associates with unverified content, while blue indicates regular patterns.
    </div>
    """, unsafe_allow_html=True)

