"""
KANO Predictor - Streamlit Web App
Calls the working predict.py script directly
"""
import streamlit as st
import pandas as pd
import subprocess
import sys
import tempfile
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG - edit these to add/remove models or change limits
# ============================================================================

CELL_LINE_MODELS = {
    "A549": "models/A549.pt",
    "A2780": "models/A2780.pt",
    "A2780cis": "models/A2780cis.pt",
    "MCF-7": "models/MCF7.pt",
}

RESULTS_DIR = "results/streamlit"

MAX_BATCH_ROWS = 100

st.set_page_config(page_title="MetalKANO Predictor", page_icon="🧪", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
    /* --- Global font & dark background (matches MB-Finder #797979 base + black overlay) --- */
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }

    /* Main app background - dark like MB-Finder */
    .stApp, .stApp > header {
        background: #797979 !important;
    }
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), #797979 !important;
    }

    /* --- All main-area text: gold for headings, white for body --- */
    .stApp h1, .stApp h2, .stApp h3 { color: #F1C969 !important; }
    .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp .stMarkdown, .stApp .stText { color: #ffffff !important; }

    /* --- Header banner (black gradient, gold accent - MB-Finder .btn-gradient-radial-black) --- */
    .header-container {
        background: linear-gradient(90deg, rgba(0,0,0,0.8) 0%, rgba(0,0,0,1) 50%, rgba(0,0,0,0.8) 100%);
        padding: 2.5rem 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0px 4px 4px 0px rgba(0,0,0,0.25);
    }
    .header-container h1 {
        margin: 0; font-size: 2.5rem; font-weight: 900; color: #F1C969 !important;
    }
    .header-container p {
        margin: 0.4rem 0 0; font-size: 1rem; color: rgba(255,255,255,0.7) !important; font-weight: 400;
    }

    /* --- Result card (dark card on dark bg) --- */
    .result-card {
        background: rgba(0,0,0,0.5); border-radius: 16px; overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0px 4px 4px 0px rgba(0,0,0,0.25);
    }
    .result-card-header {
        background: linear-gradient(76deg, rgba(0,0,0,1) 0%, rgba(66,64,64,1) 11%, rgba(0,0,0,1) 31%, rgba(0,0,0,1) 78%, rgba(66,64,64,1) 89%, rgba(0,0,0,1) 100%);
        padding: 0.75rem 1.5rem; color: #F1C969 !important; font-weight: 700; font-size: 0.95rem;
    }
    .result-card-body { padding: 1.5rem; }
    .result-card-body * { color: #ffffff !important; }
    .result-card.active .result-card-header { border-bottom: 3px solid #4CAF50; }
    .result-card.inactive .result-card-header { border-bottom: 3px solid #FF9800; }
    .metric-value { font-size: 2.2rem; font-weight: 900; color: #F1C969 !important; }
    .metric-label { font-size: 0.8rem; color: rgba(255,255,255,0.7) !important; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }

    /* --- Align button with text input --- */
    div[data-testid="stVerticalBlock"] div[data-testid="stColumns"] > div:nth-child(2) label {
        visibility: hidden;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stColumns"] > div:nth-child(2) .stButton button {
        margin-top: 0;
    }

    /* --- Predict button: MB-Finder "draw" style (dark-gray gradient, white text) --- */
    .stButton > button {
        background: linear-gradient(76deg, rgba(185,185,185,1) 0%, rgba(255,255,255,1) 50%, rgba(185,185,185,1) 100%) !important;
        color: #000 !important; border: none;
        border-radius: 8px; font-weight: 700; font-family: 'Roboto', sans-serif;
        box-shadow: 0px 4px 4px 0px rgba(0,0,0,0.25);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(76deg, rgba(210,210,210,1) 0%, rgba(255,255,255,1) 50%, rgba(210,210,210,1) 100%) !important;
        color: #000 !important;
    }
    .stButton > button p, .stButton > button span {
        color: #000 !important;
    }

    /* --- Download button (gold accent) --- */
    .stDownloadButton > button {
        background: #F1C969 !important; color: #000 !important; border: none; border-radius: 8px; font-weight: 700;
        box-shadow: 0px 4px 4px 0px rgba(0,0,0,0.25);
    }
    .stDownloadButton > button:hover { background: #ffd46b !important; }
    .stDownloadButton > button p, .stDownloadButton > button span { color: #000 !important; }

    /* --- Tab styling: active=gold, inactive=white (MB-Finder style) --- */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent; border-bottom: 1px solid rgba(255,255,255,0.2);
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-family: 'Roboto', sans-serif; font-weight: 500;
        color: rgba(255,255,255,0.8) !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #F1C969 !important; font-weight: 700;
        border-bottom-color: #F1C969 !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #F1C969 !important;
    }
    /* Tab panel background */
    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
    }

    /* --- Text input field (dark with white text) --- */
    .stTextInput input {
        background: rgba(0,0,0,0.4) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 8px;
    }
    .stTextInput input::placeholder { color: rgba(255,255,255,0.4) !important; }

    /* --- File uploader --- */
    .stFileUploader { color: #ffffff !important; }
    .stFileUploader label { color: #ffffff !important; }
    section[data-testid="stFileUploadDropzone"] {
        background: rgba(0,0,0,0.3) !important;
        border-color: rgba(255,255,255,0.3) !important;
    }

    /* --- Dataframe / table --- */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* --- Metric cards (dark bg) --- */
    [data-testid="stMetric"] {
        background: rgba(0,0,0,0.4); border-radius: 8px; padding: 0.75rem;
    }
    [data-testid="stMetric"] label { color: rgba(255,255,255,0.7) !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #F1C969 !important; }

    /* --- Sidebar (dark like MB-Finder) --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(76deg, rgba(0,0,0,1) 0%, rgba(66,64,64,1) 11%, rgba(0,0,0,1) 31%, rgba(0,0,0,1) 78%, rgba(66,64,64,1) 89%, rgba(0,0,0,1) 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #fff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label { color: rgba(255,255,255,0.7) !important; }
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background: rgba(255,255,255,0.1);
    }

    /* --- Alert boxes on dark bg --- */
    .stAlert { border-radius: 8px; }

    /* --- Divider --- */
    hr { border-color: rgba(255,255,255,0.15) !important; }

    /* --- Footer --- */
    .footer-text {
        text-align: center; color: rgba(255,255,255,0.5) !important; font-size: 0.85rem; padding: 0.5rem 0;
    }
    .footer-text a { color: #F1C969 !important; text-decoration: none; }

    /* --- Spinner --- */
    .stSpinner > div { color: #F1C969 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-container"><h1>MetalKANO</h1><p>Metal-based anticancer compound activity prediction</p></div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("⚙️ Configuration")
selected_cell_line = st.sidebar.selectbox("Cell Line", list(CELL_LINE_MODELS.keys()))
checkpoint_path = CELL_LINE_MODELS[selected_cell_line]
st.sidebar.caption(f"Model: `{checkpoint_path}`")

# ============================================================================
# PREDICTION WRAPPER - Calls predict.py directly
# ============================================================================

def predict_smiles(smiles_list, checkpoint_path):
    """
    Call predict.py as a subprocess to avoid environment issues.
    This ensures we use the exact same working code path as CLI.
    """
    
    # Create temporary CSV file with SMILES
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("smiles\n")
        for smiles in smiles_list:
            f.write(f"{smiles}\n")
        temp_csv = f.name
    
    try:
        # Run predict.py as subprocess in the same environment
        result = subprocess.run(
            [
                sys.executable, "predict.py",
                "--exp_name", RESULTS_DIR,
                "--exp_id", "pred",
                "--checkpoint_path", checkpoint_path,
                "--data_path", temp_csv
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Print CLI output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Check for errors
        if result.returncode != 0:
            raise RuntimeError(f"Prediction failed: {result.stderr}")

        # Read results CSV from the output directory
        output_file = f"./{RESULTS_DIR}/pred/predict.csv"
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not found: {output_file}")
        
        df_results = pd.read_csv(output_file)
        
        # Extract predictions and SMILES
        pred = [[row['pred_0']] for _, row in df_results.iterrows()]
        valid_smiles = df_results['smiles'].tolist()
        
        # DEBUG: Print to CLI
        print("\n" + "="*60)
        print("PREDICTIONS:")
        for i, (smiles, p) in enumerate(zip(valid_smiles, pred)):
            pred_val = float(p[0])
            is_active = "ACTIVE" if pred_val > 0.5 else "INACTIVE"
            print(f"{i+1}. {smiles}")
            print(f"   Score: {pred_val:.4f} ({pred_val:.2%}) - {is_active}")
        print("="*60 + "\n")
        
        return pred, valid_smiles
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

# ============================================================================
# TABS
# ============================================================================

tab1, tab2 = st.tabs(["Single SMILES", "Batch Upload"])

with tab1:
    st.header("Single Molecule Prediction")
    col1, col2 = st.columns([4, 1])

    with col1:
        smiles_input = st.text_input("Enter SMILES", placeholder="CC(=O)Oc1ccccc1C(=O)O")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Predict", use_container_width=True)
    
    if predict_btn and smiles_input:
        with st.spinner("Processing..."):
            try:
                pred, valid_smiles = predict_smiles([smiles_input], checkpoint_path)
                
                if pred and len(pred) > 0:
                    pred_value = float(pred[0][0])
                    is_active = pred_value > 0.5
                    
                    status_label = "LIKELY ACTIVE" if is_active else "LIKELY INACTIVE"
                    status_color = "#4CAF50" if is_active else "#FF9800"
                    card_class = "active" if is_active else "inactive"
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div class="result-card-header">Prediction Result</div>
                        <div class="result-card-body">
                            <div class="metric-label">SMILES</div>
                            <div style="font-family: 'Roboto Mono', monospace; margin-bottom: 1.5rem; color: #ffffff; font-size: 0.95rem;">{smiles_input}</div>
                            <div class="metric-label">Activity Score</div>
                            <div class="metric-value">{pred_value:.2%}</div>
                            <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.15);">
                                <span style="display: inline-block; background: {status_color}; color: #fff; padding: 0.35rem 1rem;
                                       border-radius: 6px; font-size: 0.95rem; font-weight: 700;">{status_label}</span>
                                <span style="margin-left: 0.75rem; font-size: 0.9rem; color: rgba(255,255,255,0.6);">Probability: {pred_value:.1%}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ {str(e)}")
                import traceback
                with st.expander("Debug Info"):
                    st.code(traceback.format_exc())

with tab2:
    st.header("Batch Prediction")
    st.markdown("Upload CSV with 'smiles' column")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.dataframe(df_input.head(10), use_container_width=True)

        total_rows = len(df_input)
        if total_rows > MAX_BATCH_ROWS:
            st.warning(f"⚠️ CSV has {total_rows} rows — only the first {MAX_BATCH_ROWS} will be used (limit: {MAX_BATCH_ROWS})")
            df_input = df_input.head(MAX_BATCH_ROWS)

        st.info(f"📁 {len(df_input)} molecules")

        if st.button("Predict All", use_container_width=True, key="batch"):
            if "smiles" not in df_input.columns:
                st.error("❌ CSV must have 'smiles' column")
            else:
                with st.spinner(f"Processing {len(df_input)} molecules..."):
                    try:
                        smiles_list = df_input["smiles"].tolist()
                        pred, valid_smiles = predict_smiles(smiles_list, checkpoint_path)
                        
                        df_results = pd.DataFrame({
                            "smiles": valid_smiles,
                            "activity_score": [float(p[0]) for p in pred],
                        })
                        df_results["prediction"] = df_results["activity_score"].apply(lambda x: "ACTIVE" if x > 0.5 else "INACTIVE")
                        
                        st.subheader("Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        csv_data = df_results.to_csv(index=False)
                        st.download_button("⬇️ Download CSV", csv_data, "predictions.csv", "text/csv", use_container_width=True)
                        
                        st.subheader("Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        active_count = (df_results["activity_score"] > 0.5).sum()
                        with col1: st.metric("Total", len(df_results))
                        with col2: st.metric("Active", active_count)
                        with col3: st.metric("Inactive", len(df_results) - active_count)
                        with col4: st.metric("Avg Score", f"{df_results['activity_score'].mean():.2%}")
                        
                        st.success("✓ Complete")
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
                        import traceback
                        with st.expander("Debug Info"):
                            st.code(traceback.format_exc())

st.divider()
st.markdown('<div class="footer-text">MetalKANO &mdash; Knowledge-Augmented Neural Network for metal-based anticancer compounds &middot; <a href="https://mb-finder.org" target="_blank">MB-Finder</a></div>', unsafe_allow_html=True)
