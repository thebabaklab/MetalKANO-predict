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

st.set_page_config(page_title="KANO Predictor", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
        padding: 2.5rem; border-radius: 8px; color: white; margin-bottom: 2rem;
    }
    .header-container h1 { margin: 0; font-size: 2.5rem; }
    .result-card { background: white; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #0d47a1; }
    .result-card.active { border-left-color: #2e7d32; }
    .result-card.inactive { border-left-color: #f57c00; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; color: #666; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-container"><h1>🧪 KANO Predictor</h1><p>Molecular activity prediction</p></div>', unsafe_allow_html=True)

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
    col1, col2 = st.columns([3, 1])
    
    with col1:
        smiles_input = st.text_input("Enter SMILES", placeholder="CC(=O)Oc1ccccc1C(=O)O")
    with col2:
        predict_btn = st.button("🔍 Predict", use_container_width=True)
    
    if predict_btn and smiles_input:
        with st.spinner("Processing..."):
            try:
                pred, valid_smiles = predict_smiles([smiles_input], checkpoint_path)
                
                if pred and len(pred) > 0:
                    pred_value = float(pred[0][0])
                    is_active = pred_value > 0.5
                    
                    st.markdown(f"""
                    <div class="result-card {'active' if is_active else 'inactive'}">
                        <div class="metric-label">SMILES</div>
                        <div style="font-family: monospace; margin-bottom: 1.5rem;">{smiles_input}</div>
                        <div class="metric-label">Activity Score</div>
                        <div class="metric-value">{pred_value:.2%}</div>
                        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #eee;">
                            <div style="font-size: 1.2rem; font-weight: 600;">{"✅ LIKELY ACTIVE" if is_active else "❌ LIKELY INACTIVE"}</div>
                            <div style="font-size: 0.9rem; color: #666;">Probability: {pred_value:.1%}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✓ Complete")
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

        if st.button("🔍 Predict All", use_container_width=True, key="batch"):
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
st.markdown("<div style='text-align: center; color: #666;'><p>KANO: Knowledge-Augmented Neural Network Optimizer</p></div>", unsafe_allow_html=True)
