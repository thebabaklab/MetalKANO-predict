import streamlit as st
import torch
import rdkit

st.set_page_config(page_title="MetalKANO", page_icon="🧪")
st.title("🧪 MetalKANO Predictor")
st.write(f"✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
st.write(f"✅ RDKit {rdkit.__version__}")

st.write("✅ All dependencies loaded")
