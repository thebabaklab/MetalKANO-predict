import streamlit as st
import torch

st.set_page_config(page_title="MetalKANO", page_icon="🧪")
st.title("🧪 MetalKANO Predictor")
st.write(f"✅ PyTorch {torch.__version__} loaded (CUDA: {torch.cuda.is_available()})")
