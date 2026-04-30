import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import datetime

# 1. Page Configuration
st.set_page_config(page_title="DermAI Professional", page_icon="🛡️", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar & Session State
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title("🛡️ System Control")
    st.success("Model Status: Online")
    st.write(f"**Date:** {datetime.date.today()}")
    st.write("**Model:** Custom MobileNetV2")
    st.write("**Accuracy:** 93.25%")
    st.divider()
    if st.button("🗑️ Reset Session"):
        st.session_state.history = []
        st.rerun()

# 3. Model & Data
@st.cache_resource
def load_my_model():
    model_path = os.path.join('models', 'skin_cancer_model.h5')
    return tf.keras.models.load_model(model_path)

model = load_my_model()

data_info = {
    'Actinic Keratoses': {'type': 'Pre-Malignant', 'risk': 'High', 'desc': 'Rough, scaly patches caused by sun damage.'},
    'Basal Cell Carcinoma': {'type': 'Malignant', 'risk': 'Critical', 'desc': 'Slow-growing but invasive skin cancer.'},
    'Benign Keratosis': {'type': 'Benign', 'risk': 'Low', 'desc': 'Non-cancerous waxy skin growths.'},
    'Dermatofibroma': {'type': 'Benign', 'risk': 'Low', 'desc': 'Small, harmless fibrous bumps.'},
    'Melanoma': {'type': 'Malignant', 'risk': 'Critical', 'desc': 'Aggressive skin cancer requiring urgent biopsy.'},
    'Nevus': {'type': 'Benign', 'risk': 'Low', 'desc': 'Common, stable melanocytic mole.'},
    'Vascular Lesions': {'type': 'Benign', 'risk': 'Low', 'desc': 'Harmless growths involving blood vessels.'}
}
classes = list(data_info.keys())

# 4. Main Interface
st.title("🛡️ DermAI: Clinical Diagnostic Support System")
st.caption("Computer Science Engineering Project | Automated Skin Cancer Classification")

tab1, tab2, tab3 = st.tabs(["⚡ Live Diagnostic", "📜 History", "📘 About Model"])

with tab1:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Image Acquisition")
        file = st.file_uploader("Upload Lesion Image", type=["jpg", "png", "jpeg"])
        if file:
            img = Image.open(file).convert('RGB')
            st.image(img, use_container_width=True, caption="Source Image")

    with col2:
        st.subheader("Diagnostic Intelligence")
        if file:
            # Inference
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)[0]
            idx = np.argmax(preds)
            label = classes[idx]
            conf = preds[idx] * 100

            # Store in History
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            if not any(h['Timestamp'] == timestamp for h in st.session_state.history):
                st.session_state.history.append({
                    "Timestamp": timestamp, 
                    "Classification": label, 
                    "Confidence": f"{conf:.1f}%"
                })

            # Display Result
            st.metric("Primary Classification", label, f"{conf:.1f}% Match")
            
            # Probability Visualization
            st.write("**Feature Weights Distribution:**")
            chart_df = pd.DataFrame(preds, index=classes, columns=["Confidence"])
            st.bar_chart(chart_df)

            # Clinical Report Generation
            report_text = f"""
            DERMAI CLINICAL REPORT
            ----------------------
            Date: {timestamp}
            Classification: {label}
            Confidence: {conf:.2f}%
            Type: {data_info[label]['type']}
            Risk Level: {data_info[label]['risk']}
            ----------------------
            Note: For educational research use only.
            """
            
            st.download_button(
                label="📥 Download Clinical Report",
                data=report_text,
                file_name=f"Report_{label}_{datetime.date.today()}.txt",
                mime="text/plain"
            )

with tab2:
    st.subheader("Session Log")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
    else:
        st.write("No active logs.")

with tab3:
    st.subheader("Technical Architecture")
    st.write("""
    This system utilizes **Transfer Learning** on a **MobileNetV2** backbone. 
    MobileNetV2 is an efficient convolutional neural network architecture specifically designed for 
    high-performance computer vision tasks with lower computational overhead.
    """)
    st.image("https://miro.medium.com/max/1400/1*9T9SSh3oX3B-o0A7a_l1lQ.png", caption="MobileNetV2 Architecture Block", width=600)
    st.info("The model was trained on 5,000+ images from the HAM10000 dataset, achieving 93.25% validation accuracy.")

st.divider()
st.caption("© 2026 CSE Dept. Final Year Project | AI for Healthcare")