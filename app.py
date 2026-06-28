import streamlit as st
import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random

# Page Config
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# Load CSS
with open("style/style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# JavaScript for Smooth Scroll
st.markdown("""
<script>
    function scrollToForm() {
        const form = document.getElementById("form-section");
        if (form) {
            form.scrollIntoView({ behavior: 'smooth' });
        }
    }
</script>
""", unsafe_allow_html=True)
#faltu
st.markdown("""
<style>
body, .main, .block-container {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Title Banner
st.markdown("""
<h1 style='text-align: center; color: #800080;'>🧠 Brain Tumor Detection</h1>
""", unsafe_allow_html=True)


st.markdown("""
<div style="
    background: linear-gradient(to right, #e3f2fd, #fce4ec);
    border-left: 6px solid #4a148c;
    padding: 20px;
    margin-top: 10px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    animation: fadeIn 1s ease-in-out;">
    <div style="display: flex; align-items: center;">
        <img src="https://img.favpng.com/8/3/11/brain-icon-artificial-intelligence-icon-artificial-intelligence-icon-png-favpng-UuHTAG2Ga5v97KfbJ33aK40Jm.jpg" width="60" style="margin-right: 15px;" />
        <div>
            <p style="font-size: 18px; font-weight: 600; margin: 0; color: #4a148c;">Welcome to the Brain Tumor Detection System</p>
            <p style="font-size: 15px; margin: 5px 0 0; color: #333;">An AI-powered diagnostic assistant that analyzes MRI scans and patient data for early tumor identification and better treatment outcomes.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Features
st.markdown("""
<h3 class="fade-in">🧬 Advanced Detection Technology</h3>
<div class="flex-container">
    <div class="feature-box fade-in">
        <h4>🎯 High-Precision Scanning</h4>
        <p>Detects brain tumor patterns using trained deep learning models, reducing false alarms in image analysis.</p>
    </div>
    <div class="feature-box fade-in">
        <h4>⚡ Rapid Results</h4>
        <p>Generates predictions within seconds, saving time in early screening and review.</p>
    </div>
    <div class="feature-box fade-in">
        <h4>🤖 AI-Powered Analysis</h4>
        <p>Trained on real patient data to support reliable tumor classification and decision-making.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# How it works
st.markdown("""
<h3 class="fade-in">🔍 Working</h3>
<p class="section-description">Simple 3-step process for quick and accurate brain tumor detection</p>
<div class="flex-container">
    <div class="step-box fade-in">
        <img src="https://cdn-icons-png.flaticon.com/512/9068/9068747.png" class="icon-img">
        <h4>Enter Patient Details</h4>
        <p>Provide name, age, gender, medical history etc to contextualize the analysis and improve accuracy.</p>
    </div>
    <div class="step-box fade-in">
        <img src="https://cdn-icons-png.flaticon.com/512/10855/10855701.png" class="icon-img">
        <h4>Upload MRI Scan</h4>
        <p>Upload MRI images in JPG or PNG format. Our AI model analyzes the scan for tumor patterns.</p>
    </div>
    <div class="step-box fade-in">
        <img src="https://cdn-icons-png.flaticon.com/512/7541/7541906.png" class="icon-img">
        <h4>Get Results</h4>
        <p>Receive a prediction with a confidence graph to support medical decision-making.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Awareness Box
quotes = [
    "🧠 <strong>Early detection saves lives.</strong> Don’t ignore the signs.",
    "💡 <strong>Awareness is the first step toward prevention.</strong>",
    "🩺 <strong>Timely diagnosis improves treatment outcomes.</strong>",
    "👁️‍🗨️ <strong>Vision changes can be more than age — get scanned.</strong>",
    "⏱️ <strong>Even small symptoms can signal something big.</strong>"
]
st.markdown(f"""<div class='awareness-box'>{random.choice(quotes)}</div>""", unsafe_allow_html=True)


# Centered Analyze Now Button
if "go_to_form" not in st.session_state:
    st.session_state["go_to_form"] = False

if not st.session_state["go_to_form"]:
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyze Now", key="analyze_main", use_container_width=True):
                st.session_state["go_to_form"] = True


# Show Form
form_placeholder = st.empty()
if st.session_state["go_to_form"] or st.query_params.get("go") == "form":
    form_placeholder.empty()
    st.markdown("<h2 id='form-section'>🧠 Brain Tumor Detection System</h2>", unsafe_allow_html=True)
    st.markdown("Upload an MRI scan and fill in patient information **or try with demo data.**")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.info("🩺 **Did You Know?** Brain tumors can be detected early with regular scans and attention to symptoms.")

    use_demo = st.checkbox("🧪 Try with Demo Patient")
    mri_file = st.file_uploader("📤 Upload MRI Scan", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, value=73 if use_demo else None)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if use_demo else 0)
        tumor_type = st.selectbox("Tumor Type", ["Malignant", "Benign"], index=0 if use_demo else 0)
        tumor_size = st.number_input("Tumor Size (cm)", 0.0, 10.0, value=5.37 if use_demo else 0.0)
        location = st.selectbox("Tumor Location", ["Temporal", "Frontal", "Parietal", "Occipital"], index=0)
        histology = st.selectbox("Histology Type", ["Astrocytoma", "Glioblastoma", "Oligodendroglioma"], index=0)
    with col2:
        stage = st.selectbox("Stage", [1, 2, 3, 4], index=2)
        symptom1 = st.text_input("Symptom 1", "Vision Issues" if use_demo else "")
        symptom2 = st.text_input("Symptom 2", "Seizures" if use_demo else "")
        symptom3 = st.text_input("Symptom 3", "Seizures" if use_demo else "")
        radiation = st.selectbox("Radiation Treatment", ["Yes", "No"], index=1 if use_demo else 0)
        surgery = st.selectbox("Surgery Performed", ["Yes", "No"], index=0 if use_demo else 0)
        chemo = st.selectbox("Chemotherapy", ["Yes", "No"], index=0 if use_demo else 0)

    survival_rate = st.slider("Estimated Survival Rate (%)", 0, 100, value=51 if use_demo else 0)
    growth_rate = st.number_input("Tumor Growth Rate", 0.0, 2.0, value=0.11 if use_demo else 0.0)
    family_history = st.selectbox("Family History of Tumor", ["Yes", "No"], index=0 if use_demo else 0)
    mri_result = st.selectbox("MRI Result", ["Positive", "Negative"], index=0 if use_demo else 0)

    st.markdown("<hr>", unsafe_allow_html=True)

    cnn_model = load_model("brain_tumor.h5")
    tabular_model = joblib.load("Brain_tumor_tabular.pkl")

    if st.button("🔍 Predict"):
        if not mri_file:
            st.error("❗ Please upload an MRI scan.")
            st.stop()

        img = Image.open(mri_file).resize((128, 128)).convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 128, 128, 3)
        mri_pred = cnn_model.predict(img_array)[0][0]

        def encode_binary(val): return 1 if val in ["Yes", "Positive"] else 0
        def encode_gender(val): return 1 if val == "Male" else 0
        def encode_stage(val): return val
        def encode_location(val): return {"Temporal": 1, "Frontal": 2, "Parietal": 3, "Occipital": 4}.get(val, 0)
        def encode_histology(val): return {"Astrocytoma": 1, "Glioblastoma": 2, "Oligodendroglioma": 3}.get(val, 0)

        tabular_input = np.array([[age,
            encode_gender(gender),
            1 if tumor_type == "Malignant" else 0,
            tumor_size,
            encode_location(location),
            encode_histology(histology),
            encode_stage(stage),
            len(symptom1) > 0,
            len(symptom2) > 0,
            len(symptom3) > 0,
            encode_binary(radiation),
            encode_binary(surgery),
            encode_binary(chemo),
            survival_rate,
            growth_rate,
            encode_binary(family_history),
            encode_binary(mri_result)
        ]])

        tabular_pred = tabular_model.predict_proba(tabular_input)[0][1]
        final_score = (mri_pred + tabular_pred) / 2

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🔬 Prediction Result")

        if final_score > 0.5:
            st.error(f"🚨 **Brain Tumor Detected** — Confidence: {final_score * 100:.2f}%")
            st.markdown("> 💪 **You are stronger than you think. Science, support, and spirit can conquer anything.**")
        else:
            st.success(f"✅ **No Tumor Detected** — Confidence: {(100 - final_score * 100):.2f}%")
            st.markdown("😊 <strong>Tumor not detected. Thank you for using the system!</strong>", unsafe_allow_html=True)

        # Bar Chart
        st.subheader("📊 Prediction Confidence")
        labels = ['No Tumor', 'Tumor Detected']
        scores = [100 - final_score * 100, final_score * 100]

        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(labels, scores, color=['green', 'red'])
        ax.set_ylim([0, 100])
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Model Confidence")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig)
# 🌸 Causes, Precautions & Care Section

st.markdown("""
<div style="display: flex; flex-wrap: nowrap; justify-content: center; gap: 25px;
            background: linear-gradient(to right, #f3e5f5, #e1f5fe);
            padding: 30px; border-radius: 20px; box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);">

  <div style="background-color: white; border-radius: 16px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
              padding: 25px 30px; flex: 0 0 300px; text-align: left;">
    <h4 style="color: #4a148c; font-size: 20px;">🔍 Causes</h4>
    <ul style="font-size: 15px; color: #333; line-height: 1.8; padding-left: 20px;">
      <li>📌 Genetic mutations or hereditary factors</li>
      <li>📌 Prolonged exposure to radiation</li>
      <li>📌 Environmental toxins and chemicals</li>
      <li>📌 Immune system disorders</li>
    </ul>
  </div>

  <div style="background-color: white; border-radius: 16px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
              padding: 25px 30px; flex: 0 0 300px; text-align: left;">
    <h4 style="color: #4a148c; font-size: 20px;">🛡️ Precautions</h4>
    <ul style="font-size: 15px; color: #333; line-height: 1.8; padding-left: 20px;">
      <li>✔️ Get regular health checkups</li>
      <li>✔️ Avoid unnecessary radiation exposure</li>
      <li>✔️ Maintain a balanced diet rich in antioxidants</li>
      <li>✔️ Be alert to warning signs and symptoms</li>
    </ul>
  </div>

  <div style="background-color: white; border-radius: 16px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
              padding: 25px 30px; flex: 0 0 300px; text-align: left;">
    <h4 style="color: #4a148c; font-size: 20px;">💖 Care & Support</h4>
    <ul style="font-size: 15px; color: #333; line-height: 1.8; padding-left: 20px;">
      <li>🤝 Consult neurologists and oncologists early</li>
      <li>🧘 Stay emotionally and mentally strong</li>
      <li>👨‍👩‍👧 Lean on support groups and loved ones</li>
      <li>💊 Follow medical guidance and treatment plans</li>
    </ul>
  </div>

</div>
""", unsafe_allow_html=True)

# 📞 Contact Us - Footer with Social Icons

st.markdown("""
<hr style="margin-top: 50px; border: 1px solid #ccc;" />

<div style="text-align: center; padding: 20px 0; font-size: 15px; color: #555;">
    <p><strong>📩 Contact Us</strong></p>
    <p>Email: <span style="color: #4a148c;">brainhealthsupport@example.com</span></p>
    <p>Phone: <span style="color: #4a148c;">+91 12345 67890</span></p>

    

    <p ">© 2025 Brain Tumor Detection System. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

