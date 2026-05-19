import streamlit as st
import numpy as np
from PIL import Image

from detector import detect_damage
from scorer import quality_score, severity_summary

st.set_page_config(page_title="Road Quality Detector", page_icon="Road", layout="wide")

st.markdown("""
<style>
.score-card { background:#1a1d27; border:1px solid #2a2d3a; border-radius:12px; padding:1.5rem; text-align:center; margin-bottom:1rem; }
.stat-card { background:#1a1d27; border:1px solid #2a2d3a; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.75rem; }
.stat-label { font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px; }
.stat-value { font-size:1.6rem; font-weight:600; color:#fff; }
</style>
""", unsafe_allow_html=True)

st.title("Road Quality Detector")
st.caption("Upload a road image to detect damage and get a quality score.")

col_upload, col_results = st.columns([3, 2], gap="large")

with col_upload:
    uploaded_file = st.file_uploader("Upload a road image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)

    with st.spinner("Analysing road surface..."):
        damage_contours, annotated_img, stats = detect_damage(img_array)
        score, label, color = quality_score(stats["pothole_count"], stats["damage_area_pct"])

    with col_upload:
        tab1, tab2 = st.tabs(["Detected damage", "Original"])
        with tab1:
            st.image(annotated_img, use_container_width=True)
        with tab2:
            st.image(pil_img, use_container_width=True)
        st.markdown("""
        **Legend:**
        - Green box = Small damage
        - Orange box = Medium damage
        - Red box = Large damage
        """)

    with col_results:
        st.markdown(f"""
        <div class="score-card">
            <div style="color:#888;font-size:0.8rem;text-transform:uppercase;">Road Quality Score</div>
            <div style="font-size:3.5rem;font-weight:700;color:{color};">{score}</div>
            <div style="font-size:1.1rem;font-weight:600;color:{color};text-transform:uppercase;">{label}</div>
            <div style="font-size:0.8rem;color:#888;margin-top:0.4rem;">Score = min(100 - 8 x potholes, 100 - 1.5 x damage%)</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Damage regions detected</div>
            <div class="stat-value">{stats['pothole_count']}</div>
            <div style="display:flex;gap:8px;margin-top:0.5rem;">
                <span style="font-size:0.75rem;padding:3px 10px;border-radius:20px;background:#0a2e18;color:#00C850;">{stats['severity']['small']} small</span>
                <span style="font-size:0.75rem;padding:3px 10px;border-radius:20px;background:#2e1a00;color:#FF8C00;">{stats['severity']['medium']} medium</span>
                <span style="font-size:0.75rem;padding:3px 10px;border-radius:20px;background:#2e0a0a;color:#DC143C;">{stats['severity']['large']} large</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Damage area</div>
            <div class="stat-value">{stats['damage_area_pct']}%</div>
            <div style="font-size:0.8rem;color:#888;margin-top:4px;">of total image surface</div>
        </div>""", unsafe_allow_html=True)

        if label == "Good":
            rec = "Road is in good condition. Routine monitoring recommended."
            rec_color = "#1D9E75"
        elif label == "Moderate":
            rec = "Moderate damage detected. Schedule maintenance within 3 to 6 months."
            rec_color = "#BA7517"
        else:
            rec = "Severe damage. Immediate repair required."
            rec_color = "#A32D2D"

        st.markdown(f"""
        <div class="stat-card" style="border-color:{rec_color}40;">
            <div class="stat-label">Maintenance recommendation</div>
            <div style="font-size:0.9rem;color:#ccc;margin-top:6px;line-height:1.5;">{rec}</div>
        </div>""", unsafe_allow_html=True)

else:
    with col_results:
        st.markdown("""
        <div class="score-card" style="opacity:0.4;">
            <div style="color:#888;font-size:0.8rem;text-transform:uppercase;">Road Quality Score</div>
            <div style="font-size:3.5rem;font-weight:700;color:#555;">--</div>
            <div style="font-size:1.1rem;color:#555;">Awaiting image</div>
        </div>""", unsafe_allow_html=True)