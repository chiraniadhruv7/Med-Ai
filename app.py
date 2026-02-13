import streamlit as st
import httpx
import base64
import json
import os
from io import BytesIO
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Clinical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(13, 148, 136, 0.2);
    }

    .main-header h1 {
        margin: 0 0 0.3rem 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .main-header p {
        margin: 0;
        opacity: 0.85;
        font-size: 0.95rem;
        font-weight: 300;
    }

    .badge-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }

    .badge {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.2);
    }

    .disclaimer-bar {
        background: linear-gradient(90deg, #fbbf24, #f59e0b);
        color: #1a1a1a;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 24px;
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .risk-high {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }

    .risk-moderate {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #1a1a1a;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
    }

    .risk-low {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }

    .finding-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        color: #e2e8f0;
    }

    .finding-card h4 {
        margin: 0 0 0.5rem 0;
        color: #38bdf8;
        font-size: 0.9rem;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #38bdf8;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #1e3a5f;
    }

    .confidence-bar-container {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Explainable Multimodal Clinical AI Assistant</h1>
        <p>CNN + MedGemma fusion for chest X-ray analysis with Grad-CAM explainability</p>
        <div class="badge-row">
            <span class="badge">🧠 DenseNet121</span>
            <span class="badge">🔬 MedGemma</span>
            <span class="badge">🍎 Apple Silicon</span>
            <span class="badge">📊 Grad-CAM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-bar">
        ⚠️ FOR RESEARCH AND EDUCATIONAL USE ONLY — Not a clinical diagnostic tool
    </div>
    """, unsafe_allow_html=True)


def render_risk_badge(risk_level: str):
    css_class = {
        "High": "risk-high",
        "Moderate": "risk-moderate",
        "Low": "risk-low",
    }.get(risk_level, "risk-low")
    st.markdown(f'<span class="risk-badge {css_class}">{risk_level} Risk</span>', unsafe_allow_html=True)


def render_confidence_bars(scores: dict):
    labels = {"cnn": "CNN Model", "llm": "LLM Reasoning", "fused": "Fused Confidence"}
    colors = {"cnn": "#38bdf8", "llm": "#a78bfa", "fused": "#34d399"}
    for key in ["cnn", "llm", "fused"]:
        val = scores.get(key, 0)
        pct = int(val * 100)
        color = colors[key]
        st.markdown(f"""
        <div class="confidence-bar-container">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.85rem; color:#94a3b8;">{labels[key]}</span>
                <span style="font-size:0.85rem; font-weight:600; color:{color};">{pct}%</span>
            </div>
            <div style="background:#0f172a; border-radius:6px; height:8px; overflow:hidden;">
                <div style="background:{color}; width:{pct}%; height:100%; border-radius:6px;
                     transition: width 0.5s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_findings(visual_findings: dict):
    sorted_findings = sorted(visual_findings.items(), key=lambda x: x[1], reverse=True)
    top_findings = [(k, v) for k, v in sorted_findings if v > 0.1]
    if not top_findings:
        top_findings = sorted_findings[:3]

    for label, prob in top_findings:
        pct = int(prob * 100)
        bar_color = "#ef4444" if prob > 0.7 else "#f59e0b" if prob > 0.3 else "#10b981"
        st.markdown(f"""
        <div class="finding-card">
            <h4>{label}</h4>
            <div style="display:flex; align-items:center; gap:0.8rem;">
                <div style="flex:1; background:#0f172a; border-radius:6px; height:10px; overflow:hidden;">
                    <div style="background:{bar_color}; width:{pct}%; height:100%; border-radius:6px;"></div>
                </div>
                <span style="font-weight:600; color:{bar_color}; min-width:40px;">{pct}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def call_backend(image_bytes: bytes, symptoms: str) -> dict:
    try:
        with httpx.Client(timeout=120.0) as client:
            files = {"image": ("xray.png", image_bytes, "image/png")}
            data = {"symptoms": symptoms}
            response = client.post(f"{BACKEND_URL}/analyze", files=files, data=data)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        st.error("❌ Cannot connect to backend. Make sure FastAPI is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


def main():
    render_header()

    left_col, right_col = st.columns([1, 1.5], gap="large")

    with left_col:
        st.markdown('<div class="section-header">📤 Input</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Chest X-ray",
            type=["png", "jpg", "jpeg", "dcm"],
            help="Supported formats: PNG, JPG, JPEG",
        )

        symptoms = st.text_area(
            "Patient Symptoms",
            placeholder="e.g., Persistent cough for 5 days, fever 101°F, shortness of breath...",
            height=120,
        )

        analyze_btn = st.button("🔬 Analyze", use_container_width=True, type="primary")

        if uploaded_file:
            st.markdown('<div class="section-header">🖼️ X-ray Preview</div>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

    with right_col:
        st.markdown('<div class="section-header">📋 Analysis Results</div>', unsafe_allow_html=True)

        if analyze_btn and uploaded_file:
            with st.spinner("Running analysis pipeline..."):
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
                result = call_backend(image_bytes, symptoms)

            if result:
                st.session_state["result"] = result
                st.session_state["image_bytes"] = image_bytes

        if "result" in st.session_state:
            result = st.session_state["result"]

            risk_col, conf_col = st.columns(2)
            with risk_col:
                st.markdown("**Risk Level**")
                render_risk_badge(result.get("risk_level", "Low"))
            with conf_col:
                st.markdown("**Confidence**")
                fused = result.get("confidence_scores", {}).get("fused", 0)
                st.metric("Fused", f"{int(fused * 100)}%")

            st.markdown('<div class="section-header">📊 Confidence Scores</div>', unsafe_allow_html=True)
            render_confidence_bars(result.get("confidence_scores", {}))

            st.markdown('<div class="section-header">🔍 Visual Findings</div>', unsafe_allow_html=True)
            render_findings(result.get("visual_findings", {}))

            st.markdown('<div class="section-header">🧠 Clinical Reasoning</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="finding-card">
                {result.get("clinical_reasoning", "No reasoning available.")}
            </div>
            """, unsafe_allow_html=True)

            diff_dx = result.get("differential_diagnosis", [])
            if diff_dx:
                st.markdown('<div class="section-header">📝 Differential Diagnosis</div>', unsafe_allow_html=True)
                for i, dx in enumerate(diff_dx, 1):
                    st.markdown(f"**{i}.** {dx}")

            if result.get("risk_factors"):
                st.markdown('<div class="section-header">⚠️ Risk Factors</div>', unsafe_allow_html=True)
                for rf in result["risk_factors"]:
                    st.warning(rf)

            if result.get("contradictions"):
                st.markdown('<div class="section-header">🔄 Model Contradictions</div>', unsafe_allow_html=True)
                for c in result["contradictions"]:
                    st.info(f"**{c['pathology']}**: CNN={c['cnn_probability']:.2f}, LLM did not mention → confidence reduced")

            if result.get("heatmap_available") and result.get("heatmap_base64"):
                st.markdown('<div class="section-header">🔥 Grad-CAM Heatmap</div>', unsafe_allow_html=True)
                show_heatmap = st.toggle("Show heatmap overlay", value=True)
                if show_heatmap:
                    heatmap_bytes = base64.b64decode(result["heatmap_base64"])
                    heatmap_image = Image.open(BytesIO(heatmap_bytes))
                    st.image(heatmap_image, caption="Grad-CAM attention overlay", use_container_width=True)

            with st.expander("📦 Raw JSON Output"):
                display_result = {k: v for k, v in result.items() if k != "heatmap_base64"}
                st.json(display_result)

            st.markdown("""
            <div style="background:#1e293b; border:1px solid #475569; border-radius:8px; padding:1rem;
                        margin-top:1.5rem; text-align:center; color:#94a3b8; font-size:0.8rem;">
                ⚠️ This analysis is for research and educational purposes only.<br>
                It is not intended for clinical diagnosis. Always consult a qualified healthcare professional.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#64748b;">
                <p style="font-size:2.5rem;">🩻</p>
                <p style="font-size:1.1rem; font-weight:500;">Upload a chest X-ray and enter symptoms</p>
                <p style="font-size:0.9rem;">The AI will analyze the image and provide structured findings</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
