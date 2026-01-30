import streamlit as st
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="HIRELYZER | Resume Analyzer Demo",
    layout="wide"
)

# -------------------------------
# Custom Dark CSS
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}

.metric-card {
    background: #161B22;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #30363D;
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #8B949E;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #00F5A0;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 8px;
    background: #21262D;
    margin: 4px 6px 4px 0;
    font-size: 13px;
    color: #E6EDF3;
}
.section-title {
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Fake Analysis Data
# -------------------------------
def demo_results():
    return {
        "ats_score": 78,
        "bias_score": 0.22,
        "experience": "Mid-Level",
        "skills": ["Python", "React", "SQL", "NLP"],
        "missing_skills": ["AWS", "Docker"],
        "job_matches": [
            "Frontend Developer",
            "Full Stack Engineer",
            "AI Engineer (Junior)"
        ]
    }

# -------------------------------
# UI – Header
# -------------------------------
st.markdown("## 🤖 HIRELYZER – AI Resume Analyzer")
st.caption("Dark-themed interactive demo | Portfolio Showcase")

st.divider()

# -------------------------------
# Fake Resume Upload
# -------------------------------
uploaded = st.file_uploader(
    "📄 Upload Resume (PDF)",
    type=["pdf"],
    help="Demo mode – no real file processing"
)

# -------------------------------
# Demo Flow
# -------------------------------
if uploaded:
    st.success("Resume uploaded successfully")

    st.markdown("### 🔍 Analyzing Resume")
    progress = st.progress(0)
    status = st.empty()

    steps = [
        "Parsing resume content...",
        "Extracting skills & keywords...",
        "Checking ATS compatibility...",
        "Detecting gender-biased language...",
        "Matching job roles..."
    ]

    for i, step in enumerate(steps):
        status.info(step)
        for p in range(20):
            time.sleep(0.03)
            progress.progress((i * 20) + p + 1)

    status.success("Analysis complete ✅")

    data = demo_results()

    st.divider()
    st.markdown("### 📊 Resume Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ATS Score</div>
            <div class="metric-value">{data['ats_score']}/100</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Bias Score</div>
            <div class="metric-value">{data['bias_score']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Experience Level</div>
            <div class="metric-value">{data['experience']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h4 class='section-title'>🧠 Detected Skills</h4>", unsafe_allow_html=True)
    for skill in data["skills"]:
        st.markdown(f"<span class='badge'>{skill}</span>", unsafe_allow_html=True)

    st.markdown("<h4 class='section-title'>⚠️ Missing Skills</h4>", unsafe_allow_html=True)
    for skill in data["missing_skills"]:
        st.markdown(f"<span class='badge'>{skill}</span>", unsafe_allow_html=True)

    st.markdown("<h4 class='section-title'>💼 Recommended Job Roles</h4>", unsafe_allow_html=True)
    for job in data["job_matches"]:
        st.markdown(f"✅ {job}")

else:
    st.info("Upload a resume to start the demo analysis ☝️")

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("© 2026 HIRELYZER | AI Resume Intelligence (Demo)")
