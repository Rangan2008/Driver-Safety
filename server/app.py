import cv2
import streamlit as st

from safety_engine import SafetyEngine


st.set_page_config(
    page_title="Unified Driver Safety Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .panel {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .ok { border-left: 6px solid #2E7D32; }
    .warn { border-left: 6px solid #FF9800; }
    .danger { border-left: 6px solid #C62828; }
    .metric-title {
        font-size: 0.85rem;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine() -> SafetyEngine:
    return SafetyEngine(frame_skip=2, smoothing_window=12)


engine = get_engine()

with st.sidebar:
    st.header("Unified Safety Controls")
    run = st.toggle("Start Unified Monitoring", value=False)
    st.caption("All models run together in one coordinated pipeline.")

st.markdown("# Unified AI Driver Safety Dashboard")
st.markdown("Real-time fusion of drowsiness, distraction, and heart-risk intelligence.")

video_col, info_col = st.columns([7, 5], gap="large")

with video_col:
    frame_window = st.empty()

with info_col:
    status_box = st.empty()
    alert_box = st.empty()
    model_box = st.empty()
    heart_box = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera feed unavailable. Check webcam permissions and availability.")
            break

        frame = cv2.resize(frame, (960, 540))
        state = engine.process_step(frame)

        frame_window.image(cv2.cvtColor(state["frame"], cv2.COLOR_BGR2RGB), channels="RGB")

        status = state["overall_status"]
        panel_class = "ok" if status == "SAFE" else "warn" if status == "WARNING" else "danger"

        with status_box.container():
            st.markdown(
                f"""
                <div class="panel {panel_class}">
                    <div class="metric-title">Overall Status</div>
                    <div class="metric-value">{state['overall_status']}</div>
                    <div class="metric-title">Priority: {state['priority_level']} | SOS: {state['sos_triggered']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with alert_box.container():
            if state["alerts"]:
                st.error(" | ".join(state["alerts"]))
            else:
                st.success("No active anomalies. System is monitoring all channels.")

        with model_box.container():
            st.subheader("Model Outputs (Fusion Inputs)")
            c1, c2 = st.columns(2)
            c1.metric("Drowsiness", str(state["drowsiness"]))
            c2.metric("Distraction", str(state["distraction"]))
            c3, c4 = st.columns(2)
            c3.metric("Hands Off Wheel", str(state["hands_off_wheel"]))
            c4.metric("Heart Risk", str(state["heart_risk"]))

            st.caption("Confidence Scores")
            st.write(state["confidence"])

            st.caption("Distraction Details")
            st.code(state["metrics"]["distraction"]["details"], language="text")

            d = state["metrics"]["drowsiness"]
            st.caption(
                f"Drowsiness Metrics | EAR: {d['ear']:.3f}, MAR: {d['mar']:.3f}, "
                f"Nod: {d['nod_ratio']:.3f}, Yawns: {d['yawn_count']}"
            )

        with heart_box.container():
            st.subheader("Heart & ECG")
            hc = state["metrics"]["heart"]
            h1, h2 = st.columns(2)
            h1.metric("Heart Class", hc["class"])
            h2.metric("BPM", f"{hc['bpm']:.1f}")
            st.bar_chart(hc["probabilities"])
            st.line_chart(state["heart_chart"], height=180, width="stretch")

    cap.release()
else:
    st.info("Enable 'Start Unified Monitoring' to launch the integrated pipeline.")
