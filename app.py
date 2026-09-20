import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Biomedical Engineering in Motion", layout="wide")

FONTS = ["Arial", "Helvetica", "Verdana", "Trebuchet MS", "Georgia", "Times New Roman", "Courier New", "Comic Sans MS"]
PRESENTATION_SECTIONS = {
    "Background Info": "Biomedical engineering uses math, engineering, biology, and medicine to design healthcare technologies such as prosthetic limbs.",
    "The Problem": "A recovering patient needs a prosthetic leg that mimics natural gait. Smooth velocity and acceleration reduce jarring joint forces.",
    "Modeling": "The presentation models prosthetic foot position during a step with a cubic position function over time.",
    "Parameters": "Velocity is the first derivative of position. Acceleration is the second derivative of position.",
    "Constraints": "The model works best on flat terrain, at a constant pace, with predictable data. Uneven ground, fatigue, and sensor errors can cause model failure.",
    "Solution + Application": "Sensors and adaptive algorithms can update the position function in real time as the patient's gait changes."
}
PROSTHETICS = {
    "Everyday walking foot": {"stiffness": 1.00, "damping": 1.00, "impact": 1.00, "label": "balanced comfort and stability"},
    "Microprocessor knee/ankle": {"stiffness": 1.10, "damping": 1.25, "impact": 0.88, "label": "adaptive damping and smoother transitions"},
    "Running blade": {"stiffness": 1.45, "damping": 0.82, "impact": 1.18, "label": "high energy return for sport"},
    "Hydraulic ankle": {"stiffness": 0.92, "damping": 1.35, "impact": 0.82, "label": "shock absorption for daily use"},
}
USE_CASES = {
    "Day-to-day normal activity": {"pace": 1.00, "impact": 1.00},
    "Rehabilitation / cautious gait": {"pace": 0.78, "impact": 0.72},
    "Sports / running": {"pace": 1.55, "impact": 1.45},
    "Stairs or uneven terrain": {"pace": 0.90, "impact": 1.35},
}
GENDER_STRIDE = {"Female": 0.415, "Male": 0.415, "Nonbinary / not specified": 0.415}


def css(font_family: str, base_size: int) -> str:
    return f"""
    <style>
      html, body, [class*="css"] {{ font-family: '{font_family}', sans-serif; font-size: {base_size}px; }}
      .metric-card {{ background:#e8f7fb; border:1px solid #b7e4ef; border-radius:16px; padding:14px; }}
      .big-title {{ font-size:{base_size+18}px; font-weight:800; color:#071d3a; margin-bottom:0; }}
      .subtitle {{ font-size:{base_size+2}px; color:#294056; }}
      .warning {{ background:#fff4d6; border-left:6px solid #e4573f; padding:12px; border-radius:8px; }}
      .ok {{ background:#e8f7ed; border-left:6px solid #0f9a9e; padding:12px; border-radius:8px; }}
    </style>
    """


def gait_position(t, step_length, step_time, lift_height, asymmetry, cadence_scale):
    # Smooth normalized cubic: p(0)=0, p(1)=L, v(0)=v(1)=0 before pace scaling.
    x = np.clip(t / step_time, 0, 1)
    forward = step_length * (3 * x**2 - 2 * x**3)
    lift = lift_height * np.sin(np.pi * x) * (1 + 0.12 * asymmetry * np.sin(2 * np.pi * x))
    # Presentation-inspired cubic overlay, normalized to avoid unrealistic meters.
    raw = 0.2 * t**3 - 1.5 * t**2 + 4 * t
    raw0 = 0
    raw1 = 0.2 * step_time**3 - 1.5 * step_time**2 + 4 * step_time
    overlay = 0 if abs(raw1 - raw0) < 1e-9 else step_length * (raw - raw0) / (raw1 - raw0)
    blend = 0.72 * forward + 0.28 * overlay
    return blend, lift


def derivatives(y, t):
    return np.gradient(y, t), np.gradient(np.gradient(y, t), t)


def person_segments(x_hip, y_hip, thigh, shank, prosthetic_side, phase, step_length, lift_height):
    hip = np.array([x_hip, y_hip])
    # simple walking angles
    natural_ang = math.radians(12 * math.sin(2 * math.pi * phase))
    prosth_ang = math.radians(-18 * math.sin(2 * math.pi * phase))
    if prosthetic_side == "Right":
        prosth_ang, natural_ang = natural_ang, prosth_ang
    knee_L = hip + np.array([-0.08 + thigh * math.sin(natural_ang), -thigh * math.cos(natural_ang)])
    foot_L = knee_L + np.array([shank * math.sin(natural_ang * 0.7), -shank * math.cos(natural_ang * 0.7)])
    knee_R = hip + np.array([0.08 + thigh * math.sin(prosth_ang), -thigh * math.cos(prosth_ang)])
    foot_R = knee_R + np.array([shank * math.sin(prosth_ang * 0.7), -shank * math.cos(prosth_ang * 0.7)])
    if prosthetic_side == "Left":
        prosth_foot = foot_L
    else:
        prosth_foot = foot_R
    return hip, knee_L, foot_L, knee_R, foot_R, prosth_foot


def make_motion_figure(df, params):
    L = params["step_length"]
    height = params["height_m"]
    thigh = 0.245 * height
    shank = 0.246 * height
    y_hip = thigh + shank + 0.05
    frames = []
    idxs = np.linspace(0, len(df)-1, 42).astype(int)
    max_force = max(df["impact_force_n"].max(), 1)
    for i in idxs:
        row = df.iloc[i]
        phase = (row.t / params["step_time"]) % 1
        hip, kL, fL, kR, fR, pf = person_segments(row.s - L/2, y_hip, thigh, shank, params["prosthetic_side"], phase, L, params["lift_height"])
        radius = 0.04 + 0.16 * row.impact_force_n / max_force
        frames.append(go.Frame(
            name=f"{row.t:.2f}s",
            data=[
                go.Scatter(x=[hip[0], kL[0], fL[0]], y=[hip[1], kL[1], fL[1]], mode="lines+markers", line=dict(width=6), name="biological leg"),
                go.Scatter(x=[hip[0], kR[0], fR[0]], y=[hip[1], kR[1], fR[1]], mode="lines+markers", line=dict(width=6), name="prosthetic leg"),
                go.Scatter(x=[pf[0]], y=[max(pf[1], 0.03)], mode="markers", marker=dict(size=28, symbol="circle"), name="prosthetic foot"),
                go.Scatter(x=[pf[0]], y=[0], mode="markers", marker=dict(size=radius*180, opacity=0.35), name="impact magnitude"),
            ]
        ))
    fig = go.Figure(frames=frames)
    fig.add_traces(frames[0].data)
    fig.update_layout(
        height=470, margin=dict(l=20, r=20, t=40, b=20), showlegend=True,
        xaxis=dict(range=[-0.7, L + 0.7], title="forward position (m)"),
        yaxis=dict(range=[-0.05, y_hip + 0.35], title="height (m)", scaleanchor="x", scaleratio=1),
        updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="Play motion", method="animate", args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}])])],
        sliders=[dict(steps=[dict(method="animate", args=[[fr.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], label=fr.name) for fr in frames])]
    )
    return fig


def make_calc_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.t, y=df.s, mode="lines", name="s(t) position (m)"))
    fig.add_trace(go.Scatter(x=df.t, y=df.v, mode="lines", name="v(t) velocity (m/s)"))
    fig.add_trace(go.Scatter(x=df.t, y=df.a, mode="lines", name="a(t) acceleration (m/s²)"))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="time (s)", yaxis_title="value")
    return fig


with st.sidebar:
    st.header("Display")
    font = st.selectbox("Font type", FONTS, index=0)
    font_size = st.slider("Font size", 12, 30, 18)
    st.markdown("---")
    section = st.selectbox("Presentation-based topic", list(PRESENTATION_SECTIONS.keys()))
    st.markdown("---")
    st.header("Patient and gait inputs")
    weight_kg = st.slider("Patient weight (kg)", 35.0, 160.0, 70.0, 0.5)
    height_cm = st.slider("Patient height (cm)", 120.0, 215.0, 170.0, 0.5)
    gender = st.selectbox("Gender / body proportion reference", list(GENDER_STRIDE.keys()))
    prosthetic_side = st.selectbox("Prosthetic side", ["Left", "Right"])
    gait_length = st.slider("Step / gait length (m)", 0.25, 1.25, round(GENDER_STRIDE[gender] * (height_cm/100), 2), 0.01)
    cadence = st.slider("Cadence (steps/min)", 40, 190, 95)
    prosthetic = st.selectbox("Type of prosthetic", list(PROSTHETICS.keys()))
    use_case = st.selectbox("Primary use", list(USE_CASES.keys()))
    terrain = st.selectbox("Terrain", ["Flat terrain", "Uneven ground", "Stairs", "Incline/decline"])
    friction = st.slider("Shoe-ground friction coefficient", 0.20, 1.20, 0.70, 0.01)
    fatigue = st.slider("Fatigue / gait variability", 0.0, 1.0, 0.15, 0.01)

st.markdown(css(font, font_size), unsafe_allow_html=True)
st.markdown('<p class="big-title">Biomedical Engineering in Motion: Prosthetic Gait Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Interactive Streamlit model for calculus, gait motion, and prosthetic impact.</p>', unsafe_allow_html=True)

st.info(PRESENTATION_SECTIONS[section])

height_m = height_cm / 100
step_time = max(0.32, 60 / cadence)
use = USE_CASES[use_case]
prost = PROSTHETICS[prosthetic]
terrain_factor = {"Flat terrain": 1.00, "Uneven ground": 1.22, "Stairs": 1.35, "Incline/decline": 1.15}[terrain]
lift_height = np.clip(0.045 * height_m * use["pace"] * (1 + 0.4 * fatigue), 0.04, 0.22)
asymmetry = fatigue + (0 if terrain == "Flat terrain" else 0.25)

t = np.linspace(0, step_time, 220)
s, z = gait_position(t, gait_length, step_time, lift_height, asymmetry, use["pace"])
v, a = derivatives(s, t)
jerk, _ = derivatives(a, t)
# impact model: weight + braking/landing contribution; educational approximation only.
body_weight_n = weight_kg * 9.81
vertical_velocity = derivatives(z, t)[0]
landing = np.clip(-vertical_velocity, 0, None)
impact_force = body_weight_n * use["impact"] * prost["impact"] * terrain_factor * (1 + 0.45 * fatigue) * (0.18 + 0.82 * landing / max(landing.max(), 1e-6))
impact_force *= max(0.35, 1.05 - 0.25 * friction)
comfort_index = 100 - np.clip((np.abs(a).max() * 8 + impact_force.max() / body_weight_n * 12 + np.abs(jerk).max() * 0.8 + fatigue * 20), 0, 100)

df = pd.DataFrame({"t": t, "s": s, "foot_height": z, "v": v, "a": a, "jerk": jerk, "impact_force_n": impact_force})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Step time", f"{step_time:.2f} s")
c2.metric("Peak velocity", f"{df.v.max():.2f} m/s")
c3.metric("Peak |acceleration|", f"{np.abs(df.a).max():.2f} m/s²")
c4.metric("Peak impact", f"{df.impact_force_n.max():.0f} N")

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Dynamic motion and prosthetic impact")
    params = {"step_length": gait_length, "height_m": height_m, "step_time": step_time, "lift_height": lift_height, "prosthetic_side": prosthetic_side}
    st.plotly_chart(make_motion_figure(df, params), use_container_width=True)
with right:
    st.subheader("Real-time calculus calculations")
    chosen_t = st.slider("Calculation time t (seconds)", 0.0, float(step_time), min(2.0, float(step_time)), 0.01)
    row = df.iloc[(df.t - chosen_t).abs().argmin()]
    st.latex(r"s(t)=0.2t^3-1.5t^2+4t\quad\Rightarrow\quad v(t)=s'(t),\ a(t)=v'(t)")
    st.write("This app rescales that presentation model to the selected patient, gait length, cadence, prosthetic, and use case.")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("s(t)", f"{row.s:.3f} m")
    mc2.metric("v(t)", f"{row.v:.3f} m/s")
    mc3.metric("a(t)", f"{row.a:.3f} m/s²")
    st.metric("Estimated impact at t", f"{row.impact_force_n:.0f} N")

st.subheader("Position, velocity, acceleration")
st.plotly_chart(make_calc_fig(df), use_container_width=True)

st.subheader("Engineering interpretation")
status = "ok" if comfort_index >= 55 and terrain == "Flat terrain" else "warning"
st.markdown(f"""
<div class="{status}">
<b>Comfort/stability score:</b> {comfort_index:.0f}/100. The selected prosthetic is optimized for {prost['label']}.
Negative acceleration near landing means the limb is slowing before foot contact, which can reduce shock. Higher terrain difficulty,
low friction, fatigue, and sports use increase estimated impact and may require adaptive control.
</div>
""", unsafe_allow_html=True)

st.dataframe(df.rename(columns={"t":"time_s", "s":"position_m", "v":"velocity_m_s", "a":"acceleration_m_s2", "jerk":"jerk_m_s3"}).round(4), use_container_width=True)

with st.expander("Model notes and classroom limitations"):
    st.write("""
    This is an educational simulator, not a clinical prosthetic prescription tool. It uses a smooth cubic gait model inspired by the project presentation,
    then rescales the motion using realistic patient and activity inputs. Real prosthetic design would require motion-capture data, force plates,
    residual-limb assessment, socket fit, clinician review, and validated control algorithms.
    """)
