"""Function Explorer Lab — contextual function analysis for secondary mathematics.

Deploy with: streamlit run app.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


st.set_page_config(page_title="Function Explorer Lab", page_icon="📈", layout="wide")

T = sp.Symbol("t", real=True)
TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)
SAFE_NAMES = {
    "t": T, "x": T, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sqrt": sp.sqrt,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "Abs": sp.Abs,
    "abs": sp.Abs, "pi": sp.pi, "E": sp.E,
}
SAFE_GLOBALS = {
    "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
    "Symbol": sp.Symbol, "Add": sp.Add, "Mul": sp.Mul, "Pow": sp.Pow,
}


@dataclass(frozen=True)
class Context:
    label: str
    output: str
    output_unit: str
    time_unit: str
    rate_noun: str
    icon: str


CONTEXTS = {
    "Business": Context("Business", "profit", "thousand dollars", "months", "profit growth", "business"),
    "Engineering": Context("Engineering", "displacement", "meters", "seconds", "velocity", "engineering"),
    "Medical": Context("Medical", "drug concentration", "mg/L", "hours", "concentration change", "medical"),
    "Sports science / medicine": Context("Sports science / medicine", "joint angle", "degrees", "seconds", "angular velocity", "prosthetic"),
    "Biology": Context("Biology", "population", "organisms", "days", "population growth", "biology"),
    "Bioengineering": Context("Bioengineering", "prosthetic knee angle", "degrees", "seconds", "angular velocity", "prosthetic"),
    "Economic forecasting": Context("Economic forecasting", "economic index", "index points", "quarters", "index growth", "economy"),
    "Weather": Context("Weather", "temperature", "°C", "hours", "warming/cooling rate", "weather"),
}

VISUALS = [
    "Walking person / prosthetic leg",
    "Moving object or projectile",
    "Tank, dosage, or concentration",
    "Population or cell colony",
    "Thermometer / weather",
    "Business or economic indicator",
]
DEFAULT_VISUAL = {
    "prosthetic": VISUALS[0], "engineering": VISUALS[1], "medical": VISUALS[2],
    "biology": VISUALS[3], "weather": VISUALS[4], "business": VISUALS[5],
    "economy": VISUALS[5],
}

MODELS = {
    "Honors Algebra 2": {
        "Business": ("Quadratic profit", "-2(t-6)^2 + 72", (0.0, 12.0)),
        "Engineering": ("Projectile height", "-4.9*t^2 + 24*t + 2", (0.0, 5.2)),
        "Medical": ("Treatment response", "-0.8*(t-5)^2 + 22", (0.0, 10.0)),
        "Sports science / medicine": ("Jump height", "-4.9*t^2 + 5.8*t + 0.8", (0.0, 1.35)),
        "Biology": ("Early population growth", "2*t^2 + 12*t + 80", (0.0, 10.0)),
        "Bioengineering": ("Sensor calibration", "0.5*t^2 + 3*t + 8", (0.0, 10.0)),
        "Economic forecasting": ("Business-cycle index", "-0.5*(t-8)^2 + 110", (0.0, 16.0)),
        "Weather": ("Daily temperature", "-0.7*(t-14)^2 + 29", (6.0, 22.0)),
    },
    "Precalculus": {
        "Business": ("Seasonal sales", "40 + 12*sin(pi*t/6)", (0.0, 24.0)),
        "Engineering": ("Damped vibration", "8*exp(-0.08*t)*cos(2*t)", (0.0, 15.0)),
        "Medical": ("Drug concentration", "18*t*exp(-0.5*t)", (0.0, 12.0)),
        "Sports science / medicine": ("Knee angle in a gait cycle", "35 + 25*sin(2*pi*t)", (0.0, 2.0)),
        "Biology": ("Logistic population", "500/(1 + 9*exp(-0.6*t))", (0.0, 15.0)),
        "Bioengineering": ("Prosthetic knee angle", "30 + 28*sin(2*pi*t)", (0.0, 2.0)),
        "Economic forecasting": ("Cyclical index", "100 + 15*sin(pi*t/4) + 0.8*t", (0.0, 20.0)),
        "Weather": ("Daily temperature cycle", "18 + 7*sin(pi*(t-8)/12)", (0.0, 24.0)),
    },
    "Calculus": {
        "Business": ("Revenue with changing growth", "-t^3 + 12*t^2 + 20*t + 40", (0.0, 10.0)),
        "Engineering": ("Position with direction changes", "t^3 - 9*t^2 + 18*t", (0.0, 8.0)),
        "Medical": ("Drug concentration", "24*t*exp(-0.45*t)", (0.0, 14.0)),
        "Sports science / medicine": ("Prosthetic gait angle", "32 + 24*sin(2*pi*t) + 5*sin(4*pi*t)", (0.0, 2.0)),
        "Biology": ("Logistic population", "800/(1 + 15*exp(-0.55*t))", (0.0, 18.0)),
        "Bioengineering": ("Prosthetic knee angle", "34 + 26*sin(2*pi*t) + 4*sin(4*pi*t)", (0.0, 2.0)),
        "Economic forecasting": ("Business-cycle polynomial", "0.08*t^4 - 1.6*t^3 + 8*t^2 + 3*t + 90", (0.0, 15.0)),
        "Weather": ("Temperature front", "12 + 16/(1 + exp(-0.8*(t-8)))", (0.0, 18.0)),
    },
}


def parse_function(raw: str) -> sp.Expr:
    """Parse a student expression while rejecting Python syntax and unknown names."""
    raw = raw.strip().replace("×", "*").replace("÷", "/").replace("−", "-")
    if not raw or len(raw) > 180 or "__" in raw:
        raise ValueError("Enter a function no longer than 180 characters.")
    if re.search(r"[\[\]{};'\"_:]", raw):
        raise ValueError("Brackets, quotes, underscores, and Python syntax are not allowed.")
    words = set(re.findall(r"[A-Za-z]+", raw))
    unknown = words.difference(SAFE_NAMES)
    if unknown:
        raise ValueError("Unknown name(s): " + ", ".join(sorted(unknown)))
    expression = parse_expr(
        raw, local_dict=SAFE_NAMES, global_dict=SAFE_GLOBALS,
        transformations=TRANSFORMS, evaluate=True,
    )
    if expression.free_symbols - {T}:
        raise ValueError("Use only t (or x) as the independent variable.")
    return expression


def real_number(value) -> float:
    z = complex(sp.N(value))
    return float(z.real) if abs(z.imag) < 1e-8 else math.nan


def finite_roots(expression: sp.Expr, lo: float, hi: float) -> list[float]:
    """Find real roots symbolically when practical, with a numeric fallback."""
    roots: list[float] = []
    try:
        solved = sp.solveset(expression, T, domain=sp.Interval(lo, hi))
        if isinstance(solved, sp.FiniteSet):
            roots = [real_number(v) for v in solved]
    except Exception:
        pass
    if not roots:
        xs = np.linspace(lo, hi, 900)
        fn = sp.lambdify(T, expression, "numpy")
        try:
            ys = np.asarray(fn(xs), dtype=float)
            if ys.ndim == 0:
                ys = np.full_like(xs, float(ys))
            for a, b, ya, yb in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
                if np.isfinite(ya) and abs(ya) < 1e-6:
                    roots.append(float(a))
                elif np.isfinite(ya) and np.isfinite(yb) and ya * yb < 0:
                    try:
                        roots.append(float(sp.nsolve(expression, (a + b) / 2)))
                    except Exception:
                        pass
        except Exception:
            pass
    return sorted({round(v, 7) for v in roots if np.isfinite(v) and lo <= v <= hi})


def singular_points(expr: sp.Expr, lo: float, hi: float) -> list[float]:
    points: list[float] = []
    try:
        singular = sp.singularities(expr, T)
        if isinstance(singular, sp.FiniteSet):
            points.extend(real_number(v) for v in singular)
    except Exception:
        pass
    denominator = sp.denom(sp.together(expr))
    if denominator != 1:
        points.extend(finite_roots(denominator, lo, hi))
    return sorted({round(v, 7) for v in points if np.isfinite(v) and lo <= v <= hi})


def sample_function(expr: sp.Expr, lo: float, hi: float, gaps: list[float]):
    xs = np.linspace(lo, hi, 1000)
    fn = sp.lambdify(T, expr, "numpy")
    try:
        ys = np.asarray(fn(xs), dtype=float)
        if ys.ndim == 0:
            ys = np.full_like(xs, float(ys))
    except Exception:
        ys = np.array([real_number(expr.subs(T, x)) for x in xs])
    ys[~np.isfinite(ys)] = np.nan
    for gap in gaps:
        ys[np.abs(xs - gap) < (hi - lo) / 300] = np.nan
    finite = ys[np.isfinite(ys)]
    if finite.size:
        low, high = np.nanpercentile(finite, [2, 98])
        pad = max(1.0, high - low)
        ys[(ys < low - 5 * pad) | (ys > high + 5 * pad)] = np.nan
    return xs, ys


def fmt(value: float, digits: int = 3) -> str:
    return "undefined" if not np.isfinite(value) else f"{value:,.{digits}f}".rstrip("0").rstrip(".")


def contextual_sentence(ctx: Context, t: float, y: float, slope: float, concavity: float) -> str:
    if not np.isfinite(y):
        return f"At {t:g} {ctx.time_unit}, the model is undefined. In context, it does not assign a valid {ctx.output} at this time."
    motion = "increasing" if slope > 1e-7 else "decreasing" if slope < -1e-7 else "momentarily unchanged"
    bend = "increasing at an increasing rate" if slope > 0 and concavity > 0 else (
        "increasing at a decreasing rate" if slope > 0 and concavity < 0 else
        "decreasing at an increasing rate" if slope < 0 and concavity < 0 else
        "decreasing at a decreasing rate" if slope < 0 and concavity > 0 else "changing curvature")
    return (f"At **{fmt(t)} {ctx.time_unit}**, the predicted **{ctx.output} is {fmt(y)} {ctx.output_unit}**. "
            f"It is {motion} at **{fmt(abs(slope))} {ctx.output_unit} per {ctx.time_unit}**. "
            f"Locally, the model is {bend}.")


def scaled_state(y: float, bounds: tuple[float, float], phase: float) -> float:
    """Map model output to 0–1 so unlike units can drive a physical visual."""
    low, high = bounds
    if not np.isfinite(y) or not np.isfinite(low) or not np.isfinite(high) or abs(high-low) < 1e-10:
        return phase
    return float(np.clip((y-low)/(high-low), 0, 1))


def walking_person(fig: go.Figure, phase: float, state: float, config: dict):
    """Draw a two-legged walker; the function value controls prosthetic-knee flexion."""
    height = config.get("height", 1.70)
    thigh, shank = .245*height, .246*height
    hip_y = thigh + shank + .10
    stride = config.get("stride", .70)
    hip_x = -.42 + .84*phase
    swing = math.radians(18*math.sin(2*math.pi*phase))
    natural = -swing
    prosthetic = swing
    knee_flex = math.radians(5 + 65*state) * max(0, math.sin(math.pi*phase))

    def leg(angle, flexion):
        knee = np.array([hip_x + thigh*math.sin(angle), hip_y-thigh*math.cos(angle)])
        lower_angle = angle-flexion
        foot = knee + np.array([shank*math.sin(lower_angle), -shank*math.cos(lower_angle)])
        return knee, foot

    knee_l, foot_l = leg(natural, 0.15*max(0, -math.sin(2*math.pi*phase)))
    knee_r, foot_r = leg(prosthetic, knee_flex)
    if config.get("side", "Right") == "Left":
        knee_l, knee_r, foot_l, foot_r = knee_r, knee_l, foot_r, foot_l
    hip = np.array([hip_x, hip_y])
    shoulder = hip + np.array([0, .43*height])
    head_y = shoulder[1] + .13*height
    # Torso, head and counter-swinging arms.
    fig.add_trace(go.Scatter(x=[hip_x, shoulder[0]], y=[hip_y, shoulder[1]], mode="lines", line=dict(width=12,color="#334155"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[shoulder[0]], y=[head_y], mode="markers", marker=dict(size=28,color="#f1c27d",line=dict(width=2,color="#334155")), hoverinfo="skip"))
    arm_dx = .19*math.sin(2*math.pi*phase)
    fig.add_trace(go.Scatter(x=[shoulder[0]-arm_dx, shoulder[0], shoulder[0]+arm_dx], y=[shoulder[1]-.34,shoulder[1],shoulder[1]-.34], mode="lines", line=dict(width=6,color="#64748b"), hoverinfo="skip"))
    # Biological leg and prosthetic leg use deliberately different colors.
    fig.add_trace(go.Scatter(x=[hip[0],knee_l[0],foot_l[0]], y=[hip[1],knee_l[1],max(foot_l[1],.02)], mode="lines+markers", line=dict(width=10,color="#2563eb"), marker=dict(size=9), name="biological leg"))
    fig.add_trace(go.Scatter(x=[hip[0],knee_r[0],foot_r[0]], y=[hip[1],knee_r[1],max(foot_r[1],.02)], mode="lines+markers", line=dict(width=10,color="#f97316"), marker=dict(size=9), name=f"{config.get('side','Right').lower()} prosthetic"))
    fig.add_shape(type="line",x0=-.95,x1=.95,y0=0,y1=0,line=dict(width=4,color="#64748b"))
    fig.add_annotation(x=-.92,y=head_y+.13,text=f"stride ≈ {stride:.2f} m",showarrow=False,xanchor="left")


def object_figure(ctx: Context, t: float, y: float, lo: float, hi: float,
                  bounds: tuple[float, float], config: dict) -> go.Figure:
    """A configurable visual model driven by time and the current function value."""
    phase = 0 if hi == lo else float(np.clip((t-lo)/(hi-lo),0,1))
    state = scaled_state(y,bounds,phase)
    visual = config["type"]
    fig = go.Figure()
    fig.update_xaxes(range=[-1,1],visible=False,fixedrange=True)
    fig.update_yaxes(range=[-.08,1.65],visible=False,fixedrange=True,scaleanchor="x")
    fig.update_layout(height=430,margin=dict(l=5,r=5,t=45,b=5),paper_bgcolor="#f8fbff",plot_bgcolor="#f8fbff",showlegend=False,title=config.get("title",f"{ctx.label} visual model"))
    if visual == "Walking person / prosthetic leg":
        walking_person(fig,phase,state,config)
        fig.update_xaxes(range=[-1,1]); fig.update_yaxes(range=[-.08,1.75])
    elif visual == "Moving object or projectile":
        horizontal = config.get("direction") == "Horizontal"
        x,ypos = ((-.82+1.64*state,.28) if horizontal else (0,.12+1.30*state))
        fig.add_shape(type="line",x0=-.9,x1=.9,y0=.08,y1=.08,line=dict(color="#64748b",width=4))
        fig.add_shape(type="circle",x0=x-.13,x1=x+.13,y0=ypos-.13,y1=ypos+.13,fillcolor="#3b82f6",line=dict(color="#1e3a8a",width=3))
        fig.add_annotation(x=x,y=min(1.55,ypos+.22),text=f"{ctx.output}: {fmt(y)} {ctx.output_unit}",showarrow=False)
    elif visual == "Tank, dosage, or concentration":
        fill=.06+.88*state
        fig.add_shape(type="rect",x0=-.38,x1=.38,y0=.08,y1=1.48,line=dict(color="#334155",width=5),fillcolor="white")
        fig.add_shape(type="rect",x0=-.35,x1=.35,y0=.11,y1=.11+1.32*fill,line_width=0,fillcolor="#e63971")
        for yy in [.35,.70,1.05,1.40]: fig.add_shape(type="line",x0=.25,x1=.38,y0=yy,y1=yy,line=dict(color="#334155",width=2))
        fig.add_annotation(x=0,y=.78,text=f"{fmt(y)}<br>{ctx.output_unit}",font=dict(size=18,color="white" if fill>.45 else "#334155"),showarrow=False)
    elif visual == "Population or cell colony":
        max_icons=int(config.get("icons",24)); n=max(1,int(round(1+state*(max_icons-1))))
        cols=6; px=[-.75+(i%cols)*.30 for i in range(n)]; py=[1.35-(i//cols)*.30 for i in range(n)]
        fig.add_trace(go.Scatter(x=px,y=py,mode="markers",marker=dict(size=22,color=np.arange(n),colorscale="Viridis",line=dict(width=1,color="white")),hoverinfo="skip"))
        fig.add_annotation(x=0,y=.05,text=f"Each symbol represents about {fmt(max(abs(bounds[1]),1)/max_icons)} {ctx.output_unit}",showarrow=False)
    elif visual == "Thermometer / weather":
        fill=.18+1.12*state
        color="#3b82f6" if state<.4 else "#f59e0b" if state<.7 else "#ef4444"
        fig.add_shape(type="rect",x0=-.10,x1=.10,y0=.25,y1=1.48,line=dict(color="#475569",width=5),fillcolor="white")
        fig.add_shape(type="rect",x0=-.07,x1=.07,y0=.25,y1=.25+fill,line_width=0,fillcolor=color)
        fig.add_shape(type="circle",x0=-.22,x1=.22,y0=.03,y1=.47,line=dict(color="#475569",width=5),fillcolor=color)
        fig.add_annotation(x=.48,y=.82,text=f"{fmt(y)} {ctx.output_unit}",showarrow=False,font=dict(size=20))
    else:  # business/economic scene
        heights=[.25,.42,.60,.18+.92*state]
        fig.add_trace(go.Bar(x=[-.66,-.22,.22,.66],y=heights,width=.28,marker_color=["#8ecae6","#219ebc","#ffb703","#fb8500"],hoverinfo="skip"))
        fig.add_annotation(x=0,y=1.35,text=f"{ctx.output.title()}: {fmt(y)} {ctx.output_unit}",showarrow=False,font=dict(size=18))
    fig.add_annotation(x=-.98,y=1.58,text=f"t = {fmt(t,2)} {ctx.time_unit}",showarrow=False,xanchor="left",font=dict(size=15,color="#334155"))
    return fig


def analysis_figure(expr, derivative, second, lo, hi, t1, t2, show_secant, show_tangent, level):
    gaps = singular_points(expr, lo, hi)
    xs, ys = sample_function(expr, lo, hi, gaps)
    f = sp.lambdify(T, expr, "numpy")
    fp = sp.lambdify(T, derivative, "numpy")
    y1, y2 = real_number(expr.subs(T, t1)), real_number(expr.subs(T, t2))
    slope = real_number(derivative.subs(T, t1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, name="f(t)", mode="lines", line=dict(color="#1665d8", width=4), hovertemplate="t=%{x:.3f}<br>f(t)=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[t1], y=[y1], name="Current point", mode="markers", marker=dict(size=14, color="#ff006e", line=dict(width=2, color="white"))))
    if show_secant and np.isfinite(y1) and np.isfinite(y2) and abs(t2-t1) > 1e-10:
        m = (y2-y1)/(t2-t1)
        fig.add_trace(go.Scatter(x=[t1,t2], y=[y1,y2], name=f"Secant: slope {fmt(m)}", mode="lines+markers", line=dict(color="#f59e0b", width=3, dash="dash")))
    if show_tangent and np.isfinite(y1) and np.isfinite(slope):
        width = (hi-lo)*.14
        tx = np.array([max(lo,t1-width), min(hi,t1+width)])
        fig.add_trace(go.Scatter(x=tx, y=y1+slope*(tx-t1), name=f"Tangent: slope {fmt(slope)}", mode="lines", line=dict(color="#d90429", width=3)))
    critical, inflections = [], []
    if level == "Calculus":
        critical = finite_roots(derivative, lo, hi)
        inflections = finite_roots(second, lo, hi)
        maxima = [r for r in critical if real_number(second.subs(T, r)) < -1e-7]
        minima = [r for r in critical if real_number(second.subs(T, r)) > 1e-7]
        other = [r for r in critical if r not in maxima + minima]
        feature_sets = [
            (maxima, "Local maximum", "#e63946", "triangle-down"),
            (minima, "Local minimum", "#8338ec", "triangle-up"),
            (other, "Critical point", "#6b7280", "diamond"),
            (inflections, "Possible inflection", "#06d6a0", "square"),
        ]
        for roots, name, color, symbol in feature_sets:
            if not roots:
                continue
            vals = [real_number(expr.subs(T,r)) for r in roots]
            fig.add_trace(go.Scatter(x=roots, y=vals, name=name, mode="markers", marker=dict(size=12,color=color,symbol=symbol), hovertemplate=name+"<br>t=%{x:.3f}<br>f(t)=%{y:.3f}<extra></extra>"))
    for gap in gaps:
        fig.add_vline(x=gap, line=dict(color="#e63946", width=3, dash="dot"))
        fig.add_annotation(x=gap, y=1, yref="paper", text="⚠ discontinuity", showarrow=True, arrowcolor="#e63946", bgcolor="#fff1f2")
    fig.update_layout(height=530, hovermode="x unified", margin=dict(l=15,r=15,t=35,b=10), legend=dict(orientation="h", y=1.02, x=.5, xanchor="center"), xaxis_title="time (t)", yaxis_title="f(t)", template="plotly_white")
    return fig, gaps, critical, inflections


st.title("📈 Function Explorer Lab")
st.caption("Explore functions, rates of change, derivatives, curvature, and discontinuities in a real-world context.")

with st.sidebar:
    st.header("1 · Choose the learning setup")
    level = st.selectbox("Course level", list(MODELS), index=1)
    context_choice = st.selectbox("Context", list(CONTEXTS) + ["My own context"])
    if context_choice == "My own context":
        custom_name = st.text_input("Context name", "Environmental science")
        output = st.text_input("What does f(t) measure?", "pollutant concentration")
        output_unit = st.text_input("Output units", "mg/L")
        time_unit = st.text_input("Time units", "days")
        ctx = Context(custom_name or "Custom", output, output_unit, time_unit, "rate of change", "engineering")
        base_context = "Engineering"
    else:
        ctx = CONTEXTS[context_choice]
        base_context = context_choice
    suggestion = MODELS[level][base_context]
    source = st.radio("Function source", ["Build with context parameters", "Suggested model", "Enter my own"])
    model_inputs = {}
    if source == "Build with context parameters":
        st.caption("Change an input and watch the function, graph, rates, and animation update together.")
        if base_context in ("Business","Economic forecasting"):
            model_inputs["baseline"] = st.number_input("Starting index / value",value=100.0,step=5.0)
            model_inputs["trend"] = st.number_input(f"Long-term change per {ctx.time_unit}",value=1.0,step=.25)
            model_inputs["amplitude"] = st.number_input("Cycle amplitude",min_value=0.0,value=15.0,step=1.0)
            model_inputs["period"] = st.number_input(f"Cycle length ({ctx.time_unit})",min_value=.1,value=8.0,step=.5)
            expression_text = f"{model_inputs['baseline']}+{model_inputs['trend']}*t+{model_inputs['amplitude']}*sin(2*pi*t/{model_inputs['period']})"
        elif base_context == "Engineering":
            model_inputs["initial"] = st.number_input(f"Initial {ctx.output} ({ctx.output_unit})",value=2.0)
            model_inputs["velocity"] = st.number_input(f"Initial velocity ({ctx.output_unit}/{ctx.time_unit})",value=24.0)
            model_inputs["acceleration"] = st.number_input(f"Constant acceleration ({ctx.output_unit}/{ctx.time_unit}²)",value=-9.8)
            expression_text = f"{model_inputs['initial']}+{model_inputs['velocity']}*t+0.5*({model_inputs['acceleration']})*t^2"
        elif base_context == "Medical":
            model_inputs["dose"] = st.number_input("Dose / response scale",min_value=.1,value=24.0,step=1.0)
            model_inputs["elimination"] = st.number_input(f"Elimination-rate constant (per {ctx.time_unit})",min_value=.01,value=.45,step=.01)
            expression_text = f"{model_inputs['dose']}*t*exp(-{model_inputs['elimination']}*t)"
        elif base_context in ("Sports science / medicine","Bioengineering"):
            model_inputs["rest"] = st.number_input("Resting / center joint angle (degrees)",value=32.0)
            model_inputs["flexion"] = st.number_input("Flexion amplitude (degrees)",min_value=1.0,value=28.0)
            model_inputs["cadence"] = st.slider("Cadence (steps per minute)",40,190,95)
            frequency=model_inputs["cadence"]/60
            expression_text = f"{model_inputs['rest']}+{model_inputs['flexion']}*sin(2*pi*{frequency}*t)"
        elif base_context == "Biology":
            model_inputs["capacity"] = st.number_input("Carrying capacity",min_value=1.0,value=800.0,step=25.0)
            model_inputs["initial"] = st.number_input("Initial population",min_value=.1,value=50.0,step=5.0)
            model_inputs["growth"] = st.number_input(f"Growth-rate constant (per {ctx.time_unit})",min_value=.001,value=.55,step=.05)
            ratio=max(model_inputs["capacity"]/model_inputs["initial"]-1,0)
            expression_text=f"{model_inputs['capacity']}/(1+{ratio}*exp(-{model_inputs['growth']}*t))"
        elif base_context == "Weather":
            model_inputs["mean"] = st.number_input(f"Mean {ctx.output} ({ctx.output_unit})",value=18.0)
            model_inputs["swing"] = st.number_input(f"Temperature swing ({ctx.output_unit})",min_value=0.0,value=7.0)
            model_inputs["peak"] = st.slider("Time of daily maximum (hour)",0.0,24.0,14.0,.5)
            expression_text=f"{model_inputs['mean']}+{model_inputs['swing']}*cos(2*pi*(t-{model_inputs['peak']})/24)"
        else:
            model_inputs["baseline"] = st.number_input(f"Baseline {ctx.output}",value=10.0)
            model_inputs["amplitude"] = st.number_input("Change amplitude",min_value=0.0,value=5.0)
            model_inputs["period"] = st.number_input(f"Cycle length ({ctx.time_unit})",min_value=.1,value=10.0)
            expression_text=f"{model_inputs['baseline']}+{model_inputs['amplitude']}*sin(2*pi*t/{model_inputs['period']})"
        st.code(f"f(t) = {expression_text}",language="text")
    else:
        default_expr = suggestion[1] if source == "Suggested model" else "t^3 - 6*t^2 + 9*t + 4"
        expression_text = st.text_input("f(t) =",default_expr,help="Use t or x. Examples: t^2, sin(t), exp(-t), sqrt(t), 1/(t-2). Use * for multiplication.")
    st.caption(f"Suggested: **{suggestion[0]}** — `{suggestion[1]}`")
    st.header("2 · Set the viewing window")
    c1, c2 = st.columns(2)
    lo = c1.number_input("Start time", value=float(suggestion[2][0]), step=0.5)
    hi = c2.number_input("End time", value=float(suggestion[2][1]), step=0.5)
    show_secant = st.checkbox("Show secant line", value=True)
    show_tangent = st.checkbox("Show tangent line", value=(level == "Calculus"), disabled=(level != "Calculus"))
    st.header("3 · Design the contextual visual")
    st.caption("These parameters give the function output a visible, meaningful state. They do not alter f(t) unless you incorporate them into the function above.")
    default_visual = DEFAULT_VISUAL.get(ctx.icon,VISUALS[1])
    visual_type = st.selectbox("What should students see moving or changing?",VISUALS,index=VISUALS.index(default_visual))
    visual_title = st.text_input("Visual title",f"{ctx.output.title()} over time")
    visual_config = {"type":visual_type,"title":visual_title}
    if visual_type == "Walking person / prosthetic leg":
        st.caption("The current f(t) value controls knee flexion; time controls the position within the gait cycle.")
        visual_config["height"] = st.slider("Person height (m)",1.20,2.10,1.70,.01)
        visual_config["stride"] = st.slider("Stride length (m)",.25,1.40,.70,.01)
        visual_config["side"] = st.selectbox("Prosthetic leg",["Right","Left"])
    elif visual_type == "Moving object or projectile":
        visual_config["direction"] = st.radio("Direction represented",["Horizontal","Vertical"],horizontal=True)
        st.caption(f"The displayed position is scaled from the minimum to maximum {ctx.output} in the chosen time window.")
    elif visual_type == "Tank, dosage, or concentration":
        visual_config["capacity"] = st.number_input(f"Reference capacity ({ctx.output_unit})",min_value=.01,value=100.0)
        st.caption("The fill level is normalized to the visible range; the exact modeled value remains labeled.")
    elif visual_type == "Population or cell colony":
        visual_config["icons"] = st.slider("Maximum visible symbols",12,48,24,6)
        st.caption("The symbols provide a scaled visual; the numerical model output remains exact.")
    elif visual_type == "Thermometer / weather":
        st.caption("The column height and color respond to the model output over the selected interval.")
    else:
        st.caption("The fourth indicator bar responds to the model output; the label reports its exact value and units.")

if hi <= lo:
    st.error("End time must be greater than start time.")
    st.stop()

try:
    expr = parse_function(expression_text)
except Exception as exc:
    st.error(f"I could not interpret that function: {exc}")
    st.info("Try an expression such as `-2*(t-4)^2+50`, `20*sin(t)+40`, or `100/(1+exp(-t))`.")
    st.stop()

derivative = sp.diff(expr, T)
second = sp.diff(derivative, T)
if "time_cursor" not in st.session_state or not (lo <= st.session_state.time_cursor <= hi):
    st.session_state.time_cursor = float(lo)
    st.session_state.time_slider = float(lo)
if "playing" not in st.session_state:
    st.session_state.playing = False

st.markdown("### Model and controls")
formula_cols = st.columns(3)
formula_cols[0].latex("f(t)=" + sp.latex(expr))
if level == "Calculus":
    formula_cols[1].latex("f'(t)=" + sp.latex(derivative))
    formula_cols[2].latex("f''(t)=" + sp.latex(second))
else:
    formula_cols[1].info("Use the secant line to study average rate of change.")
    formula_cols[2].info("Select Calculus to reveal derivative and concavity analysis.")


@st.fragment(run_every=0.45)
def live_lab():
    if st.session_state.playing:
        step = max((hi-lo)/100, 0.001) * st.session_state.get("speed", 2.0)
        st.session_state.time_cursor = lo if st.session_state.time_cursor + step > hi else st.session_state.time_cursor + step
        st.session_state.time_slider = float(st.session_state.time_cursor)
    elif "time_slider" not in st.session_state:
        st.session_state.time_slider = float(st.session_state.time_cursor)
    controls = st.columns([1,1,2])
    if controls[0].button("⏸ Pause" if st.session_state.playing else "▶ Play", use_container_width=True):
        st.session_state.playing = not st.session_state.playing
        st.rerun(scope="fragment")
    if controls[1].button("↺ Reset", use_container_width=True):
        st.session_state.playing = False
        st.session_state.time_cursor = float(lo)
        st.session_state.time_slider = float(lo)
        st.rerun(scope="fragment")
    st.session_state.speed = controls[2].select_slider("Playback speed", options=[0.5,1.0,2.0,4.0], value=st.session_state.get("speed",2.0), format_func=lambda x:f"{x:g}×")
    t1 = st.slider("Current time", min_value=float(lo), max_value=float(hi), step=max((hi-lo)/300,0.0001), key="time_slider")
    st.session_state.time_cursor = t1
    interval_default = min(hi, t1 + (hi-lo)/5)
    if "second_slider" not in st.session_state or not (lo <= st.session_state.second_slider <= hi):
        st.session_state.second_slider = float(interval_default)
    t2 = st.slider("Second time (for average rate of change)", min_value=float(lo), max_value=float(hi), step=max((hi-lo)/300,0.0001), key="second_slider")

    y1 = real_number(expr.subs(T,t1))
    y2 = real_number(expr.subs(T,t2))
    slope = real_number(derivative.subs(T,t1))
    concavity = real_number(second.subs(T,t1))
    avg = (y2-y1)/(t2-t1) if np.isfinite(y1) and np.isfinite(y2) and abs(t2-t1)>1e-10 else math.nan
    graph, gaps, critical, inflections = analysis_figure(expr, derivative, second, lo, hi, t1, t2, show_secant, show_tangent, level)
    _, visual_values = sample_function(expr,lo,hi,gaps)
    finite_visual = visual_values[np.isfinite(visual_values)]
    visual_bounds = ((float(np.nanmin(finite_visual)),float(np.nanmax(finite_visual)))
                     if finite_visual.size else (0.0,1.0))
    left, right = st.columns([2.2,1])
    left.plotly_chart(graph, use_container_width=True, config={"displaylogo":False})
    right.plotly_chart(object_figure(ctx,t1,y1,lo,hi,visual_bounds,visual_config), use_container_width=True, config={"displayModeBar":False})
    right.caption(f"The exact value is {fmt(y1)} {ctx.output_unit}. The visual maps the model's output range to a visible state. It is educational—not a clinical, engineering, or forecasting validation.")

    st.markdown("### What the current location means")
    st.info(contextual_sentence(ctx,t1,y1,slope,concavity))
    cards = st.columns(4)
    cards[0].metric(ctx.output.title(), f"{fmt(y1)} {ctx.output_unit}")
    cards[1].metric("Average rate", f"{fmt(avg)} {ctx.output_unit}/{ctx.time_unit}", help=f"From t={fmt(t1)} to t={fmt(t2)}")
    cards[2].metric("Instantaneous rate", f"{fmt(slope)} {ctx.output_unit}/{ctx.time_unit}" if level=="Calculus" else "Calculus feature")
    concavity_label = "Up" if concavity>1e-7 else "Down" if concavity < -1e-7 else "Neither / changing"
    cards[3].metric("Concavity", concavity_label if level=="Calculus" else "Calculus feature")

    tabs = st.tabs(["Interpretation", "Key features", "Student questions"])
    with tabs[0]:
        if model_inputs:
            st.write("**Model inputs currently shaping the function:** " + ", ".join(f"{name.replace('_',' ')} = {value:g}" for name,value in model_inputs.items()))
        if np.isfinite(avg):
            st.write(f"**Secant meaning:** From {fmt(t1)} to {fmt(t2)} {ctx.time_unit}, {ctx.output} changes on average by **{fmt(avg)} {ctx.output_unit} per {ctx.time_unit}**.")
        if level == "Calculus" and np.isfinite(slope):
            intercept = y1-slope*t1
            st.write(f"**Tangent line:**  (L(t)={fmt(y1)}+{fmt(slope)}(t-{fmt(t1)})). Near this time, it gives a linear estimate of {ctx.output}.")
            st.write(f"In slope-intercept form:  (L(t)={fmt(slope)}t+{fmt(intercept)}).")
    with tabs[1]:
        if gaps:
            st.warning("Discontinuities in the selected interval: " + ", ".join(f"t={fmt(v)}" for v in gaps) + ". These are flagged by red dotted lines on the graph.")
        else:
            st.success("No isolated discontinuities were detected in the selected interval.")
        if level == "Calculus":
            feature_lines = []
            for v in critical:
                second_at_v = real_number(second.subs(T, v))
                kind = "local maximum" if second_at_v < -1e-7 else "local minimum" if second_at_v > 1e-7 else "unclassified critical point"
                feature_lines.append(f"t={fmt(v)}: {kind}, {ctx.output} = {fmt(real_number(expr.subs(T,v)))} {ctx.output_unit}")
            st.write("**Critical-point interpretation:** " + ("; ".join(feature_lines) if feature_lines else "none detected"))
            st.write("**Possible inflection times:** " + (", ".join(fmt(v) for v in inflections) if inflections else "none detected"))
            st.caption("A possible inflection point must be verified by a change in concavity. Endpoints and model-domain restrictions also matter when identifying absolute extrema.")
    with tabs[2]:
        st.markdown(f"""
1. What does the point \\(({fmt(t1)}, {fmt(y1)})\\) mean for **{ctx.output}**?
2. Compare the average rate over the selected interval with the instantaneous rate. Why are they different?
3. Move the slider toward a maximum or minimum. What happens to the slope?
4. Where does the graph change concavity, and what does that change mean in this context?
5. What assumptions would need to be checked before using this model to make a real prediction?
""")


live_lab()

with st.expander("Teacher notes and mathematical cautions"):
    st.markdown("""
- The app treats the horizontal axis as time and uses the units selected in the context.
- Symbolic feature detection works best with standard algebraic, exponential, logarithmic, and trigonometric functions. Highly oscillatory or unusual expressions may require students to verify features visually or algebraically.
- A critical point is a candidate for a maximum or minimum; students should use sign changes, derivative tests, and endpoints to classify it.
- The contextual animation is intentionally conceptual. For scientific validity, replace the suggested function with a model fitted to measured data and document its domain.
""")

st.caption("Built for classroom exploration • Enter only mathematical expressions, not Python code.")
