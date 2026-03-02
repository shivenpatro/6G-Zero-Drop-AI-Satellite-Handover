"""
Phase 7 -- Interactive Streamlit Command Center
================================================
Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ── LSTM model definition (mirrors lstm_trainer.py) ──────────────────────────
class SatelliteLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ── constants ────────────────────────────────────────────────────────────────
EARTH_R     = 6_371.0
ALT         = 500.0
VEL         = 7.5
FREQ        = 28.0
TX_PWR      = 70.0
RSRP_FLOOR  = -130.0
DT          = 0.1
N_STEPS     = 10_000
SAT_A_X0    = 0.0
SAT_B_X0    = 1_200.0
C_KMS       = 3e5
PROC_MS     = 20.0
HO_THRESH   = -110.0
SEQ_LEN     = 50
TREND_WIN   = 10
REACTIVE_MS = 3500.0
PENALTY_STEPS = int(REACTIVE_MS / (DT * 1000))
COOLDOWN_STEPS = 100
SOFT_MS     = 25.0
EAP_MS, DHCP_MS, BGP_MS = 1200.0, 800.0, 1500.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── cached model loading ─────────────────────────────────────────────────────
@st.cache_resource
def load_lstm():
    model = SatelliteLSTM().to(DEVICE)
    model.load_state_dict(
        torch.load("satellite_lstm.pth", map_location=DEVICE, weights_only=True))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("rsrp_scaler.joblib")

@st.cache_resource
def build_fuzzy():
    pred = ctrl.Antecedent(np.arange(-130, -89, 0.5), "predicted_rsrp")
    trend = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), "rsrp_trend")
    urg = ctrl.Consequent(np.arange(0, 101, 1), "handover_urgency")
    pred["poor"]     = fuzz.trapmf(pred.universe, [-130,-130,-115,-110])
    pred["marginal"] = fuzz.trimf(pred.universe,  [-115,-107.5,-100])
    pred["good"]     = fuzz.trapmf(pred.universe, [-105,-100,-89,-89])
    trend["dropping_fast"] = fuzz.trapmf(trend.universe, [-5,-5,-2,-1])
    trend["stable"]        = fuzz.trimf(trend.universe,  [-2,0,2])
    trend["improving"]     = fuzz.trapmf(trend.universe, [1,2,5,5])
    urg["low"]      = fuzz.trapmf(urg.universe, [0,0,20,40])
    urg["medium"]   = fuzz.trimf(urg.universe,  [30,50,70])
    urg["critical"] = fuzz.trapmf(urg.universe, [60,80,100,100])
    rules = [
        ctrl.Rule(pred["poor"] & trend["dropping_fast"],    urg["critical"]),
        ctrl.Rule(pred["poor"] & trend["stable"],           urg["critical"]),
        ctrl.Rule(pred["poor"] & trend["improving"],        urg["medium"]),
        ctrl.Rule(pred["marginal"] & trend["dropping_fast"],urg["medium"]),
        ctrl.Rule(pred["marginal"] & trend["stable"],       urg["low"]),
        ctrl.Rule(pred["marginal"] & trend["improving"],    urg["low"]),
        ctrl.Rule(pred["good"],                             urg["low"]),
    ]
    return ctrl.ControlSystem(rules)


# ── protocol audit text ──────────────────────────────────────────────────────
def build_audit_text(reactive_events, ai_events):
    lines = []
    def ts(t): return f"[t={t:>7.1f}s]"

    lines.append("=" * 72)
    lines.append("  6G ZERO-DROP: CRYPTOGRAPHIC PROTOCOL AUDIT LOG")
    lines.append("=" * 72)
    lines.append(f"  EAP-AKA' Auth .... {EAP_MS:.0f} ms  |  DHCPv6 ...... {DHCP_MS:.0f} ms  |  BGP/ISL ..... {BGP_MS:.0f} ms")
    lines.append(f"  Reactive total = {REACTIVE_MS:.0f} ms   |   AI switch = {SOFT_MS:.0f} ms")
    lines.append("")

    lines.append("-" * 72)
    lines.append("  SECTION A: REACTIVE -- Break-Before-Make")
    lines.append("-" * 72)
    for idx, ev in enumerate(reactive_events, 1):
        t = ev["time"]
        lines.append(f"\n  -- Handover #{idx} --")
        lines.append(f"{ts(t)} CRITICAL: Sat-{ev['from']} lost (RSRP={ev['rsrp']:.1f} dBm)")
        lines.append(f"{ts(t)} PROCESS : [6G-AKA'] +{EAP_MS:.0f}ms")
        t += EAP_MS/1000
        lines.append(f"{ts(t)} PROCESS : [DHCPv6]  +{DHCP_MS:.0f}ms")
        t += DHCP_MS/1000
        lines.append(f"{ts(t)} PROCESS : [BGP/ISL] +{BGP_MS:.0f}ms")
        t += BGP_MS/1000
        lines.append(f"{ts(t)} RESTORED: Sat-{ev['to']}. Blackout={REACTIVE_MS:.0f}ms")

    lines.append(f"\n  Reactive total: {len(reactive_events)} HO x {REACTIVE_MS:.0f}ms "
                 f"= {len(reactive_events)*REACTIVE_MS:.0f}ms blackout")

    lines.append("\n" + "-" * 72)
    lines.append("  SECTION B: NEURO-FUZZY AI -- Make-Before-Break")
    lines.append("-" * 72)
    for idx, ev in enumerate(ai_events, 1):
        t = ev["time"]
        lines.append(f"\n  -- AI Handover #{idx} --")
        lines.append(f"{ts(t)} PREDICT : Urgency={ev['urgency']:.1f}%  RSRP={ev['rsrp']:.1f}dBm")
        lines.append(f"{ts(t)} BG-AUTH : [6G-AKA'] +{EAP_MS:.0f}ms (non-blocking)")
        lines.append(f"{ts(t+EAP_MS/1000)} BG-DHCP : [DHCPv6]  +{DHCP_MS:.0f}ms (non-blocking)")
        lines.append(f"{ts(t+(EAP_MS+DHCP_MS)/1000)} BG-BGP  : [BGP/ISL] +{BGP_MS:.0f}ms (non-blocking)")
        lines.append(f"{ts(t)} SWITCH  : Radio retuned to Sat-{ev['to']}. Delay={SOFT_MS:.0f}ms")

    lines.append(f"\n  AI total: {len(ai_events)} HO x {SOFT_MS:.0f}ms "
                 f"= {len(ai_events)*SOFT_MS:.0f}ms interruption")
    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


# ── full simulation (cached by slider values) ────────────────────────────────
@st.cache_data(show_spinner=False)
def run_full_simulation(noise_sigma: float, urgency_thresh: float):
    model  = load_lstm()
    scaler = load_scaler()
    cs     = build_fuzzy()

    # ── Phase 1: generate telemetry (vectorised numpy) ────────────────────
    rng = np.random.default_rng(42)
    t_arr = np.arange(N_STEPS) * DT
    x_a = SAT_A_X0 - VEL * t_arr
    x_b = SAT_B_X0 - VEL * t_arr
    d_a = np.sqrt(x_a**2 + ALT**2)
    d_b = np.sqrt(x_b**2 + ALT**2)
    h_dist = np.sqrt((EARTH_R + ALT)**2 - EARTH_R**2)
    ok_a = np.abs(x_a) < h_dist
    ok_b = np.abs(x_b) < h_dist

    def _rsrp(d, ok):
        fspl = 20*np.log10(d) + 20*np.log10(FREQ) + 92.45
        n = rng.normal(0, noise_sigma, d.shape)
        r = TX_PWR - fspl + n
        return np.where(ok, r, RSRP_FLOOR)

    rsrp_a = np.round(_rsrp(d_a, ok_a), 2)
    rsrp_b = np.round(_rsrp(d_b, ok_b), 2)
    dist_a = np.round(d_a, 4)
    dist_b = np.round(d_b, 4)
    times  = np.round(t_arr, 2)

    # ── Phase 2: reactive router (numpy arrays, no iloc) ─────────────────
    n = N_STEPS
    bl_lat = np.zeros(n)
    cur_r = 0  # 0=A, 1=B
    ho_rem = 0; cd_rem = 0; bl_ho_cnt = 0; tgt_r = 1
    bl_events = []

    for i in range(n):
        if ho_rem > 0:
            bl_lat[i] = REACTIVE_MS; ho_rem -= 1
            if ho_rem == 0: cur_r = tgt_r; cd_rem = COOLDOWN_STEPS
            continue
        if cd_rem > 0: cd_rem -= 1
        if cur_r == 0:
            cr, cd, ar, alt_idx = rsrp_a[i], dist_a[i], rsrp_b[i], 1
        else:
            cr, cd, ar, alt_idx = rsrp_b[i], dist_b[i], rsrp_a[i], 0
        if cr < HO_THRESH and cd_rem == 0 and ar > RSRP_FLOOR:
            f_name = "A" if cur_r == 0 else "B"
            t_name = "A" if alt_idx == 0 else "B"
            tgt_r = alt_idx; ho_rem = PENALTY_STEPS; bl_ho_cnt += 1
            bl_lat[i] = REACTIVE_MS
            bl_events.append({"time": times[i], "from": f_name, "to": t_name, "rsrp": float(cr)})
            continue
        bl_lat[i] = (cd / C_KMS) * 2 * 1000 + PROC_MS

    # ── Phase 5: neuro-fuzzy router (optimised) ──────────────────────────
    fsim = ctrl.ControlSystemSimulation(cs)
    ai_lat = np.zeros(n)
    urg_log = np.zeros(n)
    cur_ai = 0  # 0=A, 1=B
    hist: list[float] = []
    ai_events = []

    for i in range(n):
        if cur_ai == 0:
            cr, cd, ar = rsrp_a[i], dist_a[i], rsrp_b[i]
            alt_idx = 1
        else:
            cr, cd, ar = rsrp_b[i], dist_b[i], rsrp_a[i]
            alt_idx = 0
        hist.append(float(cr))
        if len(hist) < SEQ_LEN:
            ai_lat[i] = (cd / C_KMS) * 2 * 1000 + PROC_MS
            continue

        w = np.array(hist[-SEQ_LEN:])
        sc = scaler.transform(w.reshape(-1, 1)).flatten()
        t = torch.tensor(sc, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        with torch.no_grad():
            ps = model(t).item()
        pred_dbm = float(scaler.inverse_transform([[ps]])[0, 0])

        # fast-path: if prediction is clearly safe, skip fuzzy entirely
        if pred_dbm > -100.0:
            ai_lat[i] = (cd / C_KMS) * 2 * 1000 + PROC_MS
            continue

        if len(hist) >= TREND_WIN:
            tv = (hist[-1] - hist[-TREND_WIN]) / (TREND_WIN * 0.1)
        else:
            tv = 0.0

        fsim.input["predicted_rsrp"] = np.clip(pred_dbm, -130, -90)
        fsim.input["rsrp_trend"]     = np.clip(tv, -5, 5)
        try:
            fsim.compute()
            u = float(fsim.output["handover_urgency"])
        except Exception:
            u = 0.0
        urg_log[i] = u

        if u >= urgency_thresh and ar > HO_THRESH:
            f_name = "A" if cur_ai == 0 else "B"
            t_name = "A" if alt_idx == 0 else "B"
            ai_lat[i] = SOFT_MS
            ai_events.append({"time": float(times[i]), "from": f_name, "to": t_name,
                              "urgency": u, "rsrp": float(cr), "pred": pred_dbm})
            cur_ai = alt_idx; hist.clear()
            continue

        ai_lat[i] = (cd / C_KMS) * 2 * 1000 + PROC_MS

    # Build DataFrame for charts
    df = pd.DataFrame({
        "time": times,
        "satA_distance": dist_a, "satA_rsrp": rsrp_a,
        "satB_distance": dist_b, "satB_rsrp": rsrp_b,
    })

    return df, bl_lat, bl_ho_cnt, bl_events, ai_lat, urg_log, ai_events


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="6G Zero-Drop AI", layout="wide",
                   page_icon="\U0001f6f0\ufe0f")

st.markdown("""
<h1 style='text-align:center; margin-bottom:0;'>
    6G Zero-Drop: AI-Predictive Satellite Handover
</h1>
<p style='text-align:center; color:grey; font-size:0.95rem; margin-top:4px;'>
    PyTorch LSTM &middot; Fuzzy Logic (skfuzzy) &middot; 6G EAP-AKA' Protocol Simulation
    &middot; LEO @ 500 km &middot; 28 GHz mmWave
</p>
<hr style='margin-top:8px; margin-bottom:18px;'>
""", unsafe_allow_html=True)

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation Controls")

    noise_sigma = st.slider(
        "Atmospheric Interference (sigma, dB)",
        min_value=0.5, max_value=3.0, value=1.5, step=0.1,
        help="Gaussian noise standard deviation added to RSRP")

    urgency_thresh = st.slider(
        "AI Handover Urgency Threshold (%)",
        min_value=60, max_value=90, value=75, step=1,
        help="Fuzzy urgency score that triggers a soft handover")

    st.divider()
    st.markdown("**Architecture**")
    st.caption(
        "1. Environment generates orbital RSRP with tunable noise.  \n"
        "2. Reactive router uses a crisp -110 dBm threshold.  \n"
        "3. LSTM predicts RSRP 3 s ahead.  \n"
        "4. Fuzzy engine evaluates prediction + trend to produce urgency.  \n"
        "5. Protocol simulator logs 6G-AKA', DHCPv6, BGP layers.")

# ── run simulation ───────────────────────────────────────────────────────────
with st.spinner("Generating orbital telemetry & running both routers..."):
    df, bl_lat, bl_ho_cnt, bl_events, ai_lat, urg_arr, ai_events = \
        run_full_simulation(float(noise_sigma), float(urgency_thresh))

# ── KPI row ──────────────────────────────────────────────────────────────────
bl_blackout_ms = (bl_lat >= REACTIVE_MS).sum() * DT * 1000
ai_blackout_ms = len(ai_events) * SOFT_MS
if bl_blackout_ms > 0:
    efficiency = (1 - ai_blackout_ms / bl_blackout_ms) * 100
else:
    efficiency = 100.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Reactive Handovers", f"{bl_ho_cnt}")
k2.metric("Reactive Blackout", f"{bl_blackout_ms/1000:.1f} s")
k3.metric("AI Blackout", f"{ai_blackout_ms:.0f} ms")
k4.metric("Efficiency Gain", f"{efficiency:.1f} %")

st.markdown("---")

# ── Chart 1: Latency Comparison (Plotly) ─────────────────────────────────────
t_arr = df["time"].values

fig1 = go.Figure()
fig1.add_trace(go.Scattergl(
    x=t_arr, y=bl_lat, mode="lines", name="Reactive Baseline",
    line=dict(color="#EF5350", width=1), opacity=0.85))
fig1.add_trace(go.Scattergl(
    x=t_arr, y=ai_lat, mode="lines", name="Neuro-Fuzzy AI",
    line=dict(color="#1E88E5", width=1.2), opacity=0.95))
for ev in ai_events:
    fig1.add_vline(x=ev["time"], line_dash="dash", line_color="#43A047",
                   line_width=1, opacity=0.7)
fig1.update_layout(
    title="Latency Comparison: Reactive vs Neuro-Fuzzy AI Router",
    xaxis_title="Time (s)", yaxis_title="Latency (ms)",
    yaxis=dict(range=[0, 4200]),
    template="plotly_dark", height=400, margin=dict(t=40, b=40),
    legend=dict(orientation="h", yanchor="top", y=1.12, x=0.5, xanchor="center"))
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: RSRP + Urgency dual-axis ───────────────────────────────────────
fig2 = make_subplots(specs=[[{"secondary_y": True}]])

fig2.add_trace(go.Scattergl(
    x=t_arr, y=df["satA_rsrp"].values, mode="lines", name="Sat-A RSRP",
    line=dict(color="#FF7043", width=0.8), opacity=0.8), secondary_y=False)
fig2.add_trace(go.Scattergl(
    x=t_arr, y=df["satB_rsrp"].values, mode="lines", name="Sat-B RSRP",
    line=dict(color="#42A5F5", width=0.8), opacity=0.8), secondary_y=False)
fig2.add_hline(y=HO_THRESH, line_dash="dash", line_color="#F44336",
               line_width=1, annotation_text="-110 dBm threshold",
               secondary_y=False)

fig2.add_trace(go.Scattergl(
    x=t_arr, y=urg_arr, mode="lines", name="Fuzzy Urgency",
    line=dict(color="#CE93D8", width=1), opacity=0.7,
    fill="tozeroy", fillcolor="rgba(206,147,216,0.15)"), secondary_y=True)
fig2.add_hline(y=float(urgency_thresh), line_dash="dot", line_color="#FFD600",
               line_width=1.5,
               annotation_text=f"Urgency trigger ({urgency_thresh}%)",
               secondary_y=True)

for ev in ai_events:
    fig2.add_vline(x=ev["time"], line_dash="dash", line_color="#43A047",
                   line_width=1, opacity=0.6)

fig2.update_layout(
    title="The AI Brain: RSRP Signals + Fuzzy Urgency Score",
    xaxis_title="Time (s)",
    template="plotly_dark", height=420, margin=dict(t=40, b=40),
    legend=dict(orientation="h", yanchor="top", y=1.14, x=0.5, xanchor="center"))
fig2.update_yaxes(title_text="RSRP (dBm)", range=[-135, -90], secondary_y=False)
fig2.update_yaxes(title_text="Urgency (%)", range=[-5, 105], secondary_y=True)

st.plotly_chart(fig2, use_container_width=True)

# ── protocol audit terminal ──────────────────────────────────────────────────
with st.expander("View Cryptographic Protocol Audit Logs", expanded=False):
    audit = build_audit_text(bl_events, ai_events)
    st.code(audit, language="log")
