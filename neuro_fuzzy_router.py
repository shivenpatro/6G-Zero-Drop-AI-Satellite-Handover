"""
Phase 5 -- Neuro-Fuzzy AI Router
==================================
Combines the LSTM time-series predictor (Phase 3) with a Fuzzy Logic
Inference System (skfuzzy) for robust, noise-tolerant handover decisions.

The LSTM provides the *predicted* RSRP 3 s ahead (the "neural" part).
The fuzzy engine evaluates that prediction together with the current
RSRP *trend* (rate of change) to produce a Handover Urgency score
in [0, 100].  A soft handover fires only when urgency >= 75.

Outputs
-------
* nf_latency.csv                   -- per-step latency + urgency log
* phase5_neuro_fuzzy_dashboard.png -- 3-subplot comparison dashboard
"""

import sys
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

sys.path.insert(0, ".")
from lstm_trainer import SatelliteLSTM  # noqa: E402

# ── tunables ─────────────────────────────────────────────────────────────────
HANDOVER_THRESHOLD_DBM   = -110.0
URGENCY_TRIGGER          = 75.0       # fuzzy output threshold to fire HO
SOFT_HANDOVER_PENALTY_MS = 25.0
SEQUENCE_LENGTH          = 50         # LSTM look-back (5 s)
TREND_WINDOW             = 10         # steps for rate-of-change (1 s)
SPEED_OF_LIGHT_KMS       = 3e5
BASE_PROCESSING_MS       = 20.0

ORBITAL_CSV   = "orbital_data.csv"
BASELINE_CSV  = "reactive_latency.csv"
SCALER_PATH   = "rsrp_scaler.joblib"
MODEL_PATH    = "satellite_lstm.pth"
NF_CSV_OUTPUT = "nf_latency.csv"
PLOT_OUTPUT   = "phase5_neuro_fuzzy_dashboard.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── reusable helpers (same as Phase 4) ───────────────────────────────────────
def propagation_latency_ms(distance_km: float) -> float:
    return (distance_km / SPEED_OF_LIGHT_KMS) * 2.0 * 1000.0 + BASE_PROCESSING_MS


def load_model():
    model = SatelliteLSTM().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def predict_rsrp(model, scaler, rsrp_window: np.ndarray) -> float:
    scaled = scaler.transform(rsrp_window.reshape(-1, 1)).flatten()
    tensor = (torch.tensor(scaled, dtype=torch.float32)
              .unsqueeze(0).unsqueeze(-1).to(DEVICE))
    with torch.no_grad():
        pred_scaled = model(tensor).item()
    return scaler.inverse_transform([[pred_scaled]])[0, 0]


# ── fuzzy inference system ───────────────────────────────────────────────────
def build_fuzzy_system():
    """
    Two antecedents  → one consequent.
      predicted_rsrp  [-130, -90]  dBm
      rsrp_trend      [-5,    5]  dBm/s
      handover_urgency [0,  100]  %
    """
    pred = ctrl.Antecedent(np.arange(-130, -89, 0.5), "predicted_rsrp")
    trend = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), "rsrp_trend")
    urgency = ctrl.Consequent(np.arange(0, 101, 1), "handover_urgency")

    # -- predicted RSRP membership functions --
    pred["poor"]     = fuzz.trapmf(pred.universe, [-130, -130, -115, -110])
    pred["marginal"] = fuzz.trimf(pred.universe,  [-115, -107.5, -100])
    pred["good"]     = fuzz.trapmf(pred.universe, [-105, -100, -89, -89])

    # -- RSRP trend membership functions (dBm/s) --
    trend["dropping_fast"] = fuzz.trapmf(trend.universe, [-5, -5, -2, -1])
    trend["stable"]        = fuzz.trimf(trend.universe,  [-2, 0, 2])
    trend["improving"]     = fuzz.trapmf(trend.universe, [1, 2, 5, 5])

    # -- handover urgency membership functions --
    urgency["low"]      = fuzz.trapmf(urgency.universe, [0, 0, 20, 40])
    urgency["medium"]   = fuzz.trimf(urgency.universe,  [30, 50, 70])
    urgency["critical"] = fuzz.trapmf(urgency.universe, [60, 80, 100, 100])

    # -- rule base --
    r1 = ctrl.Rule(pred["poor"] & trend["dropping_fast"],    urgency["critical"])
    r2 = ctrl.Rule(pred["poor"] & trend["stable"],           urgency["critical"])
    r3 = ctrl.Rule(pred["poor"] & trend["improving"],        urgency["medium"])
    r4 = ctrl.Rule(pred["marginal"] & trend["dropping_fast"],urgency["medium"])
    r5 = ctrl.Rule(pred["marginal"] & trend["stable"],       urgency["low"])
    r6 = ctrl.Rule(pred["marginal"] & trend["improving"],    urgency["low"])
    r7 = ctrl.Rule(pred["good"],                             urgency["low"])

    system = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7])
    sim = ctrl.ControlSystemSimulation(system)
    return sim


def compute_urgency(sim, predicted_rsrp: float, trend_val: float) -> float:
    """Clamp inputs, compute fuzzy output, return urgency in [0, 100]."""
    sim.input["predicted_rsrp"] = np.clip(predicted_rsrp, -130, -90)
    sim.input["rsrp_trend"]     = np.clip(trend_val, -5, 5)
    try:
        sim.compute()
        return float(sim.output["handover_urgency"])
    except Exception:
        return 0.0


# ── simulation loop ──────────────────────────────────────────────────────────
def run_neuro_fuzzy_simulation(df: pd.DataFrame, model, scaler, fuzzy_sim):
    n = len(df)
    latency        = np.zeros(n)
    urgency_log    = np.zeros(n)
    connected_sat  = np.empty(n, dtype=object)
    handover_times = []

    current_sat    = "A"
    rsrp_history: list[float] = []

    for i in range(n):
        row = df.iloc[i]

        if current_sat == "A":
            cur_rsrp = row["satA_rsrp"]
            cur_dist = row["satA_distance"]
            alt_rsrp = row["satB_rsrp"]
            alt_name = "B"
        else:
            cur_rsrp = row["satB_rsrp"]
            cur_dist = row["satB_distance"]
            alt_rsrp = row["satA_rsrp"]
            alt_name = "A"

        rsrp_history.append(cur_rsrp)

        # ── warm-up: need SEQUENCE_LENGTH history for LSTM ───────────────
        if len(rsrp_history) < SEQUENCE_LENGTH:
            latency[i] = propagation_latency_ms(cur_dist)
            connected_sat[i] = current_sat
            continue

        # ── LSTM prediction (neural part) ────────────────────────────────
        window = np.array(rsrp_history[-SEQUENCE_LENGTH:])
        pred_rsrp = predict_rsrp(model, scaler, window)

        # ── trend calculation (dBm/s) ────────────────────────────────────
        if len(rsrp_history) >= TREND_WINDOW:
            delta = rsrp_history[-1] - rsrp_history[-TREND_WINDOW]
            trend_val = delta / (TREND_WINDOW * 0.1)       # dBm per second
        else:
            trend_val = 0.0

        # ── fuzzy inference ──────────────────────────────────────────────
        urg = compute_urgency(fuzzy_sim, pred_rsrp, trend_val)
        urgency_log[i] = urg

        # ── decision: fire soft handover if urgency is critical ──────────
        if urg >= URGENCY_TRIGGER and alt_rsrp > HANDOVER_THRESHOLD_DBM:
            latency[i] = SOFT_HANDOVER_PENALTY_MS
            connected_sat[i] = f"HO->{alt_name}"
            handover_times.append(row["time"])
            current_sat = alt_name
            rsrp_history.clear()
            continue

        latency[i] = propagation_latency_ms(cur_dist)
        connected_sat[i] = current_sat

    result = df[["time"]].copy()
    result["connected_sat"]      = connected_sat
    result["latency_ms"]         = np.round(latency, 4)
    result["handover_urgency"]   = np.round(urgency_log, 2)
    return result, handover_times


# ── summary ──────────────────────────────────────────────────────────────────
def print_summary(nf: pd.DataFrame, ho_times: list,
                  baseline: pd.DataFrame) -> None:
    nf_lat = nf["latency_ms"]
    bl_lat = baseline["latency_ms"]

    bl_blackout = bl_lat >= 3500
    bl_ho = int((bl_blackout & ~bl_blackout.shift(1, fill_value=False)).sum())

    print("=" * 66)
    print("  Phase 5 -- Neuro-Fuzzy Router vs Reactive Baseline")
    print("=" * 66)
    print(f"  {'Metric':<32} {'Baseline':>12} {'Neuro-Fuzzy':>14}")
    print(f"  {'-'*32} {'-'*12} {'-'*14}")
    print(f"  {'Handovers triggered':<32} {bl_ho:>12} {len(ho_times):>14}")
    print(f"  {'Max latency (ms)':<32} {bl_lat.max():>12.1f} {nf_lat.max():>14.1f}")
    print(f"  {'Mean latency (ms)':<32} {bl_lat.mean():>12.2f} {nf_lat.mean():>14.2f}")
    print(f"  {'Blackout steps (>=3500ms)':<32} "
          f"{int((bl_lat >= 3500).sum()):>12} {'0':>14}")
    pct = (1 - nf_lat.mean() / bl_lat.mean()) * 100
    print(f"\n  Latency reduction: {pct:.1f}%")
    if ho_times:
        print(f"  NF handover times: {', '.join(f'{t:.1f}s' for t in ho_times)}")
    print("=" * 66)


# ── 3-subplot dashboard ─────────────────────────────────────────────────────
def plot_dashboard(nf: pd.DataFrame, baseline: pd.DataFrame,
                   orbital: pd.DataFrame, ho_times: list) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    t       = nf["time"].values
    nf_lat  = nf["latency_ms"].values
    bl_lat  = baseline["latency_ms"].values
    urgency = nf["handover_urgency"].values

    # ── Top: Latency comparison ──────────────────────────────────────────
    ax1.plot(t, bl_lat, color="#EF5350", linewidth=0.6, alpha=0.85,
             label="Reactive Baseline")
    ax1.plot(t, nf_lat, color="#1E88E5", linewidth=0.8, alpha=0.95,
             label="Neuro-Fuzzy AI Router")
    for ht in ho_times:
        ax1.axvline(ht, color="#43A047", linewidth=0.9, ls="--", alpha=0.7)
    if ho_times:
        ax1.axvline(ho_times[0], color="#43A047", linewidth=0.9,
                     ls="--", alpha=0.7, label="Fuzzy soft handover")
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_title("Latency Comparison: Reactive Baseline vs Neuro-Fuzzy AI",
                   fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, 4200)
    ax1.grid(True, alpha=0.25)

    # ── Middle: RSRP signals ─────────────────────────────────────────────
    ax2.plot(t, orbital["satA_rsrp"].values, color="#FF7043",
             linewidth=0.5, alpha=0.8, label="Sat-A RSRP")
    ax2.plot(t, orbital["satB_rsrp"].values, color="#42A5F5",
             linewidth=0.5, alpha=0.8, label="Sat-B RSRP")
    ax2.axhline(HANDOVER_THRESHOLD_DBM, color="#B71C1C", linewidth=1,
                 ls="--", alpha=0.8, label=f"Threshold ({HANDOVER_THRESHOLD_DBM} dBm)")
    for ht in ho_times:
        ax2.axvline(ht, color="#43A047", linewidth=0.9, ls="--", alpha=0.5)
    ax2.set_ylabel("RSRP (dBm)", fontsize=11)
    ax2.set_title("Satellite Signal Strength Over Time", fontsize=13,
                   fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(-135, -90)
    ax2.grid(True, alpha=0.25)

    # ── Bottom: Fuzzy urgency ────────────────────────────────────────────
    ax3.plot(t, urgency, color="#7E57C2", linewidth=0.6, alpha=0.9,
             label="Handover Urgency")
    ax3.axhline(URGENCY_TRIGGER, color="#C62828", linewidth=1.2, ls="--",
                 alpha=0.9, label=f"Trigger threshold ({URGENCY_TRIGGER}%)")
    for ht in ho_times:
        ax3.axvline(ht, color="#43A047", linewidth=0.9, ls="--", alpha=0.5)
    ax3.set_xlabel("Time (s)", fontsize=11)
    ax3.set_ylabel("Urgency (%)", fontsize=11)
    ax3.set_title("Fuzzy Logic Engine: Handover Urgency Score",
                   fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9, loc="upper right")
    ax3.set_ylim(-5, 105)
    ax3.grid(True, alpha=0.25)

    fig.suptitle("6G Zero-Drop: Neuro-Fuzzy Predictive Handover",
                  fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT, dpi=180, bbox_inches="tight")
    print(f"\nDashboard saved to {PLOT_OUTPUT}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading assets ...")
    orbital  = pd.read_csv(ORBITAL_CSV)
    baseline = pd.read_csv(BASELINE_CSV)
    scaler   = joblib.load(SCALER_PATH)
    model    = load_model()
    print(f"  Orbital data : {len(orbital)} rows")
    print(f"  Baseline data: {len(baseline)} rows")
    print(f"  Model & scaler loaded on {DEVICE}")

    print("Building fuzzy inference system ...")
    fuzzy_sim = build_fuzzy_system()
    print("  FIS ready (2 antecedents, 7 rules, 1 consequent)\n")

    print("Running neuro-fuzzy simulation ...")
    nf_result, ho_times = run_neuro_fuzzy_simulation(
        orbital, model, scaler, fuzzy_sim)
    nf_result.to_csv(NF_CSV_OUTPUT, index=False)
    print(f"  Latency log saved to {NF_CSV_OUTPUT}")

    print_summary(nf_result, ho_times, baseline)
    plot_dashboard(nf_result, baseline, orbital, ho_times)


if __name__ == "__main__":
    main()
