"""
Phase 2 -- Baseline "Dumb" Reactive Router
===========================================
Simulates a traditional Break-Before-Make handover protocol.

When the connected satellite's RSRP drops below HANDOVER_THRESHOLD the link
is severed and a full search-authenticate-connect cycle is performed on the
alternate satellite, costing HANDOVER_PENALTY_MS of dead time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── tunables ─────────────────────────────────────────────────────────────────
HANDOVER_THRESHOLD_DBM = -110.0     # RSRP level that triggers a drop
HANDOVER_PENALTY_MS    = 3500.0     # blackout duration during handover
DT                     = 0.1        # time-step size (seconds) -- matches Phase 1
PENALTY_STEPS          = int(HANDOVER_PENALTY_MS / (DT * 1000))  # 35 steps
COOLDOWN_STEPS         = 100        # 10 s dwell on new satellite before allowing next HO
SPEED_OF_LIGHT_KMS     = 3e5        # km/s
BASE_PROCESSING_MS     = 20.0       # flat processing overhead

CSV_INPUT  = "orbital_data.csv"
CSV_OUTPUT = "reactive_latency.csv"
PLOT_OUTPUT = "phase2_baseline_latency.png"


def propagation_latency_ms(distance_km: float) -> float:
    """Round-trip propagation delay + base processing time."""
    return (distance_km / SPEED_OF_LIGHT_KMS) * 2.0 * 1000.0 + BASE_PROCESSING_MS


def run_reactive_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through every time-step and decide:
      - If connected and signal OK  → log normal propagation latency.
      - If signal drops below threshold → trigger handover, start penalty.
      - During penalty window (35 steps) → log 3 500 ms blackout latency.
    """
    n = len(df)
    latency       = np.zeros(n)
    connected_sat = np.empty(n, dtype=object)

    current_sat        = "A"
    handover_remaining = 0          # penalty countdown
    cooldown_remaining = 0          # post-handover dwell timer
    handover_count     = 0
    target_sat         = "B"        # alternate satellite after current HO

    rsrp_floor = -130.0             # must match environment.py RSRP_FLOOR_DBM

    for i in range(n):
        row = df.iloc[i]

        # ── inside a handover blackout ───────────────────────────────────
        if handover_remaining > 0:
            latency[i] = HANDOVER_PENALTY_MS
            connected_sat[i] = "handover"
            handover_remaining -= 1

            if handover_remaining == 0:
                current_sat = target_sat
                cooldown_remaining = COOLDOWN_STEPS
            continue

        # ── tick down cooldown (still connected normally) ────────────────
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # ── read current satellite's RSRP and distance ───────────────────
        if current_sat == "A":
            rsrp     = row["satA_rsrp"]
            dist     = row["satA_distance"]
            alt_rsrp = row["satB_rsrp"]
            alt_name = "B"
        else:
            rsrp     = row["satB_rsrp"]
            dist     = row["satB_distance"]
            alt_rsrp = row["satA_rsrp"]
            alt_name = "A"

        # ── trigger handover only if alternate satellite is visible ──────
        if (rsrp < HANDOVER_THRESHOLD_DBM
                and cooldown_remaining == 0
                and alt_rsrp > rsrp_floor):
            target_sat = alt_name
            handover_remaining = PENALTY_STEPS
            handover_count += 1
            latency[i] = HANDOVER_PENALTY_MS
            connected_sat[i] = "handover"
            continue

        latency[i] = propagation_latency_ms(dist)
        connected_sat[i] = current_sat

    result = df[["time"]].copy()
    result["connected_sat"] = connected_sat
    result["latency_ms"]    = np.round(latency, 4)
    return result, handover_count


def print_summary(result: pd.DataFrame, handover_count: int) -> None:
    lat = result["latency_ms"]
    normal = lat[lat < HANDOVER_PENALTY_MS]

    print("=" * 60)
    print("  Phase 2 -- Reactive Router Summary")
    print("=" * 60)
    print(f"  Total time-steps       : {len(result)}")
    print(f"  Handovers triggered    : {handover_count}")
    print(f"  Max latency            : {lat.max():.1f} ms")
    print(f"  Mean latency (overall) : {lat.mean():.2f} ms")
    if len(normal) > 0:
        print(f"  Mean latency (normal)  : {normal.mean():.2f} ms")
    print(f"  Steps in blackout      : {(lat >= HANDOVER_PENALTY_MS).sum()}"
          f"  ({(lat >= HANDOVER_PENALTY_MS).mean() * 100:.1f}%)")
    print("=" * 60)


def plot_latency(result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    time = result["time"].values
    lat  = result["latency_ms"].values

    normal_mask   = lat < HANDOVER_PENALTY_MS
    blackout_mask = ~normal_mask

    ax.plot(time[normal_mask], lat[normal_mask],
            color="#2196F3", linewidth=0.4, alpha=0.8, label="Normal latency")

    ax.scatter(time[blackout_mask], lat[blackout_mask],
               color="#F44336", s=8, zorder=5, label=f"Handover blackout ({HANDOVER_PENALTY_MS:.0f} ms)")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Phase 2 -- Reactive Router: Latency vs Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(bottom=0, top=HANDOVER_PENALTY_MS * 1.15)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT, dpi=150)
    print(f"\nPlot saved to {PLOT_OUTPUT}")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CSV_INPUT)
    print(f"Loaded {len(df)} rows from {CSV_INPUT}")

    result, handover_count = run_reactive_simulation(df)
    result.to_csv(CSV_OUTPUT, index=False)
    print(f"Latency log saved to {CSV_OUTPUT}")

    print_summary(result, handover_count)
    plot_latency(result)


if __name__ == "__main__":
    main()
