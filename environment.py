"""
Phase 1 – Physics & Traffic Simulator
======================================
Generates synthetic RSRP telemetry for two LEO satellites passing over a
stationary ground user.  Output: orbital_data.csv (10 000+ time-steps).

Physics assumptions
-------------------
* 2-D flat-earth approximation (x-axis ground track + fixed altitude).
* LEO altitude  h = 500 km, orbital velocity v = 7.5 km/s.
* Carrier frequency f = 28 GHz (mmWave 5G/6G NR band).
* Tx EIRP = 70 dBm (10 W Tx + 30 dB beamforming/antenna gain).
* Atmospheric / fading noise: additive Gaussian, sigma = 1.5 dB.
* Horizon mask: RSRP clamped to -130 dBm when elevation < 0 deg.
"""

import numpy as np
import pandas as pd

# ── physical constants & orbital parameters ──────────────────────────────────
EARTH_RADIUS_KM = 6_371.0          # mean Earth radius
ALTITUDE_KM     = 500.0            # LEO orbit height
VELOCITY_KMS    = 7.5              # ground-track velocity (km/s)
FREQ_GHZ        = 28.0             # carrier frequency
TX_POWER_DBM    = 70.0             # satellite EIRP (10 W Tx + 30 dB antenna gain)
NOISE_STD_DB    = 1.5              # atmospheric / fading noise sigma
RSRP_FLOOR_DBM  = -130.0           # below-horizon noise floor

# ── simulation timing ────────────────────────────────────────────────────────
DT              = 0.1              # time-step (seconds)
N_STEPS         = 10_000           # total samples  → 1 000 s of simulation

# ── satellite initial x-positions (km) ───────────────────────────────────────
# Sat-A starts directly overhead (x = 0).
# Sat-B trails by 1 200 km so its coverage window overlaps Sat-A's departure.
# Geometric horizon distance ≈ sqrt((R+h)² − R²) ≈ 2 573 km one-sided,
# so 1 200 km spacing gives a comfortable overlap region.
SAT_A_X0 = 0.0
SAT_B_X0 = 1_200.0


def horizon_distance_km(h: float = ALTITUDE_KM) -> float:
    """Max ground-track range at which a satellite is above 0° elevation."""
    return np.sqrt((EARTH_RADIUS_KM + h) ** 2 - EARTH_RADIUS_KM ** 2)


def compute_rsrp(distance_km: np.ndarray,
                 elevation_ok: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Compute RSRP (dBm) from slant range using the simplified FSPL model:
        FSPL = 20·log10(d_km) + 20·log10(f_GHz) + 92.45
        RSRP = Tx_EIRP − FSPL + noise
    Below-horizon samples are clamped to RSRP_FLOOR_DBM.
    """
    fspl = 20.0 * np.log10(distance_km) + 20.0 * np.log10(FREQ_GHZ) + 92.45
    noise = rng.normal(0.0, NOISE_STD_DB, size=distance_km.shape)
    rsrp = TX_POWER_DBM - fspl + noise
    rsrp = np.where(elevation_ok, rsrp, RSRP_FLOOR_DBM)
    return rsrp


def generate_orbital_data(seed: int = 42) -> pd.DataFrame:
    """Run the full simulation and return a DataFrame of telemetry."""
    rng = np.random.default_rng(seed)

    time = np.arange(N_STEPS) * DT                       # seconds

    # horizontal positions (satellites move in –x direction)
    x_a = SAT_A_X0 - VELOCITY_KMS * time
    x_b = SAT_B_X0 - VELOCITY_KMS * time

    # slant (Euclidean) distance to user at origin
    d_a = np.sqrt(x_a ** 2 + ALTITUDE_KM ** 2)
    d_b = np.sqrt(x_b ** 2 + ALTITUDE_KM ** 2)

    # horizon mask – satellite visible only while |x| < horizon distance
    h_dist = horizon_distance_km()
    elev_ok_a = np.abs(x_a) < h_dist
    elev_ok_b = np.abs(x_b) < h_dist

    # RSRP via FSPL + noise
    rsrp_a = compute_rsrp(d_a, elev_ok_a, rng)
    rsrp_b = compute_rsrp(d_b, elev_ok_b, rng)

    df = pd.DataFrame({
        "time":          np.round(time, 2),
        "satA_distance": np.round(d_a, 4),
        "satA_rsrp":     np.round(rsrp_a, 2),
        "satB_distance": np.round(d_b, 4),
        "satB_rsrp":     np.round(rsrp_b, 2),
    })
    return df


def main() -> None:
    df = generate_orbital_data()
    csv_path = "orbital_data.csv"
    df.to_csv(csv_path, index=False)

    # ── quick sanity summary ─────────────────────────────────────────────────
    print(f"Saved {len(df)} rows to {csv_path}\n")
    print("Simulation parameters")
    print(f"  Altitude          : {ALTITUDE_KM} km")
    print(f"  Orbital velocity  : {VELOCITY_KMS} km/s")
    print(f"  Carrier frequency : {FREQ_GHZ} GHz")
    print(f"  Tx EIRP           : {TX_POWER_DBM} dBm")
    print(f"  Time step         : {DT} s  ({N_STEPS} steps = {N_STEPS * DT:.0f} s)")
    print(f"  Horizon distance  : {horizon_distance_km():.1f} km")
    print(f"  Sat-A x0           : {SAT_A_X0} km (overhead)")
    print(f"  Sat-B x0           : {SAT_B_X0} km (trailing)\n")

    print("RSRP statistics (dBm)")
    for label, col in [("Sat-A", "satA_rsrp"), ("Sat-B", "satB_rsrp")]:
        visible = df[df[col] > RSRP_FLOOR_DBM][col]
        print(f"  {label}  min={visible.min():.1f}  max={visible.max():.1f}  "
              f"mean={visible.mean():.1f}  (visible steps: {len(visible)})")

    print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")
    print(f"\nLast  5 rows:\n{df.tail().to_string(index=False)}")


if __name__ == "__main__":
    main()
