"""
Phase 3 -- AI Predictor Engine (LSTM Trainer)
==============================================
Trains a lightweight PyTorch LSTM to predict the RSRP value 3 seconds
(30 time-steps) into the future, given the past 5 seconds (50 time-steps)
of signal history.

Outputs
-------
* satellite_lstm.pth   -- trained model weights
* rsrp_scaler.joblib   -- fitted MinMaxScaler for inverse-transform in Phase 4
"""

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ── hyperparameters ──────────────────────────────────────────────────────────
SEQUENCE_LENGTH     = 50        # look-back window  (5 s at dt=0.1)
PREDICTION_HORIZON  = 30        # forecast distance  (3 s at dt=0.1)
HIDDEN_SIZE         = 64
NUM_LAYERS          = 2
BATCH_SIZE          = 64
LEARNING_RATE       = 1e-3
EPOCHS              = 15
TRAIN_SPLIT         = 0.8

CSV_INPUT       = "orbital_data.csv"
MODEL_OUTPUT    = "satellite_lstm.pth"
SCALER_OUTPUT   = "rsrp_scaler.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── model ────────────────────────────────────────────────────────────────────
class SatelliteLSTM(nn.Module):
    def __init__(self, input_size: int = 1,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)             # (batch, seq, hidden)
        out = self.fc(out[:, -1, :])      # last time-step → single value
        return out.squeeze(-1)


# ── data helpers ─────────────────────────────────────────────────────────────
def load_and_scale(csv_path: str):
    """Load RSRP columns, stack into 1-D, scale to [-1, 1]."""
    df = pd.read_csv(csv_path)
    rsrp_a = df["satA_rsrp"].values
    rsrp_b = df["satB_rsrp"].values
    raw = np.concatenate([rsrp_a, rsrp_b])           # double the training data

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(raw.reshape(-1, 1)).flatten()
    return scaled, scaler


def make_sequences(data: np.ndarray):
    """Sliding window: X = [i : i+SEQ], Y = value at i + SEQ + HORIZON."""
    xs, ys = [], []
    limit = len(data) - SEQUENCE_LENGTH - PREDICTION_HORIZON
    for i in range(limit):
        xs.append(data[i : i + SEQUENCE_LENGTH])
        ys.append(data[i + SEQUENCE_LENGTH + PREDICTION_HORIZON])
    X = np.array(xs, dtype=np.float32)[:, :, np.newaxis]   # (N, 50, 1)
    Y = np.array(ys, dtype=np.float32)                      # (N,)
    return X, Y


# ── training ─────────────────────────────────────────────────────────────────
def train():
    print("Loading data ...")
    scaled, scaler = load_and_scale(CSV_INPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    print(f"  Scaler saved to {SCALER_OUTPUT}")
    print(f"  Total scaled samples: {len(scaled)}")

    print("Building sequences ...")
    X, Y = make_sequences(scaled)
    n_train = int(len(X) * TRAIN_SPLIT)
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]
    print(f"  Training sequences : {len(X_train)}")
    print(f"  Validation sequences: {len(X_val)}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(Y_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = SatelliteLSTM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining on {DEVICE}  |  {EPOCHS} epochs\n")
    print(f"{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}")
    print("-" * 35)

    for epoch in range(1, EPOCHS + 1):
        # ── train ────────────────────────────────────────────────────────
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            train_n += len(xb)

        # ── validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * len(xb)
                val_n += len(xb)

        train_loss = train_loss_sum / train_n
        val_loss   = val_loss_sum / val_n
        print(f"{epoch:5d}  {train_loss:12.6f}  {val_loss:12.6f}")

    torch.save(model.state_dict(), MODEL_OUTPUT)
    print(f"\nModel saved to {MODEL_OUTPUT}")
    print(f"Final validation loss (MSE, scaled): {val_loss:.6f}")


if __name__ == "__main__":
    train()
