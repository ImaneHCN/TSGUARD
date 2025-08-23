import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import models.sim_helper as helper
import streamlit as st
from utils.config import DEFAULT_VALUES
import time
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
from pathlib import Path
import os, shutil
import torch.optim as optim
from torch.utils.data import DataLoader
import io
import json
import sys
import yaml
from PriSTI.main_model import PriSTI_aqi36
from typing import List, Tuple, Optional

def get_base64_icon(path):
    icon_bytes = Path(path).read_bytes()
    b64_str = base64.b64encode(icon_bytes).decode()
    return f"data:image/png;base64,{b64_str}"

# --- Icon spec (centered) ---
ICON_URL = get_base64_icon("images/captor_icon.png")  # you already have this helper

ICON_W, ICON_H = 128, 128  # keep high-res for crispness
ICON_SPEC = {
    "url": ICON_URL,
    "width": ICON_W,
    "height": ICON_H,
    "anchorX": ICON_W // 2,   # center the icon at the point
    "anchorY": ICON_H // 2
}

def init_sensor_map(latlng_df):
    center_lat = float(latlng_df["latitude"].mean()) if len(latlng_df) else 0.0
    center_lon = float(latlng_df["longitude"].mean()) if len(latlng_df) else 0.0

    lats = latlng_df["latitude"].astype(float).tolist()
    lons = latlng_df["longitude"].astype(float).tolist()

    fig = go.Figure(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="markers+text",
        text=[""] * len(lats),
        marker=dict(size=36, opacity=0.1, color=["#2ecc71"] * len(lats), symbol="circle"),
        customdata=[[str(sid), "NA", "Imputed"] for sid in latlng_df["sensor_id"].astype(str)],
        hovertemplate="<b>Sensor</b>: %{customdata[0]}<br>"
                      "<b>Value</b>: %{customdata[1]}<br>"
                      "<b>Status</b>: %{customdata[2]}<extra></extra>",
        name="Sensors"
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",             # English labels
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11
        ),
        uirevision="keep",                      # keep camera + UI state
        transition=dict(duration=0),            # no animation
        margin=dict(l=10, r=10, t=10, b=10),
        height=480,
        showlegend=False,
    )
    return fig


def positions_to_df(positions):
    if isinstance(positions, pd.DataFrame):
        df = positions.rename(columns={"lat": "latitude", "lng": "longitude", "lon": "longitude"}).copy()
        if "sensor_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "sensor_id"})
        return df[["sensor_id", "latitude", "longitude"]]
    rows = [{"sensor_id": str(k), "latitude": float(v[1]), "longitude": float(v[0])}
            for k, v in positions.items()]
    return pd.DataFrame(rows, columns=["sensor_id", "latitude", "longitude"])


def make_sensor_map_plotly(latlng_df, values_by_sensor, is_real_by_sensor):
    """Scattermapbox: green=real, red=imputed."""
    lats, lons, colors, texts = [], [], [], []
    for _, r in latlng_df.iterrows():
        sid = str(r["sensor_id"])
        val = values_by_sensor.get(sid)
        real = bool(is_real_by_sensor.get(sid, False))
        lats.append(float(r["latitude"]))
        lons.append(float(r["longitude"]))
        colors.append("green" if real else "red")
        texts.append(f"<b>Sensor</b>: {sid}<br><b>Value</b>: {val if val is not None else 'NA'}"
                     f"<br><b>Status</b>: {'Real' if real else 'Imputed'}")

    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons, mode="markers+text",
        marker=dict(size=14, color=colors, opacity=0.90),
        text=[values_by_sensor.get(str(sid), "") for sid in latlng_df["sensor_id"]],
        textposition="top center",
        hovertext=texts, hoverinfo="text",
        name="Sensors"
    ))

    if len(latlng_df):
        center_lat = float(latlng_df["latitude"].mean())
        center_lon = float(latlng_df["longitude"].mean())
    else:
        center_lat, center_lon = 0.0, 0.0

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=480,
        showlegend=False,
        title=None,
    )
    return fig
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: (B, N), adj: (N, N) dense
        # aggregate neighbors: X @ Â^T  (Â is symmetric, so T doesn't matter)
        agg = x.matmul(adj.t())
        out = agg.matmul(self.weight)          # (B, N) @ (N, F_out) -> (B, F_out)
        return out

class GCNLSTMImputer(nn.Module):
    def __init__(self, adj, num_nodes, in_features, gcn_hidden, lstm_hidden, out_features):
        super(GCNLSTMImputer, self).__init__()
        self.adj = adj
        self.gcn = GraphConvolution(in_features, gcn_hidden)
        self.lstm = nn.LSTM(gcn_hidden, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        gcn_outputs = []
        for t in range(seq_len):
            # Apply GCN at each time step
            gcn_out = self.relu(self.gcn(x[:, t, :], self.adj))
            gcn_outputs.append(gcn_out.unsqueeze(1))

        # Concatenate GCN outputs along the sequence dimension
        gcn_sequence = torch.cat(gcn_outputs, dim=1)  # (batch_size, seq_len, gcn_hidden)

        # Feed sequence to LSTM
        lstm_out, _ = self.lstm(gcn_sequence)  # (batch_size, seq_len, lstm_hidden)

        # Pass LSTM output to the final fully connected layer
        output = self.fc(lstm_out)  # (batch_size, seq_len, out_features) which is num_nodes

        return output

class SpatioTemporalDataset(Dataset):
    def __init__(self, X, y, mask, seq_len=36):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.mask = torch.FloatTensor(mask)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],
            self.y[idx : idx + self.seq_len],
            self.mask[idx : idx + self.seq_len],
        )

criterion = nn.MSELoss(reduction='none')

def masked_loss(outputs, targets, mask):
    loss = criterion(outputs, targets)
    masked_loss = loss * mask
    # We only want the average over the non-zero elements of the mask
    return torch.sum(masked_loss) / torch.sum(mask)

def plot_sliding_custom_chart(sliding_df, sstates, sensor_cols):
    fig = go.Figure()

    STATION_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    for i, col in enumerate(sensor_cols):
        color_real = STATION_COLORS[i % len(STATION_COLORS)]
        timestamps = sliding_df["datetime"].tolist()
        values = sliding_df[col].tolist()
        states = sstates[col]  # Assume sstates is a dict of col -> list of booleans

        segment_x = []
        segment_y = []
        segment_color = color_real

        def add_segment():
            if len(segment_x) >= 2:
                fig.add_trace(go.Scatter(
                    x=segment_x,
                    y=segment_y,
                    mode='lines+markers',
                    line=dict(color=segment_color, width=2),
                    showlegend=False
                ))

        for j in range(len(values)):
            is_real = states[j]
            val = values[j]
            if pd.isna(val):
                continue  # skip NaNs

            if len(segment_x) == 0:
                # start new segment
                segment_color = color_real if is_real else "red"
                segment_x.append(timestamps[j])
                segment_y.append(val)
            else:
                same_state = (segment_color == (color_real if is_real else "red"))
                if same_state:
                    segment_x.append(timestamps[j])
                    segment_y.append(val)
                else:
                    # switch segment
                    add_segment()
                    segment_x = [timestamps[j - 1], timestamps[j]]
                    segment_y = [values[j - 1], val]
                    segment_color = color_real if is_real else "red"

        # Add last segment
        add_segment()

    fig.update_layout(
        title="10-Step Sliding Window",
        xaxis_title="Time",
        yaxis_title="Sensor Value",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# device = (
#     torch.device("cuda")
#     if torch.cuda.is_available()
#     else torch.device("mps")
#     if torch.backends.mps.is_available()
#     else torch.device("cpu")
# )
device = torch.device("cpu")

def haversine_distance(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Radius of Earth in Km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def create_adjacency_matrix(latlng_df, threshold_type='gaussian', sigma_sq_ratio=0.1):
    num_sensors = len(latlng_df)
    dist_matrix = np.zeros((num_sensors, num_sensors), dtype=np.float64)

    for i in range(num_sensors):
        for j in range(i, num_sensors):
            lat1, lon1 = latlng_df.iloc[i]['latitude'],  latlng_df.iloc[i]['longitude']
            lat2, lon2 = latlng_df.iloc[j]['latitude'],  latlng_df.iloc[j]['longitude']
            d = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist_matrix[j, i] = d

    if threshold_type == 'gaussian':
        sigma = dist_matrix.std()
        sigma_sq = (sigma * sigma) * sigma_sq_ratio
        sigma_sq = max(sigma_sq, 1e-6)  # <- avoid div-by-zero
        adj_matrix = np.exp(-np.square(dist_matrix) / sigma_sq)
    else:
        threshold = np.mean(dist_matrix) * 0.5
        adj_matrix = (dist_matrix <= threshold).astype(float)

    np.fill_diagonal(adj_matrix, 1.0)

    D = np.diag(np.sum(adj_matrix, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    adj_norm = D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    return torch.tensor(adj_norm, dtype=torch.float32)

# ---- TRAINING TSGUARD -------------------------------------------------
def train_model(
    tr: pd.DataFrame,          # ground_df (vérité terrain)
    df: pd.DataFrame,          # missing_df (avec trous)
    pf,                        # positions (DataFrame ou dict {id: (lon,lat) / {...}})
    epochs: int = 20,
    model_path: str = "gcn_lstm_imputer.pth",
    seq_len: int = 36,
    batch_size: int = 32,
    lr: float = 1e-3,
    sigma_sq_ratio: float = 0.1,
    device: torch.device = None,
):
    # ---------------- Helpers ----------------
    def _ensure_datetime_index(dfi: pd.DataFrame) -> pd.DataFrame:
        dfi = dfi.copy()
        if "datetime" in dfi.columns:
            dfi["datetime"] = pd.to_datetime(dfi["datetime"], errors="coerce")
            dfi = dfi.dropna(subset=["datetime"]).set_index("datetime")
        elif not isinstance(dfi.index, pd.DatetimeIndex):
            dfi.index = pd.to_datetime(dfi.index, errors="coerce")
            dfi = dfi[~dfi.index.isna()]
        dfi.index = dfi.index.floor("h")
        dfi = dfi[~dfi.index.duplicated(keep="first")]
        return dfi.sort_index()

    def _strip_leading_zeros_cols(dfi: pd.DataFrame) -> pd.DataFrame:
        dfi = dfi.copy()
        dfi.columns = [
            (str(c).lstrip("0") if str(c).isdigit() else str(c))
            for c in dfi.columns
        ]
        return dfi

    def _coerce_positions(pf_in) -> pd.DataFrame:
        import io, json as _json
        if isinstance(pf_in, pd.DataFrame):
            dfp = pf_in.copy()
            dfp = dfp.rename(columns={"lat": "latitude", "lng": "longitude", "lon": "longitude"})
            if "sensor_id" not in dfp.columns:
                dfp = dfp.reset_index().rename(columns={"index": "sensor_id"})
            need = {"sensor_id", "latitude", "longitude"}
            miss = need - set(dfp.columns)
            if miss:
                raise ValueError(f"Positions: colonnes manquantes {miss}. Colonnes reçues: {list(dfp.columns)}")
            dfp = dfp[list(need)]
        elif isinstance(pf_in, dict):
            rows = []
            for k, v in pf_in.items():
                sid = str(k)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    lon, lat = float(v[0]), float(v[1])
                elif isinstance(v, dict):
                    lat = v.get("latitude", v.get("lat"))
                    lon = v.get("longitude", v.get("lon", v.get("lng")))
                    if lat is None or lon is None:
                        raise ValueError(f"Dict position invalide pour {k}: {v}")
                    lat, lon = float(lat), float(lon)
                else:
                    raise ValueError(f"Type de valeur inconnu pour {k}: {type(v)}")
                rows.append({"sensor_id": sid, "latitude": lat, "longitude": lon})
            dfp = pd.DataFrame(rows, columns=["sensor_id", "latitude", "longitude"])
        elif hasattr(pf_in, "read"):
            raw = pf_in.read()
            try:
                dfp = pd.read_csv(io.BytesIO(raw))
                return _coerce_positions(dfp)
            except Exception:
                data = _json.loads(raw.decode("utf-8"))
                return _coerce_positions(data)
        else:
            from pathlib import Path as _Path
            if isinstance(pf_in, (str, _Path)):
                p = _Path(pf_in)
                if p.suffix.lower() in {".csv", ".txt"}:
                    dfp = pd.read_csv(p)
                    return _coerce_positions(dfp)
                else:
                    with open(p, "r", encoding="utf-8") as f:
                        data = _json.load(f)
                    return _coerce_positions(data)
            raise TypeError(f"type positions non supporté: {type(pf_in)}")

        dfp["sensor_id"] = dfp["sensor_id"].astype(str).str.strip()
        dfp["latitude"] = pd.to_numeric(dfp["latitude"], errors="coerce")
        dfp["longitude"] = pd.to_numeric(dfp["longitude"], errors="coerce")
        dfp = dfp.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        return dfp

    def _strip_zeros_safe(s: str) -> str:
        if s.isdigit():
            out = s.lstrip("0")
            return out if out != "" else "0"
        return s

    # ------------- Préparation DF -------------
    ground_df  = tr.copy()
    missing_df = df.copy()

    ground_df  = _ensure_datetime_index(ground_df)
    missing_df = _ensure_datetime_index(missing_df)

    ground_df  = _strip_leading_zeros_cols(ground_df)
    missing_df = _strip_leading_zeros_cols(missing_df)

    ground_df.columns  = [str(c).strip() for c in ground_df.columns]
    missing_df.columns = [str(c).strip() for c in missing_df.columns]

    latlng = _coerce_positions(pf).copy()
    latlng["sensor_id"] = latlng["sensor_id"].astype(str).str.strip()

    sid = latlng["sensor_id"]
    if sid.str.fullmatch(r"\d+").all():
        ids = sid.astype(int)
        if ids.min() == 0 and ids.max() == len(latlng) - 1 and len(set(ids)) == len(latlng):
            latlng = latlng.sort_values("sensor_id", key=lambda s: s.astype(int)).reset_index(drop=True)
            latlng["sensor_id"] = list(ground_df.columns)[:len(latlng)]
        else:
            latlng["sensor_id"] = latlng["sensor_id"].map(_strip_zeros_safe)

    sensor_cols = latlng["sensor_id"].astype(str).tolist()

    missing_in_ground  = sorted(set(sensor_cols) - set(ground_df.columns))
    missing_in_missing = sorted(set(sensor_cols) - set(missing_df.columns))
    if missing_in_ground or missing_in_missing:
        raise KeyError(
            "Colonnes capteurs manquantes après normalisation.\n"
            f"- Manquantes dans ground: {missing_in_ground}\n"
            f"- Manquantes dans missing: {missing_in_missing}\n"
            f"- Exemples colonnes ground: {list(ground_df.columns)[:8]}"
        )

    ground_df  = ground_df[sensor_cols]
    missing_df = missing_df[sensor_cols]

    common_idx = ground_df.index.intersection(missing_df.index)
    if common_idx.empty:
        raise ValueError("Aucun chevauchement temporel entre ground et missing.")
    ground_df  = ground_df.loc[common_idx].sort_index()
    missing_df = missing_df.loc[common_idx].sort_index()

    # ------------- Numpy & X imputé -------------
    ground_data  = ground_df.to_numpy(dtype=np.float32)
    missing_data = missing_df.to_numpy(dtype=np.float32)

    imputed_df   = missing_df.ffill().bfill()
    imputed_data = imputed_df.to_numpy(dtype=np.float32)

    loss_mask = np.where(np.isnan(missing_data) & np.isfinite(ground_data), 1.0, 0.0).astype(np.float32)

    # ------------- Splits & Scaler -------------
    months = missing_df.index.month
    train_month_list = [1, 2, 4, 5, 7, 8, 10, 11]
    valid_month_list = [2, 5, 8, 11]

    train_slice = np.isin(months, train_month_list)
    valid_slice = np.isin(months, valid_month_list)

    if len(common_idx) < 2 * seq_len:
        raise ValueError(f"Pas assez d'instants après alignement: {len(common_idx)} (< {2*seq_len}).")
    if not np.any(train_slice):
        raise ValueError("Train split vide (vérifie les mois présents dans tes données).")

    train_imputed = imputed_data[train_slice]
    min_val = float(np.nanmin(train_imputed))
    max_val = float(np.nanmax(train_imputed))
    denom   = (max_val - min_val) if (max_val - min_val) != 0 else 1.0

    scaler     = lambda x: (x - min_val) / denom
    inv_scaler = lambda x: x * denom + min_val

    from pathlib import Path as _P
    scaler_params = {"min_val": min_val, "max_val": max_val}
    scaler_json   = str(_P(model_path).with_suffix('')) + "_scaler.json"
    with open(scaler_json, "w") as f:
        json.dump(scaler_params, f, indent=2)

    X_norm = scaler(imputed_data)
    Y_norm = scaler(ground_data)
    Y_norm = np.nan_to_num(Y_norm, nan=0.0)

    X_train, y_train, m_train = X_norm[train_slice], Y_norm[train_slice], loss_mask[train_slice]
    X_val,   y_val,   m_val   = X_norm[valid_slice], Y_norm[valid_slice], loss_mask[valid_slice]

    # ------------- Datasets/Loaders -------------
    train_ds = SpatioTemporalDataset(X_train, y_train, m_train, seq_len=seq_len)
    val_ds   = SpatioTemporalDataset(X_val,   y_val,   m_val,   seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ------------- Modèle & Adj -------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adj = create_adjacency_matrix(latlng, threshold_type='gaussian', sigma_sq_ratio=sigma_sq_ratio).to(device)

    num_nodes = imputed_data.shape[1]
    model = GCNLSTMImputer(
        adj=adj,
        num_nodes=num_nodes,
        in_features=num_nodes,
        gcn_hidden=64,
        lstm_hidden=64,
        out_features=num_nodes,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------- Entraînement -------------
    try:
        pbar = st.progress(0.0)
        status = st.container()
        use_streamlit = True
    except Exception:
        use_streamlit = False

    model.train()
    for epoch in range(epochs):
        tot, n = 0.0, 0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = masked_loss(out, yb, mb)
            if torch.isfinite(loss) and torch.sum(mb) > 0:
                loss.backward()
                optimizer.step()
                tot += loss.item()
                n += 1

        # validation (FIX ICI: pas d'argument nommé 'out')
        model.eval()
        with torch.no_grad():
            vtot, vn = 0.0, 0
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                out = model(xb)
                vloss = masked_loss(out, yb, mb)  # <-- arguments positionnels
                if torch.isfinite(vloss) and torch.sum(mb) > 0:
                    vtot += vloss.item(); vn += 1
        model.train()

        train_loss = tot / max(n, 1)
        val_loss   = vtot / max(vn, 1)

        if use_streamlit:
            pbar.progress(int((epoch + 1) * 100 / max(epochs, 1)))
            with status:
                st.write(f"🔹 **Epoch {epoch+1}** | 📉 **Train:** `{train_loss:.4f}` | 🧪 **Val:** `{val_loss:.4f}`")
        else:
            print(f"Epoch {epoch+1}/{epochs} — train {train_loss:.4f} | val {val_loss:.4f}")

    # ------------- Sauvegarde & retour -------------
    torch.save(model.state_dict(), model_path)
    if use_streamlit:
        st.success(f"✅ Modèle sauvegardé: `{model_path}` ; Scaler: `{scaler_json}`")
    else:
        print(f"Saved model to {model_path} and scaler to {scaler_json}")

    return model


# -------------------- PRISTI Imputation functions -------------
# --- PriSTI integration helpers ----

@st.cache_resource(show_spinner=False)
def load_pristi_artifacts(CONFIG_PATH: str, WEIGHTS_PATH: str, MEANSTD_PK: str, device: torch.device):
    import yaml, pickle, numpy as np, torch
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}
    # Ensure diffusion → adj_file
    config.setdefault("diffusion", {})
    config["diffusion"].setdefault("adj_file", "AQI36")

    # Load mean/std
    with open(MEANSTD_PK, "rb") as f:
        meanstd = pickle.load(f)
    mean = np.asarray(meanstd[0], dtype=np.float32)
    std  = np.asarray(meanstd[1], dtype=np.float32)
    std_safe = np.where(std == 0, 1.0, std)

    # Load model
    from PriSTI.main_model import PriSTI_aqi36
    model = PriSTI_aqi36(config, device).to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, mean, std_safe

# these two mimic your Kaggle code (they read mean/std from session_state we’ll fill below)
def scale_window(x_2d: np.ndarray) -> np.ndarray:
    mean = st.session_state["pristi_mean"]
    std_safe = st.session_state["pristi_std"]
    return (x_2d - mean) / std_safe

def inv_scale_vec(x_1d: np.ndarray) -> np.ndarray:
    mean = st.session_state["pristi_mean"]
    std_safe = st.session_state["pristi_std"]
    return x_1d * std_safe + mean

def impute_window_with_pristi(
    missing_df: pd.DataFrame,
    sensor_cols: List[str],
    target_timestamp: pd.Timestamp,
    model: torch.nn.Module,
    device: torch.device,
    eval_len: int = 36,
    nsample: int = 100
) -> Tuple[pd.DataFrame, str]:
    """
    This is your tested Kaggle function, copied as-is (prints removed).
    It fills a (T=eval_len, N=len(sensor_cols)) window ending at target_timestamp.
    Returns (updated_df, "ok") or (original_df_copy, reason)
    """
    print(f"Imputing window with pristi ({nsample} samples)...")
    if "scale_window" not in globals() or "inv_scale_vec" not in globals():
        return missing_df.copy(), "Scaling functions not found."
    if list(missing_df.columns) != list(sensor_cols):
        return missing_df.copy(), "Columns mismatch."
    if target_timestamp not in missing_df.index:
        return missing_df.copy(), f"{target_timestamp} not in DataFrame index."

    end_loc = missing_df.index.get_loc(target_timestamp)
    if isinstance(end_loc, slice):
        return missing_df.copy(), "Ambiguous target timestamp."
    start_loc = end_loc - (eval_len - 1)
    if start_loc < 0:
        return missing_df.copy(), f"Not enough history (<{eval_len} rows)."

    time_index = missing_df.index[start_loc:end_loc + 1]
    filled_df = missing_df.ffill().bfill()

    window_filled = filled_df.iloc[start_loc:end_loc + 1].to_numpy(dtype=np.float32)
    window_orig   = missing_df.iloc[start_loc:end_loc + 1].to_numpy(dtype=np.float32)
    T, N = window_filled.shape

    # Mask (1=observed, 0=missing)
    mask_np = (~np.isnan(window_orig)).astype(np.float32)
    if not (mask_np == 0).any():
        return missing_df.copy(), "No missing values in window."

    # Scale and to tensors
    window_scaled = scale_window(window_filled)
    x_TN = torch.from_numpy(window_scaled).unsqueeze(0).to(device)  # (1,T,N)
    m_TN = torch.from_numpy(mask_np     ).unsqueeze(0).to(device)  # (1,T,N)
    x_NL = x_TN.permute(0, 2, 1).contiguous()  # (1,N,T)
    m_NL = m_TN.permute(0, 2, 1).contiguous()  # (1,N,T)

    # PriSTI API (inner model)
    inner = getattr(model, "model", getattr(model, "module", model))
    if not hasattr(inner, "get_side_info"):
        return missing_df.copy(), "PriSTI instance required."

    observed_tp = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)  # (1,T)
    side_info = inner.get_side_info(observed_tp, m_NL)

    itp_info = None
    if getattr(inner, "use_guide", False):
        coeffs = torch.zeros((1, N, T), device=device, dtype=torch.float32)
        itp_info = coeffs.unsqueeze(1)

    inner.eval()
    with torch.no_grad():
        try:
            y_pred = inner.impute(x_NL, m_NL, side_info, int(nsample), itp_info)
        except TypeError:
            y_pred = inner.impute(x_NL, m_NL, side_info, int(nsample))

    if not isinstance(y_pred, torch.Tensor):
        return missing_df.copy(), "Non-tensor output."

    # Reduce samples & align shape
    if y_pred.dim() == 4:
        if y_pred.shape[0] == nsample:
            y_pred = y_pred.mean(dim=0)
        elif y_pred.shape[1] == nsample:
            y_pred = y_pred.mean(dim=1)
        else:
            y_pred = y_pred.mean(dim=0)

    if y_pred.dim() != 3:
        return missing_df.copy(), f"Unexpected output rank: {y_pred.dim()}."

    pred3 = y_pred[0]
    if pred3.shape == (N, T):
        pred_scaled_NT = pred3
    elif pred3.shape == (T, N):
        pred_scaled_NT = pred3.transpose(0, 1).contiguous()
    else:
        return missing_df.copy(), f"Unexpected output shape {tuple(pred3.shape)}."

    pred_scaled_TN = pred_scaled_NT.transpose(0, 1).detach().cpu().numpy()  # (T,N)
    pred_unscaled_TN = np.vstack([inv_scale_vec(pred_scaled_TN[t, :]) for t in range(T)])

    # Merge into a copy and return
    updated_df = missing_df.copy()
    miss_mask_bool = (mask_np == 0)
    for t_idx in range(T):
        ts = time_index[t_idx]
        for n_idx in range(N):
            if miss_mask_bool[t_idx, n_idx]:
                sensor_name = sensor_cols[n_idx]
                updated_df.loc[ts, sensor_name] = float(pred_unscaled_TN[t_idx, n_idx])

    return updated_df, "ok"

def draw_full_time_series(global_df, sim_file, sensor_cols, sensor_color_map):
    fig = go.Figure()
    for col in sensor_cols:
        color = sensor_color_map[col]
        x_vals = global_df["datetime"]
        y_vals = global_df[col]

        segment_x, segment_y, segment_state = [], [], []
        for x, y in zip(x_vals, y_vals):
            if pd.isna(y):
                continue  # Skip missing entirely

            is_real = not pd.isna(sim_file.loc[x, col]) if x in sim_file.index else False

            segment_x.append(x)
            segment_y.append(y)
            segment_state.append(is_real)

        # Now construct segments based on changes in imputation state
        if len(segment_x) >= 2:
            for i in range(1, len(segment_x)):
                x_seg = [segment_x[i-1], segment_x[i]]
                y_seg = [segment_y[i-1], segment_y[i]]
                seg_is_real = segment_state[i-1] and segment_state[i]
                seg_color = color if seg_is_real else "red"

                fig.add_trace(go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode="lines+markers",
                    name=f"Sensor {col}",
                    line=dict(color=seg_color),
                    marker=dict(size=6, color=seg_color),
                    showlegend=False
                ))

    # Legends
    for col in sensor_cols:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=8, color=sensor_color_map[col]),
            legendgroup=col,
            showlegend=True,
            name=f"Sensor {col}"
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=8, color="red"),
        legendgroup="imputed",
        showlegend=True,
        name="Imputed Segment"
    ))
    #title = "Global Time Series",
    fig.update_layout(

        xaxis_title="Time",
        yaxis_title="Sensor Value",
        margin=dict(l=20, r=20, t=40, b=20),
        legend_title="Sensors"
    )
    return fig

def draw_full_time_series_with_mask(global_df, imputed_mask, sensor_cols, sensor_color_map):
    import plotly.graph_objects as go
    import pandas as pd

    fig = go.Figure()
    # segments colorés selon imputation
    for col in sensor_cols:
        base_color = sensor_color_map[col]
        x_vals = global_df["datetime"].tolist()
        y_vals = global_df[col].tolist()

        seg_x, seg_y, seg_imp = [], [], []
        for x, y in zip(x_vals, y_vals):
            if pd.isna(y):
                continue
            imputed = bool(imputed_mask.loc[x, col]) if (x in imputed_mask.index) else False
            seg_x.append(x); seg_y.append(y); seg_imp.append(imputed)
            if len(seg_x) >= 2:
                seg_color = "red" if any(seg_imp) else base_color
                fig.add_trace(go.Scatter(
                    x=seg_x, y=seg_y, mode="lines+markers",
                    line=dict(color=seg_color), marker=dict(size=6, color=seg_color),
                    showlegend=False
                ))
                seg_x, seg_y, seg_imp = [seg_x[-1]], [seg_y[-1]], [seg_imp[-1]]
        if len(seg_x) >= 2:
            seg_color = "red" if any(seg_imp) else base_color
            fig.add_trace(go.Scatter(
                x=seg_x, y=seg_y, mode="lines+markers",
                line=dict(color=seg_color), marker=dict(size=6, color=seg_color),
                showlegend=False
            ))

    # légendes
    for col in sensor_cols:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=8, color=sensor_color_map[col]),
            legendgroup=col, showlegend=True, name=f"Sensor {col}"
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=8, color="red"),
        legendgroup="imputed", showlegend=True, name="Imputed Segment"
    ))
    fig.update_layout(
        xaxis_title="Time", yaxis_title="Sensor Value",
        margin=dict(l=20, r=20, t=40, b=20), legend_title="Sensors"
    )
    return fig
def draw_full_time_series_with_mask_gap(global_df, imputed_mask, sensor_cols, sensor_color_map, gap_hours=12):
    fig = go.Figure()
    for col in sensor_cols:
        base_color = sensor_color_map[col]
        sub = (global_df[["datetime", col]]
               .dropna()
               .sort_values("datetime")
               .rename(columns={col: "value"}))
        if sub.empty:
            continue
        is_imp = lambda t: (t in imputed_mask.index) and bool(imputed_mask.loc[t, col])
        add_imputed_segments(fig, sub, is_imp, base_color, gap_hours=gap_hours)

    # legend
    for col in sensor_cols:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=8, color=sensor_color_map[col]),
            showlegend=True, name=f"Sensor {col}"
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=8, color="red"),
        showlegend=True, name="Imputed segment"
    ))
    fig.update_layout(
        xaxis_title="Time", yaxis_title="Sensor Value",
        margin=dict(l=20, r=20, t=40, b=20),
        legend_title="Sensors",
        uirevision="global",
    )
    return fig

def add_imputed_segments(fig, df_xy, mask_col_bool, base_color, gap_hours=6):
    """
    df_xy: DataFrame with columns ['datetime', 'value'] sorted by datetime, NaNs removed.
    mask_col_bool: function(datetime) -> bool  (returns True if the point is imputed)
    base_color: sensor color when not imputed
    gap_hours: break line if time gap exceeds this number of hours
    """
    GAP = pd.Timedelta(hours=gap_hours)

    prev_x = prev_y = prev_imp = None
    for x, y in zip(df_xy["datetime"], df_xy["value"]):
        imp = bool(mask_col_bool(x))

        if prev_x is not None:
            # break the line if the time gap is too large
            if x - prev_x > GAP:
                prev_x, prev_y, prev_imp = x, y, imp
                continue

            seg_color = "red" if (imp or prev_imp) else base_color
            fig.add_trace(go.Scatter(
                x=[prev_x, x],
                y=[prev_y, y],
                mode="lines+markers",
                line=dict(width=2, color=seg_color),
                marker=dict(size=6, color=seg_color),
                showlegend=False,
            ))
        prev_x, prev_y, prev_imp = x, y, imp

def draw_gauge_figure(sim_file, current_time, sensor_cols):
    green_min, green_max = DEFAULT_VALUES["gauge_green_min"], DEFAULT_VALUES["gauge_green_max"]
    yellow_min, yellow_max = DEFAULT_VALUES["gauge_yellow_min"], DEFAULT_VALUES["gauge_yellow_max"]
    red_min, red_max = DEFAULT_VALUES["gauge_red_min"], DEFAULT_VALUES["gauge_red_max"]

    if st.session_state.get('missing_value_thresholds'):
        thresholds = st.session_state['missing_value_thresholds']
        green_min, green_max = thresholds.get("Green", (green_min, green_max))
        yellow_min, yellow_max = thresholds.get("Yellow", (yellow_min, yellow_max))
        red_min, red_max = thresholds.get("Red", (red_min, red_max))

    sim_file_up_to_now = sim_file[sim_file.index <= current_time]
    total = sim_file_up_to_now[sensor_cols].size
    missed = sim_file_up_to_now[sensor_cols].isna().sum().sum()
    pmiss = (missed / total) * 100 if total > 0 else 0

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pmiss,
        title={"text": "Overall Missed Data (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if pmiss > red_max else "green"},
            "steps": [
                {"range": [green_min, green_max], "color": "lightgreen"},
                {"range": [yellow_min, yellow_max], "color": "yellow"},
                {"range": [red_min, red_max], "color": "red"}
            ]
        }
    ))
    gauge_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return gauge_fig

def predict_single_missing_value(
    historical_window: np.ndarray,
    target_sensor_index: int,
    model: torch.nn.Module,
    scaler: callable,
    inv_scaler: callable,
    device: torch.device
) -> float:
    """
    Predicts the next value for a single sensor using a historical window.

    Args:
        historical_window (np.ndarray): A 2D NumPy array of shape (seq_len, num_sensors)
                                        containing the recent historical data.
                                        This window should NOT contain NaNs.
        target_sensor_index (int): The integer index of the sensor whose value needs to be predicted.
        model (torch.nn.Module): The pre-trained and loaded GCN-LSTM model.
        scaler (callable): The function to normalize the data.
        inv_scaler (callable): The function to inverse-normalize the data.
        device (torch.device): The device (CPU or CUDA) to run the model on.

    Returns:
        float: The single imputed value for the target sensor at the next time step.
    """
    # --- 1. Input Validation ---
    #expected_seq_len = model.lstm.input_size // model.gcn.weight.shape[0] * model.lstm.input_size
    #print (expected_seq_len)
    # A bit of a complex way to get seq_len, let's assume it's known. A better way:
    # Let's say model config is available. For now, we'll rely on a fixed known value.
    EVAL_LENGTH = 24 # This must match the model's training sequence length
    
    #if historical_window.shape[0] != EVAL_LENGTH:
    #    raise ValueError(f"Historical window must have sequence length {EVAL_LENGTH}, but got {historical_window.shape[0]}")
    if np.isnan(historical_window).any():
        raise ValueError("The input 'historical_window' cannot contain NaN values. Please pre-fill it.")

    # --- 2. Preprocessing ---
    # Normalize the data
    normalized_window = scaler(historical_window)
    
    # Convert to a PyTorch tensor, add a batch dimension, and move to device
    input_tensor = torch.FloatTensor(normalized_window).unsqueeze(0).to(device)

    # --- 3. Model Inference ---
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for efficiency
        # The model predicts the *next* step for each step in the input sequence
        output_sequence = model(input_tensor)

    # --- 4. Post-processing and Output ---
    # We need the prediction corresponding to the *last* time step of the input window.
    # This gives us the predictions for all sensors at the next time step.
    # Shape: (1, seq_len, num_sensors) -> (num_sensors,)
    last_step_prediction_normalized = output_sequence[0, -1, :]

    # Inverse-transform the predictions back to the original scale
    all_sensor_predictions = inv_scaler(last_step_prediction_normalized.cpu().numpy())

    # Extract the prediction for our specific target sensor
    imputed_value = all_sensor_predictions[target_sensor_index]

    return float(imputed_value)
# ---------- helpers for ground + comparison plot ----------

@st.cache_data(show_spinner=False)
def _load_pristi_ground(path: str) -> pd.DataFrame:
    """Load PriSTI ground file -> datetime index; string sensor IDs."""
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    # normalize sensor-id-like columns to strings (e.g., keep zero padding)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.sort_index()
    # drop duplicate timestamps if any
    df = df[~df.index.duplicated(keep="first")]
    return df

def _cmp_timeseries_fig(
    time_index: pd.DatetimeIndex,
    sensors: list[str],
    tsg_block: pd.DataFrame,        # (len(T), 36) values from your stream (TSGuard)
    pristi_block: pd.DataFrame,     # (len(T), 36) values from PriSTI df
    ground_block: Optional[pd.DataFrame],
    miss_mask_block: pd.DataFrame,  # bools: originally missing (len(T), 36)
) -> go.Figure:
    """Build comparison figure: real line + green (TSG) & red (PriSTI) markers on imputed cells, with MSE in hover."""
    fig = go.Figure()

    T = len(time_index)

    for sid in sensors:
        x = time_index

        # Ground (real) line – thin grey
        if ground_block is not None and sid in ground_block.columns:
            y_real = ground_block[sid].to_numpy(dtype="float64")
            fig.add_trace(go.Scatter(
                x=x, y=y_real, mode="lines",
                line=dict(width=1, color="#7a7a7a"),
                name=f"{sid} real", legendgroup=f"{sid}_real", showlegend=False,
                hovertemplate="Sensor "+sid+"<br>%{x|%Y-%m-%d %H:%M}<br>Real: %{y:.2f}<extra></extra>",
            ))

        # TSG imputed markers (green)
        if sid in tsg_block.columns:
            y_tsg = tsg_block[sid].to_numpy(dtype="float64")
            mask  = miss_mask_block[sid].to_numpy(dtype=bool)
            # Only show markers on imputed cells; else None
            y_disp = [y if m else None for y, m in zip(y_tsg, mask)]
            if ground_block is not None and sid in ground_block.columns:
                g = ground_block[sid].to_numpy(dtype="float64")
                mse = (y_tsg - g) ** 2
                custom = np.stack([g, mse], axis=1)  # (T,2)
            else:
                custom = np.full((T, 2), np.nan)

            fig.add_trace(go.Scatter(
                x=x, y=y_disp, mode="markers",
                marker=dict(size=7, color="green"),
                name=f"{sid} TSGuard", legendgroup=f"{sid}_tsg", showlegend=False,
                customdata=custom,
                hovertemplate=(
                    "Sensor "+sid+"<br>%{x|%Y-%m-%d %H:%M}"
                    "<br>TSGuard: %{y:.2f}"
                    "<br>Real: %{customdata[0]:.2f}"
                    "<br>MSE: %{customdata[1]:.3f}<extra></extra>"
                ),
            ))

        # PriSTI imputed markers (red)
        if sid in pristi_block.columns:
            y_pri = pristi_block[sid].to_numpy(dtype="float64")
            mask  = miss_mask_block[sid].to_numpy(dtype=bool)
            y_disp = [y if m else None for y, m in zip(y_pri, mask)]
            if ground_block is not None and sid in ground_block.columns:
                g = ground_block[sid].to_numpy(dtype="float64")
                mse = (y_pri - g) ** 2
                custom = np.stack([g, mse], axis=1)
            else:
                custom = np.full((T, 2), np.nan)

            fig.add_trace(go.Scatter(
                x=x, y=y_disp, mode="markers",
                marker=dict(size=7, color="red"),
                name=f"{sid} PriSTI", legendgroup=f"{sid}_pri", showlegend=False,
                customdata=custom,
                hovertemplate=(
                    "Sensor "+sid+"<br>%{x|%Y-%m-%d %H:%M}"
                    "<br>PriSTI: %{y:.2f}"
                    "<br>Real: %{customdata[0]:.2f}"
                    "<br>MSE: %{customdata[1]:.3f}<extra></extra>"
                ),
            ))

    # Dummy legend entries (one each) so the colors are explained once
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=7, color="#7a7a7a"),
                             name="Real (ground)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=7, color="green"),
                             name="TSGuard (imputed)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=7, color="red"),
                             name="PriSTI (imputed)"))

    fig.update_layout(
        title="PriSTI vs TSGuard (non-overlapping 36-step windows)",
        xaxis_title="Time", yaxis_title="Sensor value",
        margin=dict(l=20, r=20, t=50, b=30),
        uirevision="pristi_cmp",
    )
    return fig

def render_pristi_vs_tsg_with_gt(
    ph,
    time_index,             # pd.DatetimeIndex de longueur 36
    pristi_cols,            # liste des 36 capteurs
    ours_block,             # DataFrame (36,36) TSGuard (échelle originale)
    pristi_block,           # DataFrame (36,36) PriSTI  (échelle originale)
    win_mask_df,            # DataFrame bool (36,36) -> True si manquant à l'origine
    ground_block=None,      # DataFrame (36,36) réel, optionnel
    title_suffix=""
):
    import numpy as np, pandas as pd, plotly.graph_objects as go

    have_gt = isinstance(ground_block, pd.DataFrame) and not ground_block.empty

    # petits décalages pour séparer visuellement les 3 points au même instant
    OFF_TSG = pd.Timedelta(seconds=-60)
    OFF_PRI = pd.Timedelta(seconds=+60)
    OFF_GT  = pd.Timedelta(seconds=0)

    x_gt,  y_gt,  hov_gt  = [], [], []
    x_tsg, y_tsg, hov_tsg = [], [], []
    x_pri, y_pri, hov_pri = [], [], []

    se_tsg, se_pri = [], []

    for j, sid in enumerate(pristi_cols):
        gt_col = ground_block[sid] if have_gt and (sid in (ground_block.columns if have_gt else [])) else None
        for i, ts in enumerate(time_index):
            if not bool(win_mask_df.iat[i, j]):   # uniquement les cellules imputées
                continue

            gt_val = float(gt_col.iat[i]) if (have_gt and pd.notna(gt_col.iat[i])) else np.nan
            tsg    = float(ours_block.iat[i, j])  if pd.notna(ours_block.iat[i, j])  else np.nan
            pri    = float(pristi_block.iat[i, j]) if pd.notna(pristi_block.iat[i, j]) else np.nan

            # --- réel (vert) ---
            if not np.isnan(gt_val):
                x_gt.append(ts + OFF_GT); y_gt.append(gt_val)
                hov_gt.append(f"Sensor {sid}<br>{ts}<br>Réel={gt_val:.3f}")

            # --- TSGuard (bleu) ---
            if not np.isnan(tsg):
                if not np.isnan(gt_val):
                    mse = (tsg - gt_val) ** 2; se_tsg.append(mse)
                    hov_tsg.append(f"Sensor {sid}<br>{ts}<br>TSGuard={tsg:.3f}<br>GT={gt_val:.3f}<br>MSE={mse:.4f}")
                else:
                    hov_tsg.append(f"Sensor {sid}<br>{ts}<br>TSGuard={tsg:.3f}<br>GT=NA")
                x_tsg.append(ts + OFF_TSG); y_tsg.append(tsg)

            # --- PriSTI (rouge) ---
            if not np.isnan(pri):
                if not np.isnan(gt_val):
                    mse = (pri - gt_val) ** 2; se_pri.append(mse)
                    hov_pri.append(f"Sensor {sid}<br>{ts}<br>PriSTI={pri:.3f}<br>GT={gt_val:.3f}<br>MSE={mse:.4f}")
                else:
                    hov_pri.append(f"Sensor {sid}<br>{ts}<br>PriSTI={pri:.3f}<br>GT=NA")
                x_pri.append(ts + OFF_PRI); y_pri.append(pri)

    mse_tsg = float(np.mean(se_tsg)) if len(se_tsg) else float("nan")
    mse_pri = float(np.mean(se_pri)) if len(se_pri) else float("nan")

    title = "PriSTI vs TSGuard — points imputés (triplets GT/TSG/PriSTI)"
    if not np.isnan(mse_tsg) or not np.isnan(mse_pri):
        title += f" • MSE(TSG)={mse_tsg:.3f} • MSE(PriSTI)={mse_pri:.3f}"
    if title_suffix:
        title += f" {title_suffix}"

    fig = go.Figure()

    # Réel = VERT
    fig.add_trace(go.Scatter(
        x=x_gt, y=y_gt, mode="markers", name="Réel (ground)",
        marker=dict(size=8, color="green", opacity=0.9),
        hovertemplate="%{text}", text=hov_gt, showlegend=True
    ))
    # TSGuard = BLEU
    fig.add_trace(go.Scatter(
        x=x_tsg, y=y_tsg, mode="markers", name="TSGuard (imputed)",
        marker=dict(size=7, color="#1f77b4", opacity=0.9),
        hovertemplate="%{text}", text=hov_tsg, showlegend=True
    ))
    # PriSTI = ROUGE
    fig.add_trace(go.Scatter(
        x=x_pri, y=y_pri, mode="markers", name="PriSTI (imputed)",
        marker=dict(size=7, color="red", opacity=0.9),
        hovertemplate="%{text}", text=hov_pri, showlegend=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Sensor value",
        margin=dict(l=10, r=10, t=48, b=10),
        uirevision="pristi_cmp_triplet",
        legend_title="Séries",
    )

    ts_key = int(pd.Timestamp(time_index[-1]).value)
    ph.plotly_chart(fig, use_container_width=True, key=f"cmp_triplet_{ts_key}")

def _avg_mse_per_timestamp_pair(
    ours_block: pd.DataFrame,
    pristi_block: pd.DataFrame,
    ground_block: Optional[pd.DataFrame],
    win_mask_df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    idx_time = ours_block.index
    if ground_block is None or ground_block.empty:
        nan_series = pd.Series(np.nan, index=idx_time, dtype="float64")
        return nan_series, nan_series

    e1 = ours_block.to_numpy(dtype="float64")    # TSGuard
    e2 = pristi_block.to_numpy(dtype="float64")  # PriSTI
    gt = ground_block.to_numpy(dtype="float64")
    m  = win_mask_df.to_numpy(dtype=bool)

    f1 = np.isfinite(e1); f2 = np.isfinite(e2); fg = np.isfinite(gt)
    T  = e1.shape[0]
    out1 = np.full(T, np.nan, dtype="float64")
    out2 = np.full(T, np.nan, dtype="float64")

    for t in range(T):
        common = m[t] & f1[t] & f2[t] & fg[t]
        if common.any():
            d1 = e1[t, common] - gt[t, common]
            d2 = e2[t, common] - gt[t, common]
            out1[t] = float(np.mean(d1 * d1))
            out2[t] = float(np.mean(d2 * d2))
    return pd.Series(out1, index=idx_time), pd.Series(out2, index=idx_time)

def render_pristi_window_only(
    ph,
    time_index: pd.DatetimeIndex,
    pristi_cols: list[str],
    pristi_block: pd.DataFrame,   # (36, N) valeurs PriSTI (échelle originale)
    win_mask_df: pd.DataFrame,    # (36, N) bool -> True si cellule était manquante à l’origine
    title_suffix: str = "",
):
    # palette fixe
    palette = ["#000000", "#003366", "#009999", "#006600", "#66CC66",
               "#FF9933", "#FFD700", "#708090", "#4682B4", "#99FF33"]
    sensor_color_map = {c: palette[i % len(palette)] for i, c in enumerate(pristi_cols)}

    # aligne les grilles
    pristi_block = pristi_block.reindex(index=time_index, columns=pristi_cols)
    win_mask_df  = win_mask_df.reindex(index=time_index, columns=pristi_cols).fillna(False)

    fig = go.Figure()
    # segments style TSGuard : rouge si un des bouts est imputé
    _window_lines(
        fig=fig,
        block=pristi_block,
        mask_df=win_mask_df,
        sensor_cols=pristi_cols,
        sensor_color_map=sensor_color_map,
        gap_hours=6,
        only_last_imputed=False,  # on garde tout, ça « glisse » à gauche naturellement
    )

    title = "PriSTI — Last 36 timestamps"
    if title_suffix:
        title += f" {title_suffix}"

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Sensor value",
        margin=dict(l=10, r=10, t=48, b=10),
        uirevision="pristi_only",  # conserve zoom/viewport
    )

    # clé stable pour que Streamlit mette juste à jour le graphe
    ts_key = int(pd.Timestamp(time_index[-1]).value)
    ph.plotly_chart(fig, use_container_width=True, key=f"pri_only_{ts_key}")

def _window_lines(fig: go.Figure,
                  block: pd.DataFrame,          # (T,N) values for 36-step window
                  mask_df: pd.DataFrame,        # same shape; True where imputed (originally missing)
                  sensor_cols: list[str],
                  sensor_color_map: dict[str, str],
                  gap_hours: int = 6,
                  only_last_imputed: bool = False):
    """
    Draw TSGuard-like line segments for a 36-step window.
    Red segment if either end is imputed; otherwise use base color per sensor.
    """
    sub_df = block.copy()

    # Turn the datetime index into a *column* and drop the index name
    # so there's no index level named "datetime".
    sub_df = sub_df.reset_index()

    # Ensure the timestamp column is called "datetime"
    # (after reset_index it will be the index name if one existed, otherwise "index")
    first_col = sub_df.columns[0]
    if first_col != "datetime":
        sub_df = sub_df.rename(columns={first_col: "datetime"})
    last_ts = pd.to_datetime(sub_df["datetime"]).max()

    for col in sensor_cols:
        base_color = sensor_color_map.get(col, "#444")
        xy = (sub_df[["datetime", col]]
              .dropna()
              .sort_values("datetime")
              .rename(columns={col: "value"}))

        if xy.empty:
            continue

        # imputed iff that cell was originally missing
        def is_imp(ts):
            try:
                imp = bool(mask_df.loc[ts, col])
                # if only_last_imputed, highlight ONLY the newest timestamp
                return imp and (not only_last_imputed or ts == last_ts)
            except Exception:
                return False

        add_imputed_segments(fig, xy, is_imp, base_color, gap_hours=gap_hours)

    # add a visible legend swatch per sensor
    for col in sensor_cols:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=8, color=sensor_color_map.get(col, "#444")),
                                 showlegend=True, name=f"{col}"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=8, color="red"),
                             showlegend=True, name="Imputed segment"))


def render_pristi_window_plus_mse(
    ph,
    time_index: pd.DatetimeIndex,
    pristi_cols: list,
    ours_block: pd.DataFrame,          # TSGuard window values (T,N) for MSE only
    pristi_block: pd.DataFrame,        # PriSTI window values (T,N)
    win_mask_df: pd.DataFrame,         # originally-missing mask (T,N)
    ground_block: pd.DataFrame = None, # real window (T,N)
    title_suffix: str = "",
):
    # --- align everything to the same (T,N) grid ---
    pristi_block = pristi_block.reindex(index=time_index, columns=pristi_cols)
    ours_block   = ours_block.reindex(index=time_index, columns=pristi_cols)
    win_mask_df  = win_mask_df.reindex(index=time_index, columns=pristi_cols).fillna(False)
    if ground_block is not None:
        ground_block = ground_block.reindex(index=time_index, columns=pristi_cols)

    # palette…
    palette = ["#000000", "#003366", "#009999", "#006600", "#66CC66",
               "#FF9933", "#FFD700", "#708090", "#4682B4", "#99FF33"]
    sensor_color_map = {c: palette[i % len(palette)] for i, c in enumerate(pristi_cols)}

    # ----- PriSTI window (on n’affiche plus la fenêtre TSGuard)
    fig_pri = go.Figure()
    fig_pri.update_layout(
        title=None,
        xaxis_title="Time", yaxis_title="Sensor value",
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision="pristi_only",
    )
    _window_lines(fig_pri, pristi_block, win_mask_df, pristi_cols, sensor_color_map)

    # --- Avg MSE sur masque commun ---
    mse_tsg, mse_pri = _avg_mse_per_timestamp_pair(
        ours_block=ours_block,
        pristi_block=pristi_block,
        ground_block=ground_block,
        win_mask_df=win_mask_df,
    )

    fig_mse = go.Figure()
    fig_mse.add_trace(go.Scatter(
        x=mse_tsg.index, y=mse_tsg.values,
        mode="lines+markers", name="TSGuard MSE",
        connectgaps=False
    ))
    fig_mse.add_trace(go.Scatter(
        x=mse_pri.index, y=mse_pri.values,
        mode="lines+markers", name="PriSTI MSE",
        connectgaps=False
    ))
    fig_mse.update_layout(
        title="Avg MSE per timestamp (masked cells)",
        xaxis_title="Time", yaxis_title="MSE",
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision="mse_cmp",
        yaxis=dict(rangemode="tozero"),
    )

    # ----- layout: 2 colonnes (PriSTI | MSE)
    ph.empty()
    cont = ph.container()
    c1, c2 = cont.columns([3, 2])

    ts_key = f"{int(pd.Timestamp(time_index[-1]).value)}"
    c1.plotly_chart(fig_pri, use_container_width=True, key=f"pri_win_only_{ts_key}")
    c2.plotly_chart(fig_mse, use_container_width=True, key=f"mse_cmp_{ts_key}")


def run_simulation_with_live_imputation(
    sim_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    positions,
    model: torch.nn.Module,
    scaler: callable,
    inv_scaler: callable,
    device: torch.device,
    graph_placeholder,              # placeholder de la carte (ou None)
    sliding_chart_placeholder,      # inutilisé mais conservé pour compatibilité
    gauge_placeholder,              # inutilisé mais conservé pour compatibilité
    window_hours: int = 24,
):
    """
    Simulation + imputation en direct.
    - En-tête (titre + heure + bouton) AU-DESSUS de la carte.
    - Carte pydeck juste en dessous.
    - Fenêtre séparée (expander) pour la comparaison Snapshot 36 : TSGUARD / PRISTI.
    """

    import os, time, uuid
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import pydeck as pdk
    import streamlit as st

    # ---------- Utilitaires internes ----------
    def zpad6(s: str) -> str:
        return s if not s.isdigit() else s.zfill(6)

    def strip0(s: str) -> str:
        t = s.lstrip("0")
        return t if t else "0"

    GREEN = [46, 204, 113, 200]   # réel
    RED   = [231, 76, 60, 200]    # imputé

    def make_bg_layer(df):
        return pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["longitude", "latitude"],
            get_fill_color="bg_color",
            get_radius="bg_radius",
            radius_scale=1,
            radius_min_pixels=6,
            radius_max_pixels=22,
            stroked=True,
            get_line_color=[255, 255, 255, 180],
            line_width_min_pixels=1,
            pickable=False,
        )

    def make_icon_layer(df):
        return pdk.Layer(
            "IconLayer",
            data=df,
            get_icon="icon",
            get_position=["longitude", "latitude"],
            get_size="icon_size",
            size_scale=8,
            size_min_pixels=14,
            size_max_pixels=28,
            pickable=True,
        )

    def fit_view_simple(df: pd.DataFrame, padding_deg=0.02) -> pdk.ViewState:
        if df is None or df.empty:
            return pdk.ViewState(latitude=0, longitude=0, zoom=2, bearing=0, pitch=0)
        lat_min = float(df["latitude"].min());  lat_max = float(df["latitude"].max())
        lon_min = float(df["longitude"].min()); lon_max = float(df["longitude"].max())
        lat_c = (lat_min + lat_max) / 2.0
        lon_c = (lon_min + lon_max) / 2.0
        span = max(lat_max - lat_min, lon_max - lon_min) + padding_deg
        span = max(span, 1e-3)
        zoom = max(1.0, min(16.0, np.log2(360.0 / span)))
        return pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom, bearing=0, pitch=0)

    # ---------- Sélection des capteurs ----------
    all_sensor_cols = [c for c in sim_df.columns if c != "datetime"]
    graph_size = int(st.session_state.get("graph_size", DEFAULT_VALUES["graph_size"]))
    sensor_cols = [str(c) for c in all_sensor_cols[:graph_size]]

    # ---------- Positions & mapping capteurs ----------
    latlng_raw = positions_to_df(positions).copy()
    latlng_raw["sensor_id"] = latlng_raw["sensor_id"].astype(str).str.strip()
    latlng_raw["latitude"]  = pd.to_numeric(latlng_raw["latitude"],  errors="coerce")
    latlng_raw["longitude"] = pd.to_numeric(latlng_raw["longitude"], errors="coerce")
    latlng_raw = latlng_raw.dropna(subset=["latitude", "longitude"])

    pos_ids = latlng_raw["sensor_id"].tolist()

    map_exact  = {pid: pid for pid in pos_ids if pid in sensor_cols}
    map_pad6   = {pid: zpad6(pid) for pid in pos_ids if zpad6(pid) in sensor_cols}
    map_strip0 = {}
    for pid in pos_ids:
        s  = strip0(pid); s6 = zpad6(s)
        if s in sensor_cols:    map_strip0[pid] = s
        elif s6 in sensor_cols: map_strip0[pid] = s6

    map_index = {}
    if all(p.isdigit() for p in pos_ids):
        nums = sorted(int(p) for p in pos_ids)
        if nums and nums[0] == 0 and nums[-1] == len(nums) - 1:
            for i, pid in enumerate(sorted(pos_ids, key=lambda x: int(x))):
                if i < len(sensor_cols):
                    map_index[pid] = sensor_cols[i]

    candidates = [map_exact, map_pad6, map_strip0, map_index]
    best_map = max(candidates, key=lambda m: len(m))
    if len(best_map) == 0:
        st.info("No matching positions for selected sensors. Nothing to display.")
        return

    latlng = latlng_raw.copy()
    latlng["data_col"] = latlng["sensor_id"].map(best_map)
    latlng = latlng[latlng["data_col"].notna()].copy()
    order_index = {c: i for i, c in enumerate(sensor_cols)}
    latlng["__ord"] = latlng["data_col"].map(order_index)
    latlng = latlng.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    # garde l’ordre existant
    sensor_cols = [c for c in sensor_cols if c in set(latlng["data_col"])]
    if not sensor_cols:
        st.info("After mapping, no sensors remain to plot.")
        return
    col_to_idx = {c: i for i, c in enumerate(sensor_cols)}

    # ---------- Alignement temporel ----------
    def ensure_datetime_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "datetime" in df.columns:
            return df
        if isinstance(df.index, pd.DatetimeIndex):
            return df.reset_index().rename(columns={"index": "datetime"})
        for alt in ("timestamp", "date", "time"):
            if alt in df.columns:
                return df.rename(columns={alt: "datetime"})
        try:
            idx_as_dt = pd.to_datetime(df.index, errors="raise")
            out = df.reset_index().rename(columns={"index": "datetime"})
            out["datetime"] = idx_as_dt
            return out
        except Exception:
            raise KeyError(f"{name} has no 'datetime' column or datetime-like index.")

    sim_df     = ensure_datetime_column(sim_df, "sim_df")
    missing_df = ensure_datetime_column(missing_df, "missing_df")

    sim_df["datetime"]     = pd.to_datetime(sim_df["datetime"], errors="coerce")
    missing_df["datetime"] = pd.to_datetime(missing_df["datetime"], errors="coerce")
    sim_df     = sim_df.dropna(subset=["datetime"]).copy()
    missing_df = missing_df.dropna(subset=["datetime"]).copy()

    sim_df["datetime"]     = sim_df["datetime"].dt.floor("h")
    missing_df["datetime"] = missing_df["datetime"].dt.floor("h")
    sim_df.set_index("datetime", inplace=True)
    missing_df.set_index("datetime", inplace=True)
    sim_df     = sim_df[~sim_df.index.duplicated(keep="first")].copy()
    missing_df = missing_df[~missing_df.index.duplicated(keep="first")].copy()

    if "orig_missing_baseline" not in st.session_state:
        st.session_state.orig_missing_baseline = missing_df.isna().copy()
    else:
        if (not st.session_state.orig_missing_baseline.index.equals(missing_df.index) or
            list(st.session_state.orig_missing_baseline.columns) != list(missing_df.columns)):
            st.session_state.orig_missing_baseline = missing_df.isna().copy()

    base_index   = missing_df.index
    sim_df       = sim_df.reindex(base_index)
    common_index = base_index
    if common_index.empty or latlng.empty:
        st.info("No matching timeline or positions/sensors. Nothing to display.")
        return

    # ---------- PriSTI (optionnel) ----------
    PRISTI_ROOT = "./PriSTI"
    CONFIG_PATH = f"{PRISTI_ROOT}/config/base.yaml"
    WEIGHTS_PATH = f"{PRISTI_ROOT}/save/aqi36/model.pth"
    MEANSTD_PK   = f"{PRISTI_ROOT}/data/pm25/pm25_meanstd.pk"

    all_cols_for_pristi = list(missing_df.columns)
    have_36 = len(all_cols_for_pristi) >= 36
    have_cfg = os.path.exists(CONFIG_PATH)
    have_wts = os.path.exists(WEIGHTS_PATH)
    have_ms  = os.path.exists(MEANSTD_PK)

    pristi_enabled = have_36 and have_cfg and have_wts and have_ms
    if pristi_enabled:
        pristi_cols = all_cols_for_pristi[:36]
        pristi_model, pristi_mean, pristi_std = load_pristi_artifacts(CONFIG_PATH, WEIGHTS_PATH, MEANSTD_PK, device)
        st.session_state["pristi_model"] = pristi_model
        st.session_state["pristi_mean"]  = pristi_mean
        st.session_state["pristi_std"]   = pristi_std
        if "pristi_running_df" not in st.session_state:
            st.session_state["pristi_running_df"] = missing_df.copy()
    else:
        miss = []
        if not have_36: miss.append("≥36 capteurs requis")
        if not have_cfg: miss.append("config manquante")
        if not have_wts: miss.append("poids manquants")
        if not have_ms:  miss.append("mean/std manquants")
        if miss:
            st.warning("PriSTI désactivé — " + ", ".join(miss))

    # ===================== EN-TÊTE AU-DESSUS DE LA CARTE ======================
    st.markdown("### Sensor Simulation Graph")

    col_l, col_r = st.columns([7, 2])
    with col_l:
        # placeholder heure (init '—')
        time_slot = st.markdown(
            "<div style='font-weight:600;margin-bottom:6px'>Current Time: —</div>",
            unsafe_allow_html=True
        )
    with col_r:
        fit_clicked = st.button("Fit map to sensors", key="fit_to_sensors_btn", use_container_width=True)

    # map juste en dessous (petite marge)
    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)
    map_ph = graph_placeholder if graph_placeholder is not None else st.empty()

    # ===================== CARTE (initialisation sûre) ========================
    global ICON_SPEC
    if "ICON_SPEC" not in globals() or ICON_SPEC is None:
        ICON_SPEC = {"url": "", "width": 1, "height": 1, "anchorX": 0, "anchorY": 0}

    if "deck_obj" not in st.session_state:
        base_df = latlng.copy()
        base_df["sensor_id"] = base_df["sensor_id"].astype(str)
        base_df["value"] = "NA"
        base_df["status"] = "Predicted"
        base_df["bg_color"] = [[231, 76, 60, 200] for _ in range(len(base_df))]
        base_df["bg_radius"] = 10
        base_df["icon"] = [ICON_SPEC] * len(base_df)
        base_df["icon_size"] = 1.0
        st.session_state._fit_base_df = base_df.copy()

        initial_view = fit_view_simple(base_df)
        st.session_state.deck_obj = pdk.Deck(
            layers=[make_bg_layer(base_df), make_icon_layer(base_df)],
            initial_view_state=initial_view,
            map_style="mapbox://styles/mapbox/light-v11",
            tooltip={"text": "Sensor {sensor_id}\nValue: {value}\nStatus: {status}"},
        )

    if fit_clicked:
        st.session_state.deck_obj.initial_view_state = fit_view_simple(st.session_state._fit_base_df)

    # Render initial map
    map_ph.pydeck_chart(st.session_state.deck_obj, use_container_width=True)



    # ===================== AUTRES ZONES (gauge + global TS) ===================
    if not st.session_state.get("_charts_layout_inited", False):
        st.markdown("---")
        cL, cR = st.columns([1, 3])
        with cL:
            st.subheader("Missed Data (%)")
            st.session_state["_gauge_ph"] = st.empty()
        with cR:
            st.subheader("Global Time Series")
            st.session_state["_global_ts_ph"] = st.empty()
        st.session_state["_charts_layout_inited"] = True

        # ===================== FENÊTRE SÉPARÉE — SNAPSHOT 36 ======================
        if not st.session_state.get("_cmp_layout_inited", False):
            st.markdown("---")
            with st.expander("Comparaison — Snapshot 36", expanded=False):
                st.subheader("Snapshot 36")
                st.caption("TSGUARD")
                ph_tsg = st.empty()
                st.caption("PRISTI")
                ph_pri = st.empty()
            st.session_state["_snapshot36_ph"] = ph_tsg
            st.session_state["_pristi_cmp_ph"] = ph_pri
            st.session_state["_cmp_layout_inited"] = True

    # Raccourcis placeholders
    snapshot_ph  = st.session_state["_snapshot36_ph"]
    pristi_ph    = st.session_state["_pristi_cmp_ph"]
    gauge_ph     = st.session_state["_gauge_ph"]
    global_ts_ph = st.session_state["_global_ts_ph"]

    # ===================== Buffers & couleurs ================================
    if "sim_uid" not in st.session_state:
        st.session_state.sim_uid = f"sim_{uuid.uuid4().hex[:8]}"
    if "sim_iter" not in st.session_state:
        st.session_state.sim_iter = 0
    uid = st.session_state.sim_uid

    SNAPSHOT_STEPS = 36
    palette = ["#000000", "#003366", "#009999", "#006600", "#66CC66",
               "#FF9933", "#FFD700", "#708090", "#4682B4", "#99FF33"]
    sensor_color_map = {c: palette[i % len(palette)] for i, c in enumerate(sensor_cols)}
    sliding_window_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))
    global_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))

    # masque d’imputation (pour le rendu)
    imputed_mask = pd.DataFrame(False, index=common_index, columns=sensor_cols, dtype=bool)

    # ===================== Boucle principale =================================
    arrived_values_number = 0
    missingness_number = 0
    use_model = model is not None
    if use_model:
        model.eval()

    for current_time in list(common_index):
        arrived_values_number += 1
        st.session_state.sim_iter += 1
        iter_key = st.session_state.sim_iter
        current_time = pd.Timestamp(current_time)

        # Fenêtre historique
        hist_end = current_time - pd.Timedelta(hours=1)
        if hist_end in missing_df.index:
            hist_idx = missing_df.loc[:hist_end].index[-window_hours:]
        else:
            hist_idx = missing_df.index[missing_df.index < current_time][-window_hours:]
        hist_win = missing_df.loc[hist_idx, sensor_cols] if len(hist_idx) > 0 else pd.DataFrame()

        # Valeurs du tick
        svals, sstatus = [], []
        for col in sensor_cols:
            v = missing_df.at[current_time, col]
            if pd.isna(v):
                if not hist_win.empty:
                    if use_model:
                        try:
                            target_idx = col_to_idx[col]
                            pred_val = predict_single_missing_value(
                                historical_window=np.asarray(hist_win.values, dtype=np.float32),
                                target_sensor_index=target_idx,
                                model=model, scaler=scaler, inv_scaler=inv_scaler, device=device
                            )
                        except Exception:
                            last = pd.to_numeric(hist_win[col].dropna(), errors="coerce")
                            pred_val = float(last.iloc[-1]) if len(last) else np.nan
                    else:
                        last = pd.to_numeric(hist_win[col].dropna(), errors="coerce")
                        pred_val = float(last.iloc[-1]) if len(last) else np.nan

                    missing_df.at[current_time, col] = pred_val if pd.notna(pred_val) else np.nan
                    svals.append(pred_val if pd.notna(pred_val) else np.nan)
                    sstatus.append(False)  # imputé
                    imputed_mask.at[current_time, col] = pd.notna(pred_val)
                else:
                    svals.append(np.nan); sstatus.append(False)
                    imputed_mask.at[current_time, col] = False
            else:
                svals.append(v); sstatus.append(True)
                imputed_mask.at[current_time, col] = False

        # Buffers pour graphes
        row_dict = {"datetime": current_time}
        for i, col in enumerate(sensor_cols):
            row_dict[col] = svals[i]
        sliding_window_df.loc[len(sliding_window_df)] = row_dict
        global_df.loc[len(global_df)] = row_dict
        if len(sliding_window_df) > SNAPSHOT_STEPS:
            sliding_window_df = sliding_window_df.tail(SNAPSHOT_STEPS)

        # PriSTI — fenêtre glissante si dispo
        if pristi_enabled:
            try:
                pristi_running_df = st.session_state["pristi_running_df"]
                end_loc = pristi_running_df.index.get_loc(current_time)
                EVAL_LENGTH = 36
                if not isinstance(end_loc, slice) and (end_loc + 1) >= EVAL_LENGTH:
                    start_loc = end_loc - (EVAL_LENGTH - 1)
                    time_index = pristi_running_df.index[start_loc:end_loc + 1]

                    updated_df, info = impute_window_with_pristi(
                        missing_df=pristi_running_df.copy(),
                        sensor_cols=pristi_cols,
                        target_timestamp=current_time,
                        model=st.session_state["pristi_model"],
                        device=device, eval_len=EVAL_LENGTH, nsample=100
                    )
                    if info == "ok":
                        pristi_running_df.loc[time_index, pristi_cols] = updated_df.loc[time_index, pristi_cols].values
                        st.session_state["pristi_running_df"] = pristi_running_df

                    # Rendu PRISTI dans la fenêtre séparée (sans titre interne)
                    win_mask_df = st.session_state.orig_missing_baseline.reindex(
                        index=time_index, columns=pristi_cols).fillna(False)
                    pristi_block = pristi_running_df.loc[time_index, pristi_cols].copy()

                    # Construire une figure type "window lines" sans titre
                    fig_pri = go.Figure()
                    # réutilise votre _window_lines si dispo; sinon, fallback simple
                    if "_window_lines" in globals():
                        palette2 = ["#000000", "#003366", "#009999", "#006600", "#66CC66",
                                    "#FF9933", "#FFD700", "#708090", "#4682B4", "#99FF33"]
                        sensor_color_map2 = {c: palette2[i % len(palette2)] for i, c in enumerate(pristi_cols)}
                        _window_lines(fig_pri, pristi_block, win_mask_df, pristi_cols, sensor_color_map2)
                    fig_pri.update_layout(
                        title=None, xaxis_title="Time", yaxis_title="Sensor value",
                        margin=dict(l=10, r=10, t=30, b=10), uirevision="pri_only",
                    )
                    pristi_ph.plotly_chart(fig_pri, use_container_width=True, key=f"{uid}_pri_{iter_key}")

            except Exception as e:
                st.caption(f"PriSTI error @ {current_time}: {e}")

        # ===================== Mise à jour carte + heure ======================
        vals_by_col = {col: svals[i] for i, col in enumerate(sensor_cols)}
        real_by_col = {col: sstatus[i] for i, col in enumerate(sensor_cols)}

        tick_df = latlng.copy()
        tick_df["value"]  = tick_df["data_col"].map(vals_by_col).fillna("NA")
        tick_df["status"] = tick_df["data_col"].map(lambda c: "Real" if real_by_col.get(c, False) else "Predicted")
        tick_df["bg_color"] = [GREEN if s == "Real" else RED for s in tick_df["status"]]
        tick_df["bg_radius"] = 10
        tick_df["icon"] = [ICON_SPEC] * len(tick_df)
        tick_df["icon_size"] = 1.0

        st.session_state.deck_obj.layers = [make_bg_layer(tick_df), make_icon_layer(tick_df)]

        time_slot.markdown(
            f"<div style='font-weight:600;margin-bottom:6px'>Current Time: {current_time}</div>",
            unsafe_allow_html=True
        )
        map_ph.pydeck_chart(st.session_state.deck_obj, use_container_width=True)

        # ===================== Snapshot 36 (TSGUARD) ==========================
        sliding_fig = go.Figure()
        # segments colorés (rouge si un bout imputé)
        def add_imputed_segments(fig, df_xy, mask_col_bool, base_color, gap_hours=6):
            GAP = pd.Timedelta(hours=gap_hours)
            prev_x = prev_y = prev_imp = None
            for x, y in zip(df_xy["datetime"], df_xy["value"]):
                imp = bool(mask_col_bool(x))
                if prev_x is not None:
                    if x - prev_x > GAP:
                        prev_x, prev_y, prev_imp = x, y, imp
                        continue
                    seg_color = "red" if (imp or prev_imp) else base_color
                    fig.add_trace(go.Scatter(
                        x=[prev_x, x], y=[prev_y, y],
                        mode="lines+markers",
                        line=dict(width=2, color=seg_color),
                        marker=dict(size=6, color=seg_color),
                        showlegend=False,
                    ))
                prev_x, prev_y, prev_imp = x, y, imp

        for col in sensor_cols:
            base_color = sensor_color_map[col]
            sub = (sliding_window_df[["datetime", col]]
                   .dropna().sort_values("datetime").rename(columns={col: "value"}))
            if sub.empty:
                continue
            is_imp = lambda t, c=col: (t in imputed_mask.index) and bool(imputed_mask.loc[t, c])
            add_imputed_segments(sliding_fig, sub, is_imp, base_color, gap_hours=6)

        # Légende compacte
        for col in sensor_cols:
            sliding_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                marker=dict(size=8, color=sensor_color_map[col]),
                showlegend=True, name=f"Sensor {col}"
            ))
        sliding_fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(size=8, color="red"), showlegend=True, name="Imputed segment"
        ))

        sliding_fig.update_layout(
            title=None,  # un seul titre dans l'expander
            xaxis_title="Time", yaxis_title="Sensor Value",
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title="Sensors",
            uirevision="snapshot",
        )
        snapshot_ph.plotly_chart(sliding_fig, use_container_width=True, key=f"{uid}_snapshot_{iter_key}")

        # ===================== Global TS + Gauge ==============================
        full_ts_fig = draw_full_time_series_with_mask_gap(
            global_df.copy(), imputed_mask, sensor_cols, sensor_color_map)

        row_imp   = imputed_mask.reindex(index=[current_time], columns=sensor_cols, fill_value=False).iloc[0]
        imputed_now = int(row_imp.sum())
        missingness_number += imputed_now
        sensors_total = len(sensor_cols) if len(sensor_cols) > 0 else 1
        real_now = sensors_total - imputed_now
        imputed_pct = 100.0 * missingness_number / (sensors_total * arrived_values_number)

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number", value=imputed_pct,
            title={"text": f"Imputed now: {imputed_now} • Real: {real_now}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if imputed_pct >= DEFAULT_VALUES['gauge_red_max'] else "green"},
                "steps": [
                    {"range": [DEFAULT_VALUES['gauge_green_min'], DEFAULT_VALUES['gauge_green_max']], "color": "lightgreen"},
                    {"range": [DEFAULT_VALUES['gauge_yellow_min'], DEFAULT_VALUES['gauge_yellow_max']], "color": "yellow"},
                    {"range": [DEFAULT_VALUES['gauge_red_min'],    DEFAULT_VALUES['gauge_red_max']],    "color": "red"},
                ],
            },
        ))
        gauge_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), uirevision="gauge")

        global_ts_ph.plotly_chart(full_ts_fig, use_container_width=True, key=f"{uid}_global_{iter_key}")
        gauge_ph.plotly_chart(gauge_fig, use_container_width=True, key=f"{uid}_gauge_{iter_key}")

        # pacing de la simulation
        time.sleep(1)




