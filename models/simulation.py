from __future__ import annotations
import pickle
import torch
import torch.nn as nn
from streamlit import title
from sympy import false
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
# ---- Plotly: force light theme globally ----
import plotly.io as pio
pio.templates.default = "plotly_white"

LIGHT_BG = "rgba(0,0,0,0)"
BASE_FONT = dict(
    family="Inter, Segoe UI, -apple-system, system-ui, sans-serif",
    size=14,
    color="#0F172A",  # dark text
)

def lightify(fig, *, title=None):
    """Make any Plotly fig blend with the light UI."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=LIGHT_BG,   # outside of axes
        plot_bgcolor=LIGHT_BG,    # inside axes
        font=BASE_FONT,
        title=title if title is not None else fig.layout.title.text if fig.layout.title else None,
        legend=dict(bgcolor="rgba(255,255,255,0.6)", borderwidth=0),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(showline=False, gridcolor="rgba(148,163,184,0.35)", zerolinecolor="rgba(148,163,184,0.4)")
    fig.update_yaxes(showline=False, gridcolor="rgba(148,163,184,0.35)", zerolinecolor="rgba(148,163,184,0.4)")
    return fig

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

# ===================== Constraint & Scenario Helpers =====================

def _month_name_from_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%B")  # "January", "February", ...

def _neighbors_within_km(latlng_df: pd.DataFrame, km: float) -> dict[str, list[str]]:
    """Build neighbor lists by distance threshold (km). Keys are sensor_id (string)."""
    ids = latlng_df["sensor_id"].astype(str).tolist()
    lat = latlng_df["latitude"].astype(float).to_numpy()
    lon = latlng_df["longitude"].astype(float).to_numpy()

    # simple O(N^2) because N is small on screen; fine for UI
    neigh = {sid: [] for sid in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = haversine_distance(lat[i], lon[i], lat[j], lon[j])
            if d <= km:
                a, b = ids[i], ids[j]
                neigh[a].append(b)
                neigh[b].append(a)
    return neigh

def _check_value_against_constraints(
    sensor_id: str,
    value: float,
    ts: pd.Timestamp,
    constraints: list[dict],
) -> list[str]:
    """
    Returns a list of violation labels for this sensor's *own* constraints (no neighbor diffs here).
    Currently supports your Temporal constraints, e.g.:
      {"type":"Temporal","month":"January","option":"Greater than","temp_threshold":50.0}
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []

    vios = []
    month_name = _month_name_from_ts(ts)
    for c in constraints:
        if c.get("type") != "Temporal":
            continue
        if c.get("month") != month_name:
            continue
        opt = c.get("option", "").lower()
        thr = float(c.get("temp_threshold", np.nan))
        if not np.isfinite(thr):
            continue

        if opt.startswith("greater"):
            if not (value > thr):
                vios.append(f"Temporal: expected > {thr}, got {value:.2f}")
        elif opt.startswith("less"):
            if not (value < thr):
                vios.append(f"Temporal: expected < {thr}, got {value:.2f}")
    return vios

def _check_spatial_diffs_at_timestamp(
    ts: pd.Timestamp,
    latlng_df: pd.DataFrame,
    values_by_sensor: dict[str, float],
    constraints: list[dict],
) -> list[str]:
    """
    Checks all *Spatial* constraints at timestamp ts.
    For each spatial constraint: if two sensors are within 'distance in km', then |diff| must be <= 'diff'.
    Returns violation messages like "Spatial: |S1-S2| 7.3 > 5.0 (≤2.0km)".
    """
    msgs = []
    spatial_cs = [c for c in constraints if c.get("type") == "Spatial"]
    if not spatial_cs or latlng_df.empty:
        return msgs

    # Precompute once per distinct km
    km_to_neighbors = {}
    for c in spatial_cs:
        km = float(c.get("distance in km", 0))
        if km <= 0:
            continue
        if km not in km_to_neighbors:
            km_to_neighbors[km] = _neighbors_within_km(latlng_df, km)

    for c in spatial_cs:
        km = float(c.get("distance in km", 0))
        max_diff = float(c.get("diff", np.nan))
        if km <= 0 or not np.isfinite(max_diff):
            continue
        neigh = km_to_neighbors.get(km, {})
        for sid, nlist in neigh.items():
            v1 = values_by_sensor.get(sid)
            if v1 is None or not np.isfinite(v1):
                continue
            for nid in nlist:
                v2 = values_by_sensor.get(nid)
                if v2 is None or not np.isfinite(v2):
                    continue
                d = abs(v1 - v2)
                if d > max_diff:
                    msgs.append(f"Spatial: |{sid}-{nid}| {d:.2f} > {max_diff:.2f} (≤{km:.1f} km)")
    return msgs
from typing import Optional
def evaluate_constraints_and_scenarios(
    ts: pd.Timestamp,
    latlng_mapped: pd.DataFrame,     # expects columns: sensor_id, data_col, latitude, longitude
    baseline_row_ts: pd.Series,      # True where *originally missing* at ts (indexed by data_col)
    values_by_data_col: dict[str, float],   # values after your imputation logic; keys = data_col
    imputed_mask_row: Optional[pd.Series],     # True where you imputed at ts (indexed by data_col)
    constraints: list[dict],
    sigma_minutes: float,
    missing_streak_hours: dict[str, int],   # keeps how long (hours) each data_col has been missing
) -> list[tuple[str, str]]:
    """
    Returns a list of alerts as (level, message).
      level ∈ {"info","warning","error"}
    Does NOT alter your plotting; only emits messages describing Scenario 1/2/3 and constraint violations.
    """

    alerts = []

    # --- Build dictionaries keyed by *display sensor_id* ---
    # data_col = the column name used in df; sensor_id = display name from positions
    # We’ll evaluate spatial on sensor_id (geometry) and temporal on its own value.
    sid_to_val = {}
    sid_to_missing = {}
    sid_to_imputed = {}

    for _, r in latlng_mapped.iterrows():
        sid = str(r["sensor_id"])
        col = str(r["data_col"])
        sid_to_val[sid] = values_by_data_col.get(col, np.nan)
        sid_to_missing[sid] = bool(baseline_row_ts.get(col, False))
        if imputed_mask_row is not None:
            sid_to_imputed[sid] = bool(imputed_mask_row.get(col, False))
        else:
            sid_to_imputed[sid] = False

    # --- Temporal checks per sensor (own constraints) ---
    for sid, val in sid_to_val.items():
        vios = _check_value_against_constraints(sid, val, ts, constraints)
        for v in vios:
            alerts.append(("warning", f"⏳ {sid} @ {ts}: {v}"))

    # --- Spatial checks (pairwise diffs) ---
    spatial_vios = _check_spatial_diffs_at_timestamp(ts, latlng_mapped[["sensor_id","latitude","longitude"]], sid_to_val, constraints)
    for msg in spatial_vios:
        alerts.append(("warning", f"📍 {ts}: {msg}"))

    # --- Scenario classification per sensor, based on delay + neighbor availability ---
    sigma_hours = max(0.0, float(sigma_minutes) / 60.0)

    # precompute neighbors using the *widest* spatial constraint (if any)
    spat = [c for c in constraints if c.get("type") == "Spatial" and float(c.get("distance in km", 0)) > 0]
    max_km = max((float(c["distance in km"]) for c in spat), default=0.0)
    neigh_idx = _neighbors_within_km(latlng_mapped[["sensor_id","latitude","longitude"]], max_km) if max_km > 0 else {}

    # Helper to check if “neighbors available” = any neighbor has originally-present value at this ts
    present_sids = {sid for sid, miss in sid_to_missing.items() if not miss}

    for _, r in latlng_mapped.iterrows():
        sid = str(r["sensor_id"])
        col = str(r["data_col"])

        is_missing = sid_to_missing.get(sid, False)
        streak_h = float(missing_streak_hours.get(col, 0))
        neighbors = neigh_idx.get(sid, []) if max_km > 0 else []

        if not is_missing:
            continue  # scenario classification concerns missing-at-ts only

        if streak_h < sigma_hours:
            alerts.append(("info", f"🕒 Scenario 1 — {sid} @ {ts}: delay below Δt, waiting for late data."))
        else:
            # decide 2 vs 3 by neighbor availability
            has_neighbor_present = any(n in present_sids for n in neighbors)
            if has_neighbor_present:
                # Scenario 2 — with neighbors; check sub-cases
                val = sid_to_val.get(sid, np.nan)
                own_vios = _check_value_against_constraints(sid, val, ts, constraints)
                # neighbor vios are already pushed above; check simple logic for sub-cases:
                neighbors_vals = [(n, sid_to_val.get(n, np.nan)) for n in neighbors]
                # consider neighbor temporal violations too:
                any_neigh_vio = False
                for n, nv in neighbors_vals:
                    nvios = _check_value_against_constraints(n, nv, ts, constraints)
                    if nvios:
                        any_neigh_vio = True
                        break

                if not own_vios and not any_neigh_vio:
                    alerts.append(("info", f"✅ Scenario 2.1 — {sid} @ {ts}: imputed within range; neighbors in range."))
                elif own_vios:
                    alerts.append(("error", f"🚨 Scenario 2.2 — {sid} @ {ts}: imputed value violates constraints ({'; '.join(own_vios)})."))
                elif any_neigh_vio:
                    alerts.append(("warning", f"⚠️ Scenario 2.3 — {sid} @ {ts}: neighbors out-of-range; possible masked anomaly."))
            else:
                # Scenario 3 — no neighbors
                alerts.append(("warning", f"🛰️ Scenario 3 — {sid} @ {ts}: neighbors unavailable; fallback to history."))
    return alerts


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


device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
#device = torch.device("cpu")

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
    title = "Global Sensors Time Series",
    fig.update_layout(

        xaxis_title="Time",
        yaxis_title="Sensor Value",
        margin=dict(l=20, r=20, t=40, b=20),
        legend_title="Sensors"
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
        title_text="Global Sensors Time Series",
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



def render_pristi_window_only(
    ph,
    time_index: pd.DatetimeIndex,
    pristi_cols: list[str],
    pristi_block: pd.DataFrame,   # (36, N) valeurs PriSTI (échelle originale)
    win_mask_df: pd.DataFrame,    # (36, N) bool -> True si cellule était manquante à l’origine
    title_suffix: str = "",
):
    # palette fixe
    palette = [
        "#000000", "#003366", "#009999", "#006600", "#66CC66",
        "#FF9933", "#FFD700", "#708090", "#4682B4", "#99FF33",
        "#1F77B4", "#5DA5DA", "#1E90FF", "#00BFFF", "#00CED1",
        "#17BECF", "#40E0D0", "#20B2AA", "#16A085", "#1ABC9C",
        "#2ECC71", "#3CB371", "#2CA02C", "#00FA9A", "#7FFFD4",
        "#ADFF2F", "#F1C40F", "#F4D03F", "#B7950B", "#4B0082",
        "#6A5ACD", "#7B68EE", "#483D8B", "#3F51B5", "#2E4057",
        "#A9A9A9"
    ]
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


def verify_constraints_and_alerts_for_timestamp(
    ts: pd.Timestamp,
    latlng_df: pd.DataFrame,
    values_by_col: dict,
    imputed_mask_row: pd.Series | None,      # index = sensor_cols
    baseline_row: pd.Series,          # index = sensor_cols (manquait à l’origine)
    missing_streak_hours: dict | None = None,   # {sensor_id: hours_without_value_when_missing}
):
    import numpy as np
    import streamlit as st
    from utils.config import DEFAULT_VALUES

    constraints = st.session_state.get("constraints", [])
    sigma_minutes = float(st.session_state.get("sigma_threshold", DEFAULT_VALUES.get("sigma_threshold", 30)))
    sigma_hours = max(0.0, sigma_minutes / 60.0)

    # états persistants pour anti-spam
    if "_fault_state" not in st.session_state:
        # state ∈ {"ok","waiting","fault"}
        st.session_state["_fault_state"] = {}
    fault_state = st.session_state["_fault_state"]

    # placeholder persistant
    if "_alerts_ph" not in st.session_state:
        st.session_state["_alerts_ph"] = st.container()
    alerts = st.session_state["_alerts_ph"]

    # on n’utilise pas les contraintes s’il n’y en a pas ; on garde la logique scenarios
    spatial_rules  = [c for c in constraints if c.get("type") == "Spatial"]
    temporal_rules = [c for c in constraints if c.get("type") == "Temporal"]

    # mapping positions (seulement si on a besoin de spatial)
    if spatial_rules:
        pos_map = {
            str(row["data_col"]): (float(row["latitude"]), float(row["longitude"]))
            for _, row in latlng_df.iterrows()
            if str(row.get("data_col")) in values_by_col
        }
        def haversine_km(lat1, lon1, lat2, lon2):
            from math import radians, sin, cos, sqrt, atan2
            R = 6371.0
            dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
            return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    messages = []

    for sid, val in values_by_col.items():
        v = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
        was_missing = bool(baseline_row.get(sid, False))
        is_imputed  = bool(imputed_mask_row.get(sid, False))
        streak_h    = float((missing_streak_hours or {}).get(sid, 0.0))

        prev = fault_state.get(sid, "ok")
        curr = prev  # par défaut

        # ---- Scénario 1 vs 3 (sans contraintes)
        if was_missing and (v is None) and not is_imputed:
            # pas d’estimation à ce timestamp
            if streak_h < sigma_hours:
                curr = "waiting"
                # n’alerter qu’au changement d’état
                if prev != "waiting":
                    messages.append(("info", f"⏳ {sid}: waiting for late data @ {ts} (delay below Δt)."))
            else:
                curr = "fault"
                if prev != "fault":
                    messages.append(("warning", f"⚠️ {sid}: no reliable estimate @ {ts}. Possible sensor/network fault."))
        else:
            # valeur disponible (réelle ou imputée)
            curr = "ok"
            if prev in ("waiting", "fault"):
                messages.append(("success", f"✅ {sid}: value available again @ {ts}."))

        # ---- Contraintes temporelles (optionnel – seulement si définies)
        if temporal_rules and (v is not None):
            mo_name = ts.strftime("%B")
            for rule in temporal_rules:
                if rule.get("month") != mo_name:
                    continue
                opt = rule.get("option")
                thr = rule.get("temp_threshold", None)
                try:
                    thr = float(thr)
                except Exception:
                    continue
                if opt == "Greater than" and not (v > thr):
                    messages.append(("warning", f"🚨 {sid} @ {ts}: value {v:.2f} ≤ {thr} in {mo_name}."))
                if opt == "Less than" and not (v < thr):
                    messages.append(("warning", f"🚨 {sid} @ {ts}: value {v:.2f} ≥ {thr} in {mo_name}."))

        # ---- Contraintes spatiales (optionnel – seulement si définies)
        if spatial_rules and (v is not None) and ('pos_map' in locals()) and (sid in pos_map):
            lat1, lon1 = pos_map[sid]
            for rule in spatial_rules:
                try:
                    dist_km  = float(rule.get("distance in km", 0))
                    max_diff = float(rule.get("diff", np.inf))
                except Exception:
                    continue
                for nid, nval in values_by_col.items():
                    if nid == sid or nid not in pos_map:
                        continue
                    nv = None if (nval is None or (isinstance(nval, float) and np.isnan(nval))) else float(nval)
                    if nv is None:
                        continue
                    lat2, lon2 = pos_map[nid]
                    d = haversine_km(lat1, lon1, lat2, lon2)
                    if d <= dist_km and abs(v - nv) > max_diff:
                        messages.append(("warning",
                            f"🚨 Spatial: {sid} vs {nid} @ {ts} (d≈{d:.1f} km): |Δ|={abs(v-nv):.2f} > {max_diff}"
                        ))

        # maj état
        fault_state[sid] = curr

    # rendu compact (et anti-spam)
    with alerts:
        for lvl, msg in messages:
            if lvl == "success":
                st.success(msg)
            elif lvl == "warning":
                st.warning(msg)
            else:
                st.info(msg)



def run_simulation_with_live_imputation(
    sim_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    positions,
    model: torch.nn.Module,
    scaler: callable,
    inv_scaler: callable,
    device: torch.device,
    graph_placeholder,              # unused, kept for signature
    sliding_chart_placeholder,      # unused, kept for signature
    gauge_placeholder,              # unused, kept for signature
    window_hours: int = 24,
):
    """
    Drop-in: fixes Fit button (rebuilds deck with new view), places it at top-right of left column,
    keeps map/gauge/global/snapshot10 + PriSTI/TSGuard comparison with imputation times.
    Requires helpers & constants already defined elsewhere in your module.
    """
    import os, time, uuid
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import pydeck as pdk
    import streamlit as st

    SS = st.session_state

    # ---------- small helpers ----------
    def init_once(key, val):
        if key not in SS:
            SS[key] = val
        return SS[key]

    def zpad6(s: str) -> str:
        return s if not s.isdigit() else s.zfill(6)

    def strip0(s: str) -> str:
        t = s.lstrip("0")
        return t if t else "0"

    GREEN = [46, 204, 113, 200]
    RED   = [231, 76, 60, 200]

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

    # ---------- select sensors for map/TS/snapshot10 ----------
    all_sensor_cols = [c for c in sim_df.columns if c != "datetime"]
    graph_size = int(SS.get("graph_size", DEFAULT_VALUES["graph_size"]))
    sensor_cols = [str(c) for c in all_sensor_cols[:graph_size]]
    col_to_idx = {c: i for i, c in enumerate(sensor_cols)}

    # ---------- positions ----------
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
        s = strip0(pid); s6 = zpad6(s)
        if s in sensor_cols: map_strip0[pid] = s
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
        st.info("No matching positions for selected sensors.")
        return

    latlng = latlng_raw.copy()
    latlng["data_col"] = latlng["sensor_id"].map(best_map)
    latlng = latlng[latlng["data_col"].notna()].copy()
    order_index = {c: i for i, c in enumerate(sensor_cols)}
    latlng["__ord"] = latlng["data_col"].map(order_index)
    latlng = latlng.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    sensor_cols = [c for c in sensor_cols if c in set(latlng["data_col"])]
    if not sensor_cols:
        st.info("After mapping, no sensors remain to plot.")
        return
    col_to_idx = {c: i for i, c in enumerate(sensor_cols)}

    # ---------- time alignment ----------
    def ensure_datetime_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "datetime" in df.columns:
            return df
        if isinstance(df.index, pd.DatetimeIndex):
            return df.reset_index().rename(columns={"index": "datetime"})
        for alt in ("timestamp", "date", "time"):
            if alt in df.columns:
                return df.rename(columns={alt: "datetime"})
        idx_as_dt = pd.to_datetime(df.index, errors="coerce")
        if idx_as_dt.notna().all():
            out = df.reset_index().rename(columns={"index": "datetime"})
            out["datetime"] = idx_as_dt
            return out
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

    if "orig_missing_baseline" not in SS:
        SS.orig_missing_baseline = missing_df.isna().copy()
    else:
        if (not SS.orig_missing_baseline.index.equals(missing_df.index) or
            list(SS.orig_missing_baseline.columns) != list(missing_df.columns)):
            SS.orig_missing_baseline = missing_df.isna().copy()

    # >>> ADD (freeze the original values so PriSTI plots can show originals)
    if "orig_missing_values" not in SS:
        SS.orig_missing_values = missing_df.copy()
    # <<< END ADD

    base_index   = missing_df.index
    sim_df       = sim_df.reindex(base_index)
    common_index = base_index
    if common_index.empty or latlng.empty:
        st.info("No matching timeline or positions/sensors.")
        return

    # ---------- persistent state ----------
    uid = init_once("sim_uid", f"sim_{uuid.uuid4().hex[:8]}")
    init_once("sim_iter", 0)
    init_once("sim_ptr", 0)  # position dans la timeline
    if "imputed_mask" not in SS:
        SS.imputed_mask = pd.DataFrame(False, index=common_index, columns=sensor_cols, dtype=bool)
    SS.imputed_mask = SS.imputed_mask.reindex(index=common_index, columns=sensor_cols, fill_value=False)

    init_once("sliding_window_df", pd.DataFrame(columns=["datetime"] + list(sensor_cols)))
    init_once("global_df", pd.DataFrame(columns=["datetime"] + list(sensor_cols)))
    init_once("impute_time_tsg", {})  # per-timestamp seconds
    init_once("impute_time_pri", {})




    # ---------- UI (created once) ----------
    if "_ui_inited" not in SS:
        SS["_ui_inited"] = True

        # Row 1: two columns: left (title, time, button, map) | right (gauge)
        col_left, col_right = st.columns([3, 1], gap="small")

        with col_left:
            st.markdown("### Sensor Visualization")
            hdr_l, hdr_r = st.columns([5, 2], gap="small")
            with hdr_l:
                SS["ph_time"] = st.markdown("**Current Time:** —")
            with hdr_r:
                SS["ph_fitbtn"] = st.empty()     # we render the button each run below

            # map placeholder lives in the left column
            SS["ph_map"] = st.empty()

        with col_right:
            st.markdown(
                "<div style='text-align:center;font-weight:700;margin:0.25rem 0'>Data Quality & Activity</div>",
                unsafe_allow_html=True
            )
            SS["ph_counts_active"]  = st.empty()
            SS["ph_counts_missing"] = st.empty()
            SS["ph_gauge"] = st.empty()

        st.markdown("---")

        # Row 2: Global TS (left) + snapshot-10 (right)
        row2_l, row2_r = st.columns([3, 2], gap="small")
        with row2_l:
            SS["ph_global"] = st.empty()
        with row2_r:
            SS["ph_snap10"] = st.empty()

        st.markdown("---")


    # Render a single Fit button (in left header, aligned right)
    with SS["ph_fitbtn"]:
        # unique, stable key
        if st.button("Fit map to sensors", use_container_width=True, key=f"{uid}_fitbtn"):
            SS["_fit_event"] = True

    # ---------- deck.gl persistent ----------
    global ICON_SPEC
    if "ICON_SPEC" not in globals() or ICON_SPEC is None:
        ICON_SPEC = {"url": "", "width": 1, "height": 1, "anchorX": 0, "anchorY": 0}

    if "deck_obj" not in SS:
        base_df = latlng.copy()
        base_df["sensor_id"] = base_df["sensor_id"].astype(str)
        base_df["value"] = "NA"
        base_df["status"] = "Predicted"
        base_df["bg_color"] = [[231, 76, 60, 200] for _ in range(len(base_df))]
        base_df["bg_radius"] = 10
        base_df["icon"] = [ICON_SPEC] * len(base_df)
        base_df["icon_size"] = 1.0

        init_view = fit_view_simple(base_df)
        SS["deck_tooltip"] = {"text": "Sensor {sensor_id}\nValue: {value}\nStatus: {status}"}
        SS.deck_obj = pdk.Deck(
            layers=[make_bg_layer(base_df), make_icon_layer(base_df)],
            initial_view_state=init_view,
            map_style="mapbox://styles/mapbox/light-v11",
            tooltip=SS["deck_tooltip"],
        )
        SS["_fit_base_df"] = base_df.copy()

    # Handle Fit event: REBUILD the deck with a NEW view state, then re-render
    if SS.get("_fit_event", False):
        df_to_fit = SS.get("_fit_base_df", latlng)
        new_view = fit_view_simple(df_to_fit)
        layers = SS.deck_obj.layers
        # rebuild deck to force refit
        SS.deck_obj = pdk.Deck(
            layers=layers,
            initial_view_state=new_view,
            map_style=getattr(SS.deck_obj, "map_style", "mapbox://styles/mapbox/light-v11"),
            tooltip=SS.get("deck_tooltip", {"text": "Sensor {sensor_id}\nValue: {value}\nStatus: {status}"}),
        )
        SS["ph_map"].pydeck_chart(SS.deck_obj, use_container_width=True)
        SS["_fit_event"] = False

    # Initial map draw (if not drawn by fit)
    SS["ph_map"].pydeck_chart(SS.deck_obj, use_container_width=True)

    # ---------- palettes ----------
    base_palette = ["#000000", "#003366", "#009999", "#006600", "#66CC66",
                    "#FF9933", "#FFD700", "#708090", "#4682B4", "#99FF33"]
    sensor_color_map = {c: base_palette[i % len(base_palette)] for i, c in enumerate(sensor_cols)}


    # add a place to show alerts (non-intrusive)
    if "ph_alerts" not in SS:
        SS["ph_alerts"] = st.empty()

    # missing-streak tracker (by data_col) for scenario 1 thresholding
    if "_missing_streak_hours" not in SS:
        SS["_missing_streak_hours"] = {c: 0 for c in sensor_cols}
    else:
        # keep in sync with current selection
        for c in sensor_cols:
            SS["_missing_streak_hours"].setdefault(c, 0)

    # ---------- main loop (resume from pointer) ----------
    use_model = model is not None
    if use_model:
        model.eval()

    SNAP10 = 10
    ptr = int(SS["sim_ptr"])
    total_steps = len(common_index)

    while ptr < total_steps:
        ts = pd.Timestamp(common_index[ptr])
        SS["sim_iter"] += 1
        iter_key = SS["sim_iter"]
        baseline_row_ts = SS.orig_missing_baseline.reindex(index=[ts], columns=sensor_cols).iloc[0]

        # time label
        SS["ph_time"].markdown(f"<div style='font-weight:600'>Current Time: {ts}</div>", unsafe_allow_html=True)

        # history window strictly before ts
        hist_end = ts - pd.Timedelta(hours=1)
        if hist_end in missing_df.index:
            hist_idx = missing_df.loc[:hist_end].index[-window_hours:]
        else:
            hist_idx = missing_df.index[missing_df.index < ts][-window_hours:]
        hist_win = missing_df.loc[hist_idx, sensor_cols] if len(hist_idx) > 0 else pd.DataFrame()

        # --- TSGuard imputation (alignée baseline) ---
        tsg_start = time.perf_counter()
        baseline_row_ts = SS.orig_missing_baseline.reindex(index=[ts], columns=sensor_cols).iloc[0]

        svals, sstatus = [], []

        for col in sensor_cols:
            is_missing_now = bool(baseline_row_ts.get(col, False))

            if is_missing_now:
                pred_val = np.nan

                if not hist_win.empty:
                    if use_model:
                        try:
                            pred_val = predict_single_missing_value(
                                historical_window=np.asarray(hist_win.values, dtype=np.float32),
                                target_sensor_index=col_to_idx[col],
                                model=model, scaler=scaler, inv_scaler=inv_scaler, device=device
                            )
                        except Exception:
                            pred_val = np.nan
                    if (not np.isfinite(pred_val)) or pd.isna(pred_val):
                        # fallback: dernier réel dans l'historique
                        last = pd.to_numeric(hist_win[col].dropna(), errors="coerce")
                        pred_val = float(last.iloc[-1]) if len(last) else np.nan

                # write + flags
                try:
                    missing_df.at[ts, col] = pred_val if pd.notna(pred_val) else np.nan
                except Exception:
                    pass
                svals.append(pred_val)
                sstatus.append(False)                     # imputé (rouge sur la map)
                SS.imputed_mask.at[ts, col] = pd.notna(pred_val)

            else:
                # présent à l'origine → garder la valeur
                v = missing_df.at[ts, col] if (ts in missing_df.index and col in missing_df.columns) else np.nan
                svals.append(v)
                sstatus.append(True)                      # réel (vert sur la map)
                SS.imputed_mask.at[ts, col] = False

        SS["impute_time_tsg"][ts] = time.perf_counter() - tsg_start

        # Sécurité : parité des tailles
        if len(svals)   < len(sensor_cols): svals   += [np.nan] * (len(sensor_cols) - len(svals))
        if len(sstatus) < len(sensor_cols): sstatus += [False]  * (len(sensor_cols) - len(sstatus))

        # ---- Dicts capteur -> valeur/état ----
        vals_by_col = dict(zip(sensor_cols, svals))
        real_by_col = dict(zip(sensor_cols, sstatus))
        imputed_row = SS.imputed_mask.reindex(index=[ts], columns=sensor_cols, fill_value=False).iloc[0]

        # --- Update missing streak (Scénario 1) ---
        if "_missing_streak_hours" not in SS:
            SS["_missing_streak_hours"] = {c: 0.0 for c in sensor_cols}

        for c in sensor_cols:
            originally_missing = bool(baseline_row_ts.get(c, False))
            val_now = vals_by_col.get(c, np.nan)
            no_value_now = (val_now is None) or (isinstance(val_now, float) and np.isnan(val_now))
            # On incrémente uniquement quand il manquait à l’origine ET qu’on n’a toujours pas de valeur à afficher
            if originally_missing and no_value_now:
                SS["_missing_streak_hours"][c] = SS["_missing_streak_hours"].get(c, 0.0) + 1.0  # +1h par tick
            else:
                SS["_missing_streak_hours"][c] = 0.0

        # ---- Vérification des contraintes & alertes (appelle TA fonction) ----
        verify_constraints_and_alerts_for_timestamp(
            ts=ts,
            latlng_df=latlng[["sensor_id", "data_col", "latitude", "longitude"]],
            values_by_col=vals_by_col,
            imputed_mask_row=imputed_row,
            baseline_row=baseline_row_ts,
        )

        # ---- Buffers de rendu (inchangé) ----
        row = {"datetime": ts}
        for i, c in enumerate(sensor_cols):
            row[c] = svals[i]
        SS.sliding_window_df.loc[len(SS.sliding_window_df)] = row
        SS.global_df.loc[len(SS.global_df)] = row
        if len(SS.sliding_window_df) > 36:
            SS.sliding_window_df = SS.sliding_window_df.tail(36)


        # --- Map update & store base for next Fit ---
        # --- juste avant la mise à jour de la carte ---
        # (garde les mêmes noms de variables que ton code)
        if len(svals) != len(sensor_cols) or len(sstatus) != len(sensor_cols):
            if not st.session_state.get("_warned_len_mismatch", False):
                st.warning(
                    f"[Guard] Mismatch tailles — sensors={len(sensor_cols)}, svals={len(svals)}, sstatus={len(sstatus)}. "
                    "On complète seulement ce tick."
                )
                st.session_state["_warned_len_mismatch"] = True

        # mapping sûr (pas d'IndexError); les capteurs sans valeur recevront NaN/False et la map affichera 'NA'
        vals_by_col = dict(zip(sensor_cols, svals))
        real_by_col = dict(zip(sensor_cols, sstatus))
        tick_df = latlng.copy()
        tick_df["value"]  = tick_df["data_col"].map(vals_by_col).fillna("NA")
        tick_df["status"] = tick_df["data_col"].map(lambda c: "Real" if real_by_col.get(c, False) else "Predicted")
        tick_df["bg_color"] = [GREEN if s == "Real" else RED for s in tick_df["status"]]
        tick_df["bg_radius"] = 10
        tick_df["icon"] = [ICON_SPEC] * len(tick_df)
        tick_df["icon_size"] = 1.0
        SS.deck_obj.layers = [make_bg_layer(tick_df), make_icon_layer(tick_df)]
        SS["_fit_base_df"] = tick_df.copy()
        SS["ph_map"].pydeck_chart(SS.deck_obj, use_container_width=True)

        # --- Gauge & counts (CUMULATIVE MISSED from the beginning to now) ---
        baseline_mask_to_now = SS.orig_missing_baseline.loc[:ts, sensor_cols]
        cumulative_missed = int(baseline_mask_to_now.values.sum())  # total # of originally-missing cells up to ts
        total_cells_to_now = baseline_mask_to_now.size
        pct_missed_to_now = (cumulative_missed / total_cells_to_now * 100.0) if total_cells_to_now else 0.0

        # Active sensors NOW (same as before, just for info)
        row_imp = SS.imputed_mask.reindex(index=[ts], columns=sensor_cols, fill_value=False).iloc[0]
        imputed_now = int(row_imp.sum())
        sensors_total = max(1, len(sensor_cols))
        real_now = sensors_total - imputed_now

        # Per-timestamp (NOW) counts that match the map colors:
        missed_now = int(baseline_row_ts.sum())
        active_now = len(sensor_cols) - missed_now

        SS["ph_counts_active"].markdown(f"Active sensors now: **{active_now}**")
        SS["ph_counts_missing"].markdown(f"Delayed sensors now: **{missed_now}**")

        # Gauge shows MISSED DATA (%) cumulatively
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct_missed_to_now,
            title={"text": "Missed Data (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if pct_missed_to_now >= DEFAULT_VALUES["gauge_red_max"] else "green"},
                "steps": [
                    {"range": [DEFAULT_VALUES["gauge_green_min"], DEFAULT_VALUES["gauge_green_max"]],
                     "color": "lightgreen"},
                    {"range": [DEFAULT_VALUES["gauge_yellow_min"], DEFAULT_VALUES["gauge_yellow_max"]],
                     "color": "yellow"},
                    {"range": [DEFAULT_VALUES["gauge_red_min"], DEFAULT_VALUES["gauge_red_max"]], "color": "red"},
                ],
            },
        ))
        gauge_fig.update_layout(title="",margin=dict(l=10, r=10, t=30, b=10))
        lightify(gauge_fig)
        SS["ph_gauge"].plotly_chart(gauge_fig, use_container_width=True, key=f"{uid}_gauge_{iter_key}")

        # --- Global TS ---
        full_ts_fig = draw_full_time_series_with_mask_gap(SS.global_df.copy(), SS.imputed_mask, sensor_cols, sensor_color_map)
        lightify(full_ts_fig)
        SS["ph_global"].plotly_chart(full_ts_fig, use_container_width=True, key=f"{uid}_global_{iter_key}")

        # --- Snapshot 10 (no legend) ---
        snap10 = SS.sliding_window_df.tail(SNAP10)
        snap10_fig = go.Figure()
        for col in sensor_cols:
            base_color = sensor_color_map[col]
            sub = (snap10[["datetime", col]].dropna()
                   .sort_values("datetime").rename(columns={col: "value"}))
            if sub.empty:
                continue
            def is_imp(t, c=col):
                try: return bool(SS.imputed_mask.loc[t, c])
                except: return False
            add_imputed_segments(snap10_fig, sub, is_imp, base_color, gap_hours=6)
        snap10_fig.update_layout(title="Snapshot (last 10)",
                                 xaxis_title="Time", yaxis_title="Value",
                                 margin=dict(l=20, r=20, t=40, b=20),
                                 showlegend=False)
        lightify(snap10_fig)

        SS["ph_snap10"].plotly_chart(snap10_fig, use_container_width=True, key=f"{uid}_snap10_{iter_key}")

        # advance pointer
        ptr += 1
        SS["sim_ptr"] = ptr
        time.sleep(1)

def run_tsguard_vs_pristi_comparison(
    sim_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    model: torch.nn.Module,
    scaler: callable,
    inv_scaler: callable,
    device: torch.device,
    eval_len: int = 36,
    window_hours: int = 24,
):
    import time
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    from utils.config import DEFAULT_VALUES
    def _last_observed(hist_df: pd.DataFrame, col: str) -> float:
        if hist_df.empty or col not in hist_df.columns:
            return np.nan
        s = pd.to_numeric(hist_df[col].dropna(), errors="coerce")
        return float(s.iloc[-1]) if len(s) else np.nan

    SS = st.session_state
    print("Running TSGuard vs PTISTI")

    # --- Header & notices ---
    st.markdown("## TSGuard vs PriSTI — Comparison")
    st.warning("⚠️ The comparison can be long because PriSTI computation is heavy.", icon="⚠️")
    ph_status = st.info("⏳ Preparing the first window…", icon="⏳")

    # ---------- Align time & columns (same as your sim) ----------
    def ensure_datetime(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "datetime" in df.columns:
            df = df.copy(); df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"]).set_index("datetime")
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy(); df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()]
        df.index = df.index.floor("h")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        return df

    sim_df     = ensure_datetime(sim_df, "sim_df")
    missing_df = ensure_datetime(missing_df, "missing_df")

    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [(str(c).strip().zfill(6) if str(c).isdigit() else str(c).strip()) for c in out.columns]
        return out

    sim_df = _norm_cols(sim_df)
    missing_df = _norm_cols(missing_df)

    # columns like your sim (keep order)
    all_sensor_cols = [c for c in sim_df.columns if c != "datetime"]
    graph_size = int(SS.get("graph_size", DEFAULT_VALUES["graph_size"]))
    sensor_cols = [str(c) for c in all_sensor_cols[:graph_size]]

    # baseline: originally missing
    orig_missing_baseline = missing_df.isna().copy()
    # keep ORIGINAL values for PriSTI display
    orig_values_frozen = missing_df.copy()

    # reindex both on common timeline
    common_index = missing_df.index
    sim_df = sim_df.reindex(common_index)
    if common_index.empty:
        st.info("No timeline to compare."); return

    # PriSTI artifacts (as in your helpers)
    PRISTI_ROOT = "./PriSTI"
    CONFIG_PATH = f"{PRISTI_ROOT}/config/base.yaml"
    WEIGHTS_PATH = f"{PRISTI_ROOT}/save/aqi36/model.pth"
    MEANSTD_PK   = f"{PRISTI_ROOT}/data/pm25/pm25_meanstd.pk"
    pristi_ready = (len(missing_df.columns) >= 36
                    and os.path.exists(CONFIG_PATH)
                    and os.path.exists(WEIGHTS_PATH)
                    and os.path.exists(MEANSTD_PK))
    if not pristi_ready:
        st.error("PriSTI backend not available (need config/weights/meanstd and ≥36 sensors)."); return

    pristi_cols = list(missing_df.columns)[:36]
    pristi_model, pristi_mean, pristi_std = load_pristi_artifacts(CONFIG_PATH, WEIGHTS_PATH, MEANSTD_PK, device)
    SS["pristi_model"] = pristi_model; SS["pristi_mean"] = pristi_mean; SS["pristi_std"] = pristi_std

    # Regulators (keep your logic / names)
    c1, c2 = st.columns(2)
    with c1:
        cmp_sensors = st.slider("Sensors to display", 1, min(36, len(pristi_cols)),
                                min(6, len(pristi_cols)), key="cmp_sensors")
    with c2:
        cmp_steps = st.slider("Timestamps to display (≤36, last)", 6, 36, 36, key="cmp_steps")

    # Output areas
    ph_times = st.empty()
    lcol, rcol = st.columns(2)
    ph_tsg = lcol.empty()
    ph_pri = rcol.empty()

    # Buffers (comparison runs from zero; no map/other charts)
    tsg_running_df   = missing_df.copy()
    pristi_running_df = missing_df.copy()

    # color palette for lines
    base_palette = [
        "#000000", "#003366", "#009999", "#006600", "#66CC66",
        "#FF9933", "#FFD700", "#708090", "#4682B4", "#99FF33"
    ]
    color_map = {c: base_palette[i % len(base_palette)] for i, c in enumerate(pristi_cols)}

    # iteration from the beginning
    col_to_idx = {c: i for i, c in enumerate(sensor_cols)}
    use_model = model is not None
    if use_model: model.eval()

    start_time = time.perf_counter()
    first_window_done = False

    for i, ts in enumerate(common_index):
        ts = pd.Timestamp(ts)

        # ---- TSGuard: impute ONLY where originally missing at ts (your logic) ----
        hist_end = ts - pd.Timedelta(hours=1)
        if hist_end in tsg_running_df.index:
            hist_idx = tsg_running_df.loc[:hist_end].index[-window_hours:]
        else:
            hist_idx = tsg_running_df.index[tsg_running_df.index < ts][-window_hours:]
        if len(hist_idx) > 0:
            hist_win = tsg_running_df.loc[hist_idx, sensor_cols].copy()
        else:
            # DataFrame vide MAIS avec les mêmes colonnes pour éviter KeyError
            hist_win = pd.DataFrame(columns=sensor_cols)

        baseline_row = orig_missing_baseline.reindex(index=[ts], columns=sensor_cols).iloc[0]
        for col in sensor_cols:
            if bool(baseline_row[col]):  # originally missing → impute
                if not hist_win.empty and use_model:
                    try:
                        pred_val = predict_single_missing_value(
                            historical_window=np.asarray(hist_win.values, dtype=np.float32),
                            target_sensor_index=col_to_idx[col],
                            model=model, scaler=scaler, inv_scaler=inv_scaler, device=device
                        )
                    except Exception:
                        pred_val = _last_observed(hist_win, col)
                else:
                    pred_val = _last_observed(hist_win, col)

                tsg_running_df.at[ts, col] = pred_val if pd.notna(pred_val) else np.nan

            # else: keep the original value at ts

        # ---- PriSTI: when we have ≥ eval_len steps, impute the last 36 window ----
        try:
            end_loc = pristi_running_df.index.get_loc(ts)
        except KeyError:
            end_loc = None

        if isinstance(end_loc, int) and (end_loc + 1) >= eval_len:
            start_loc = end_loc - (eval_len - 1)
            time_index = pristi_running_df.index[start_loc:end_loc + 1]

            pri_t0 = time.perf_counter()
            updated_df, info = impute_window_with_pristi(
                missing_df=pristi_running_df.copy(),
                sensor_cols=pristi_cols,
                target_timestamp=ts,
                model=pristi_model,
                device=device, eval_len=eval_len, nsample=100
            )
            pri_dt = time.perf_counter() - pri_t0
            if info == "ok":
                pristi_running_df.loc[time_index, pristi_cols] = updated_df.loc[time_index, pristi_cols].values

            # ---- Build & show comparison figures (ONLY the comparison) ----
            cols_show = pristi_cols[:cmp_sensors]
            mask_block = orig_missing_baseline.reindex(index=time_index, columns=pristi_cols).fillna(False)

            tsg_block = tsg_running_df.loc[time_index, pristi_cols].copy()
            pri_block = pristi_running_df.loc[time_index, pristi_cols].copy()

            # PriSTI display = original values, replaced ONLY on originally-missing cells
            base_block = orig_values_frozen.loc[time_index, pristi_cols].copy()
            display_pri = base_block.copy()
            display_pri[mask_block] = pri_block[mask_block]

            def _window_lines(fig, block, mask_df, cmap):
                sub_df = block.reset_index()
                first_col = sub_df.columns[0]
                if first_col != "datetime":
                    sub_df = sub_df.rename(columns={first_col: "datetime"})
                for col in block.columns:
                    if col not in cols_show: continue
                    base_color = cmap.get(col, "#444")
                    xy = (sub_df[["datetime", col]].dropna()
                          .sort_values("datetime").rename(columns={col: "value"}))
                    if xy.empty: continue
                    def is_imp(t):
                        try: return bool(mask_df.loc[t, col])
                        except: return False
                    add_imputed_segments(fig, xy, is_imp, base_color, gap_hours=6)
                for col in cols_show:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                             marker=dict(size=8, color=cmap.get(col, "#444")),
                                             showlegend=True, name=f"{col}"))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                         marker=dict(size=8, color="red"),
                                         showlegend=True, name="Imputed segment"))

            cmap = {c: color_map[c] for c in cols_show}
            tsg_show  = tsg_block.iloc[-cmp_steps:, :][cols_show]
            pri_show  = display_pri.iloc[-cmp_steps:, :][cols_show]
            mask_show = mask_block.iloc[-cmp_steps:, :][cols_show]

            fig_tsg = go.Figure(); _window_lines(fig_tsg, tsg_show, mask_show, cmap)
            fig_pri = go.Figure(); _window_lines(fig_pri, pri_show, mask_show, cmap)
            for _f in (fig_tsg, fig_pri):
                _f.update_layout(title=None, xaxis_title="Time", yaxis_title="Value",
                                 margin=dict(l=10, r=10, t=20, b=10))
                lightify(_f)

            ph_times.markdown(
                f"**Imputation time @ {ts}** — PriSTI: {pri_dt*1000:.1f} ms  •  TSGuard computed per-step."
            )
            ph_tsg.plotly_chart(fig_tsg, use_container_width=True, key=f"cmp_tsg_{int(ts.value)}")
            ph_pri.plotly_chart(fig_pri, use_container_width=True, key=f"cmp_pri_{int(ts.value)}")

            if not first_window_done:
                first_window_done = True
                ph_status.success("First 36-step window ready. Comparison is running.", icon="✅")

        else:
            # still building up the first 36 steps
            have = (end_loc + 1) if isinstance(end_loc, int) else 0
            need = max(0, eval_len - have)
            ph_status.info(f"⏳ Preparing the first window… need {need} more timestamp(s).", icon="⏳")

        # pacing a little so UI updates are visible
        time.sleep(0.1)

