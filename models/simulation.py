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
# ---- Dismissible alerts -----------------------------------------------------
from typing import Optional
from uuid import uuid4


# ---------- Sticky alerts (top-right) ----------
import time
import streamlit as st

def _init_alert_store(max_items:int=20):
    SS = st.session_state
    if "_alert_store" not in SS:
        # chaque entrée: {id, level, text, dedup, ts, dismissed, category, group}
        SS["_alert_store"] = []
    if "_alert_seq" not in SS:
        SS["_alert_seq"] = 0
    SS["_alert_max"] = max_items

def push_alert(level: str, text: str, dedup_key: str = None,
               category: str = "general", group: str = "general"):
    """
    level ∈ {"success","info","warning","error"}
    category ∈ {"scenario","constraint","general", ...}
    group: sous-groupe d'affichage (ex: "Scenario 1" / "Temporal" / "Spatial")
    """
    _init_alert_store()
    store = st.session_state["_alert_store"]

    if dedup_key:
        for a in store:
            if (not a.get("dismissed")) and a.get("dedup")==dedup_key and a.get("text")==text:
                a["ts"] = time.time()   # rafraîchit
                return

    st.session_state["_alert_seq"] += 1
    store.append({
        "id": st.session_state["_alert_seq"],
        "level": level, "text": text,
        "dedup": dedup_key, "ts": time.time(),
        "dismissed": False,
        "category": category, "group": group
    })

    # trim (on supprime d'abord les dismissed, sinon le plus ancien)
    max_items = st.session_state.get("_alert_max", 20)
    if len(store) > max_items:
        idx = next((i for i,a in enumerate(store) if a.get("dismissed")), 0)
        store.pop(idx)

def render_alert_center(max_items: int = 12, *, key_suffix: str = "", placeholder=None):
    """
    Rend un stack d'alertes regroupées.
    - key_suffix: p.ex. "tick_123" pour rendre les keys uniques à chaque tick
    - placeholder: st.empty() persistant (obligatoire pour réécrire le bloc)
    """
    _init_alert_store(max_items)
    if placeholder is None:
        placeholder = st.empty()  # fallback

    palette = {"success":"#16a34a","info":"#2563eb","warning":"#f59e0b","error":"#dc2626"}
    icon    = {"success":"✅","info":"ℹ️","warning":"⚠️","error":"🚨"}
    severity = {"error":3, "warning":2, "info":1, "success":0}

    # éléments visibles, les plus récents d'abord
    visible = [a for a in st.session_state["_alert_store"] if not a.get("dismissed")]
    visible = visible[-max_items:][::-1]

    # ---- regroupement (catégorie, groupe) ----
    groups = {}  # (cat, grp) -> {"items":[...], "level":worst_level}
    for a in visible:
        gk = (a.get("category","general"), a.get("group", a.get("category","general")))
        if gk not in groups:
            groups[gk] = {"items": [], "level": a["level"]}
        groups[gk]["items"].append(a)
        # niveau = le plus sévère du groupe
        if severity[a["level"]] > severity[groups[gk]["level"]]:
            groups[gk]["level"] = a["level"]

    # ---- rendu dans le placeholder ----
    with placeholder.container():
        # CSS simple; pas de :has() (plus sûr)
        st.markdown("""
        <style>
          .alert-root { position: fixed; top: 86px; right: 22px; width: 420px; z-index: 99999; }
          .alert-card{
            background: #fff; border: 1px solid #e5e7eb; border-left-width: 6px;
            border-radius: 10px; padding: 10px 12px; margin-bottom: 10px;
            box-shadow: 0 8px 22px rgba(0,0,0,.10);
          }
          .alert-row{ display:flex; align-items:flex-start; gap:10px; }
          .alert-x button{ border:none; background:transparent; padding:0 4px; font-size:16px; line-height:1; }
          .alert-title{ font-weight:600; margin-bottom:6px; }
          .alert-list{ margin:0; padding-left:18px; }
          .alert-list li{ margin:2px 0; }
        </style>
        """, unsafe_allow_html=True)

        # racine fixe (on l'entoure d'un container Streamlit)
        anchor = st.container()
        with anchor:
            st.markdown("<div class='alert-root'>", unsafe_allow_html=True)

            for (cat, grp), data in groups.items():
                lvl = data["level"]
                color = palette.get(lvl, "#2563eb")
                head = f"{icon.get(lvl,'ℹ️')} {cat.capitalize()} — {grp}"

                # bouton ✕ de groupe (ferme toutes les entrées du groupe)
                cols = st.columns([12,1], gap="small")
                with cols[0]:
                    st.markdown(
                        f"<div class='alert-card' style='border-left-color:{color}'>"
                        f"<div class='alert-title'>{head}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<ul class='alert-list'>", unsafe_allow_html=True)
                    for it in data["items"]:
                        st.markdown(f"<li>{it['text']}</li>", unsafe_allow_html=True)
                    st.markdown("</ul></div>", unsafe_allow_html=True)
                with cols[1]:
                    if st.button("✕", key=f"alert_group_close_{cat}_{grp}_{key_suffix}"):
                        for it in data["items"]:
                            it["dismissed"] = True
                        # pas de rerun ici

            st.markdown("</div>", unsafe_allow_html=True)

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
            push_alert(
                "info",
                f"🕒 Scenario 1 — {sid} @ {ts}: delay below Δt, waiting for late data.",
                dedup_key=f"s1-{sid}"
            )
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
                    # Scenario 2.1
                    alerts.append(("info", f"✅ {sid} @ {ts}: imputed within range; neighbors in range."))
                elif own_vios:
                    # Scenario 2.2
                    alerts.append(("error", f"🚨 {sid} @ {ts}: imputed value violates constraints ({'; '.join(own_vios)})."))
                elif any_neigh_vio:
                    # Scenario 2.3
                    alerts.append(("warning", f"⚠️ {sid} @ {ts}: neighbors out-of-range; possible masked anomaly."))
            else:
                # Scenario 3 — no neighbors
                alerts.append(("warning", f"🛰️ {sid} @ {ts}: neighbors unavailable; fallback to history."))
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
        st.success(f"✅ Model saved to: `{model_path}` ; Scaler saved to: `{scaler_json}`")
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

from collections import defaultdict
import re

# sévérité pour choisir le pire niveau d'un groupe
_SEV = {"success":0, "info":1, "warning":2, "error":3}

# =============== Alertes groupées (boîtes fermables) ===============
import re
import time
import pandas as pd
import streamlit as st

_SEV_ORDER = {"success": 0, "info": 1, "warning": 2, "error": 3}

def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "", str(s))


def add_group_alert(group_title: str, level: str, message: str, ts: pd.Timestamp):
    """
    Ajoute une 'note' dans la boîte identifiée par (ts, group_title).
    'level' met à jour la sévérité de la boîte si plus grave.
    """
    _init_group_store()
    SS = st.session_state
    ts = pd.Timestamp(ts)
    gid = f"{ts.isoformat()}|{group_title}"
    iid = f"{gid}|{hash(message)}"
    if iid in SS["_grp_seen"]:
        return

    # chercher ou créer le groupe
    grp = next((g for g in SS["_grp_alerts"] if g["gid"] == gid), None)
    if grp is None:
        grp = {
            "gid": gid,
            "title": group_title,
            "level": level,
            "ts": ts,
            "items": [],
            "dismissed": False,
        }
        SS["_grp_alerts"].append(grp)

    # escalade niveau si besoin
    if _SEV_ORDER.get(level, 1) > _SEV_ORDER.get(grp["level"], 1):
        grp["level"] = level

    # ajouter l'item
    grp["items"].append({"iid": iid, "text": message, "dismissed": False})
    SS["_grp_seen"].add(iid)

    # trimming (garder récentes)
    max_groups = SS.get("_grp_max", 40)
    active = [g for g in SS["_grp_alerts"] if not g.get("dismissed")]
    if len(active) > max_groups:
        # ferme la plus ancienne
        oldest_idx = min(range(len(SS["_grp_alerts"])), key=lambda i: SS["_grp_alerts"][i]["ts"])
        SS["_grp_alerts"][oldest_idx]["dismissed"] = True

import re
from uuid import uuid4

def _stable_key(seed: str) -> str:
    # sanitize + add a stable hash suffix so weird chars (|, :, —) are safe
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(seed))
    return f"{base}_{abs(hash(seed)) & 0xFFFF_FFFF:X}"

def _init_group_store():
    SS = st.session_state
    if "_grp_alerts" not in SS:
        # dict: gid -> {gid, title, level, ts, items:[{iid,text,dismissed}], dismissed}
        SS["_grp_alerts"] = {}
    elif isinstance(SS["_grp_alerts"], list):
        # MIGRATION from the older list-based store to dict to avoid double-rendering
        d = {}
        for g in SS["_grp_alerts"]:
            gid = g.get("gid") or f"{g.get('ts')}|{g.get('title')}|{g.get('level','info')}"
            d[gid] = {
                "gid": gid,
                "title": g.get("title",""),
                "level": g.get("level","info"),
                "ts": pd.Timestamp(g.get("ts")),
                "items": g.get("items", []),
                "dismissed": g.get("dismissed", False),
            }
        SS["_grp_alerts"] = d
    if "_grp_ph" not in SS:
        SS["_grp_ph"] = st.empty()
    if "_grp_render_seq" not in SS:
        SS["_grp_render_seq"] = 0

def render_grouped_alerts():
    _init_group_store()
    SS = st.session_state

    # bump a per-render sequence to guarantee unique widget keys per run
    SS["_grp_render_seq"] += 1
    render_suffix = f"__r{SS['_grp_render_seq']}"

    ph = SS["_grp_ph"]
    ph.empty()

    st.markdown("""
    <style>
      div[data-testid="stVerticalBlock"]:has(> div#grp-alert-anchor){
        position: fixed; top: 86px; right: 22px; width: 420px; z-index: 99999;
      }
      .grp-card{background:#fff;border:1px solid #e5e7eb;border-left-width:6px;
                border-radius:12px;padding:10px 12px;margin:10px 0;box-shadow:0 8px 22px rgba(0,0,0,.06);}
      .grp-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;}
      .grp-title{font-weight:700}
      .grp-dot{margin-top:4px}
    </style>
    <div id="grp-alert-anchor"></div>
    """, unsafe_allow_html=True)

    palette = {"success":"#16a34a","info":"#2563eb","warning":"#f59e0b","error":"#dc2626"}
    icon    = {"success":"✅","info":"ℹ️","warning":"⚠️","error":"🚨"}

    groups = [g for g in SS["_grp_alerts"].values() if not g.get("dismissed")]
    groups.sort(key=lambda g: g["ts"], reverse=True)

    with ph.container():
        for g in groups:
            sev = g.get("level", "info")
            color = palette.get(sev, "#2563eb")
            gid = str(g["gid"])
            gid_key = _stable_key(gid)  # sanitized + hashed

            st.markdown(f"<div class='grp-card' style='border-left-color:{color}'>",
                        unsafe_allow_html=True)
            h1, h2 = st.columns([12,1], gap="small")
            with h1:
                st.markdown(
                    f"<div class='grp-head'><div class='grp-title'>{icon.get(sev,'ℹ️')} {g['title']}</div>"
                    f"<div style='opacity:.6;font-size:12px'>{g['ts']}</div></div>",
                    unsafe_allow_html=True
                )
            with h2:
                if st.button("✕", key=f"grpclose_{gid_key}{render_suffix}"):
                    g["dismissed"] = True

            live_items = [it for it in g["items"] if not it.get("dismissed")]
            for idx, it in enumerate(live_items):
                c1, c2, c3 = st.columns([1,22,1], gap="small")
                with c1:
                    st.markdown("<div class='grp-dot'>•</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(it["text"])
                with c3:
                    if st.button("✕", key=f"itemclose_{gid_key}_{idx}{render_suffix}"):
                        it["dismissed"] = True

            st.markdown("</div>", unsafe_allow_html=True)

class TickAlertBuffer:
    """Collecte les lignes puis pousse 1 groupe par (title, level) pour le timestamp ts."""
    def __init__(self):
        self.buckets = {}  # key=(title, level) -> list[str]

    def add(self, title: str, level: str, text: str):
        self.buckets.setdefault((title, level), []).append(text)

    def flush(self, ts):
        _init_group_store()
        store = st.session_state["_grp_alerts"]
        ts = pd.Timestamp(ts)
        for (title, level), lines in self.buckets.items():
            # gid unique = ts + title + level  (=> 1 boîte par type à ce tick)
            gid = f"{ts.isoformat()}|{title}|{level}"
            g = store.get(gid)
            if not g:
                g = {"gid": gid, "title": title, "level": level, "ts": ts,
                     "items": [], "dismissed": False}
                store[gid] = g
            # ajoute les lignes (chacune fermable)
            for line in lines:
                g["items"].append({"iid": uuid4().hex, "text": line, "dismissed": False})
        self.buckets.clear()



def verify_constraints_and_alerts_for_timestamp(
    ts: pd.Timestamp,
    latlng_df: pd.DataFrame,
    values_by_col: dict,
    imputed_mask_row: Optional[pd.Series],
    baseline_row: pd.Series,
    missing_streak_hours: Optional[dict] = None,
):
    import numpy as np
    import streamlit as st
    from utils.config import DEFAULT_VALUES

    buf = TickAlertBuffer()  # <-- NOUVEAU
    constraints = st.session_state.get("constraints", [])
    sigma_minutes = float(st.session_state.get("sigma_threshold", DEFAULT_VALUES.get("sigma_threshold", 30)))
    sigma_hours = max(0.0, sigma_minutes / 60.0)

    if "_fault_state" not in st.session_state:
        st.session_state["_fault_state"] = {}
    fault_state = st.session_state["_fault_state"]

    spatial_rules  = [c for c in constraints if c.get("type") == "Spatial"]
    temporal_rules = [c for c in constraints if c.get("type") == "Temporal"]

    if spatial_rules:
        pos_map = {str(r["data_col"]): (float(r["latitude"]), float(r["longitude"]))
                   for _, r in latlng_df.iterrows() if str(r.get("data_col")) in values_by_col}
        def haversine_km(lat1, lon1, lat2, lon2):
            from math import radians, sin, cos, sqrt, atan2
            R=6371.0; dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
            a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
            return 2*R*atan2(sqrt(a), sqrt(1-a))

    for sid, val in values_by_col.items():
        v = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
        was_missing = bool(baseline_row.get(sid, False))
        is_imputed  = bool(imputed_mask_row.get(sid, False)) if imputed_mask_row is not None else False
        streak_h    = float((missing_streak_hours or {}).get(sid, 0.0))

        prev = fault_state.get(sid, "ok")
        curr = prev

        from typing import Optional
        # remplace les buf.add(...) par ceci :
        def _emit(level: str, title: str, line: str, ts):
            # un toast synthétique (titre) + détail sur la 1ère ligne
            toast_notify(level, f"{title} — {line}", dedup_key=f"{title}|{line}|{ts}")

        # ---- Scénarios ----
        if was_missing and (v is None) and not is_imputed:
            if streak_h < sigma_hours:
                if prev != "waiting":
                    # Scenario 1
                    _emit("info", "Attente", f"{sid}: Δt non dépassé (streak {streak_h:.1f}h)", ts)
            else:
                if prev != "fault":
                    # Scenario 3:
                    _emit("warning", "indisponible", f"{sid}: aucune estimation fiable @ {ts}", ts)
        else:
            if prev in ("waiting", "fault"):
                _emit("success", "Rétabli", f"{sid}: valeur à nouveau disponible @ {ts}", ts)

        # ---- Temporel ----
        if temporal_rules and (v is not None):
            mo = ts.strftime("%B")
            for rule in temporal_rules:
                if rule.get("month") != mo: continue
                opt = rule.get("option"); thr = rule.get("temp_threshold", None)
                try: thr = float(thr)
                except: continue
                title = f"Temporal — {mo} | {opt} {thr:g}"
                if opt == "Greater than" and not (v > thr):
                    _emit("warning", f"Temporal — {mo} (> {thr:g})", f"{sid}: {v:.2f} ≤ {thr:g}", ts)
                if opt == "Less than" and not (v < thr):
                    _emit(
                        "warning",
                        f"Temporal — {mo} (< {thr:g})",
                        f"{sid}: {v:.2f} ≥ {thr:g} @ {ts}",
                        ts,
                    )

        # ---- Spatial (no duplicates, clean toast) ----
        if spatial_rules and (v is not None) and ('pos_map' in locals()) and (sid in pos_map):
            lat1, lon1 = pos_map[sid]

            # set local pour éviter A–B et B–A au même ts
            # (on le crée une seule fois par appel de la fonction)
            emitted_pairs = locals().get("_spatial_emitted_pairs")
            if emitted_pairs is None:
                emitted_pairs = set()
                locals()["_spatial_emitted_pairs"] = emitted_pairs

            for rule in spatial_rules:
                try:
                    dist_km = float(rule.get("distance in km", 0))
                    max_diff = float(rule.get("diff", np.inf))
                except Exception:
                    continue
                if not np.isfinite(dist_km) or dist_km <= 0 or not np.isfinite(max_diff):
                    continue

                title = f"Spatial — ≤{dist_km:g} km | maxΔ {max_diff:g}"

                for nid, nval in values_by_col.items():
                    # 1) skip self
                    if nid == sid:
                        continue
                    # 2) skip si pas de position
                    if nid not in pos_map:
                        continue
                    # 3) faire chaque paire une seule fois (ordre lexicographique)
                    a, b = (sid, nid) if sid < nid else (nid, sid)
                    pair_key = (a, b, dist_km, max_diff, pd.Timestamp(ts))
                    if pair_key in emitted_pairs:
                        continue

                    nv = None if (nval is None or (isinstance(nval, float) and np.isnan(nval))) else float(nval)
                    if nv is None:
                        continue

                    lat2, lon2 = pos_map[nid]
                    d = haversine_km(lat1, lon1, lat2, lon2)
                    if d <= dist_km:
                        delta = abs(v - nv)
                        if delta > max_diff:
                            line = f"{a} vs {b}: |Δ|={delta:.2f} > {max_diff:g} (d≈{d:.1f} km) @ {ts}"
                            _emit("warning", title, line, ts)
                            emitted_pairs.add(pair_key)

        # état persisté (anti-spam)
        fault_state[sid] = curr

    # ⚠️ flush UNE SEULE FOIS, après la boucle
    buf.flush(ts)

# ========= Toast notifications instead of grouped cards =========
import time, re


_ICON = {"success":"✅","info":"ℹ️","warning":"⚠️","error":"🚨"}

def _init_toast_state():
    SS = st.session_state
    if "_toast_seen" not in SS:
        SS["_toast_seen"] = set()     # anti-dup court terme
    if "_toast_ttl" not in SS:
        SS["_toast_ttl"] = 8          # secondes d’affichage / anti-spam

def toast_notify(level: str, text: str, dedup_key: str|None = None):
    """
    Envoie une notification toast non bloquante.
    dedup_key permet d’éviter le spam si on rerend plusieurs fois dans la même seconde.
    """
    _init_toast_state()
    key = f"{level}|{dedup_key or text}"
    # anti-spam: si déjà toasté très récemment, on ignore
    if key in st.session_state["_toast_seen"]:
        return
    st.session_state["_toast_seen"].add(key)

    # st.toast accepte un "icon" (emoji) + du texte
    icon = _ICON.get(level, "ℹ️")
    st.toast(f"{icon} {text}", icon=None)  # icon dans le texte pour garder les couleurs Streamlit




from typing import Any, Callable

from typing import Any, Callable


from typing import Any, Callable



from typing import Any, Callable


from typing import Any, Callable

from typing import Any, Callable

def run_simulation_with_live_imputation(
    sim_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    positions,
    model: "torch.nn.Module|None",
    scaler: callable,
    inv_scaler: callable,
    device: "torch.device|None",
    graph_placeholder,              # unused, kept for signature
    sliding_chart_placeholder,      # unused, kept for signature
    gauge_placeholder,              # unused, kept for signature
    window_hours: int = 24,
    intro_delay_seconds: float = 1.0,   # 1er tick sans alertes, puis on arme
):
    """
    • PLOTS: identiques à ta version (draw_full_time_series_with_mask_gap + add_imputed_segments).
    • NOTIFS: overlay style “Facebook” en HAUT-GAUCHE.
    • MAP: capteur 'Delayed' en JAUNE (< Δtᵢ), puis ROUGE (≥ Δtᵢ). 'Predicted' = ROUGE.
    • Δtᵢ = st.session_state['sigma_threshold'] (minutes) → heures.
    • Scénarios:
        - S1 (delay < Δtᵢ): on attend → notif info “Awaiting late data”, map JAUNE.
        - S2 (delay ≥ Δtᵢ + voisins dispo): imputation → contraintes (temporal/spatial) → notif success/erreur.
        - S3 (delay ≥ Δtᵢ + pas de voisins/estim. fiable): notif erreur “System/Sensor fault?”.
        - Restauration: notif success “Data restored”.
    """
    import time, uuid, math
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import pydeck as pdk
    import streamlit as st
    from streamlit.components.v1 import html as st_html


    SS = st.session_state

    ICON_SPEC = SS.setdefault("_icon_spec", {
        "url": "",
        "width": 1,
        "height": 1,
        "anchorX": 0,
        "anchorY": 0,
    })

    # -------------------- Defaults & helpers --------------------
    DEFAULT_VALUES = globals().get("DEFAULT_VALUES", {
        "gauge_green_min": 0, "gauge_green_max": 20,
        "gauge_yellow_min": 20, "gauge_yellow_max": 50,
        "gauge_red_min": 50, "gauge_red_max": 100,
        "graph_size": 20, "sigma_threshold": 120,
    })

    def init_once(key, val):
        if key not in SS: SS[key] = val
        return SS[key]

    def zpad6(s: str) -> str: return s if not str(s).isdigit() else str(s).zfill(6)
    def strip0(s: str) -> str:
        t = str(s).lstrip("0"); return t if t else "0"

    def positions_to_df_safe(pos):
        try: return positions_to_df(pos)  # ta version si dispo
        except Exception: pass
        if isinstance(pos, pd.DataFrame):
            cols = {c.lower(): c for c in pos.columns}
            lat = cols.get("latitude") or cols.get("lat")
            lon = cols.get("longitude") or cols.get("lon") or cols.get("lng")
            sid = cols.get("sensor_id") or cols.get("id") or cols.get("name")
            if not (sid and lat and lon): raise ValueError("positions DF must have sensor_id/latitude/longitude")
            out = pos.rename(columns={sid:"sensor_id", lat:"latitude", lon:"longitude"}).copy()
            return out[["sensor_id","latitude","longitude"]]
        raise ValueError("Unsupported positions type")

    def ensure_datetime_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "datetime" in df.columns: return df
        if isinstance(df.index, pd.DatetimeIndex):
            return df.reset_index().rename(columns={"index":"datetime"})
        for alt in ("timestamp","date","time"):
            if alt in df.columns: return df.rename(columns={alt:"datetime"})
        idx_as_dt = pd.to_datetime(df.index, errors="coerce")
        if idx_as_dt.notna().all():
            out = df.reset_index().rename(columns={"index":"datetime"})
            out["datetime"] = idx_as_dt
            return out
        raise KeyError(f"{name} has no 'datetime' column or datetime-like index.")

    def lightify_safe(fig):
        try: lightify(fig)  # ta skin
        except Exception: pass

    # -------------------- Notifications “Facebook” (haut-gauche) --------------------
    SS.setdefault("_notif_items", [])
    init_once("alerts_armed", False)

    def _render_left_notifs_component():
        import time as _t
        now = _t.time()
        SS["_notif_items"] = [n for n in SS["_notif_items"] if n.get("until", now) > now]
        icons = {"info":"ℹ️","success":"✅","warning":"⚠️","error":"⛔"}
        items = []
        for n in SS["_notif_items"][-10:]:
            items.append(f"""
            <div class="tsg-left-notif {n['level']}">
              <div class="tsg-icon">{icons.get(n['level'],'🔹')}</div>
              <div class="tsg-body"><div>{n['text']}</div><div class="tsg-ts">{n['ts']}</div></div>
            </div>""")
        css = """
        <style>
          .tsg-root { position: fixed; top: 12px; left: 12px; width: 320px; z-index: 10000; pointer-events: none; }
          .tsg-left-notif { background: rgba(33,33,33,.92); color:#fff; border-radius:10px; padding:10px 12px; margin:6px 0;
                            box-shadow:0 6px 18px rgba(0,0,0,.25); display:flex; gap:8px; align-items:flex-start; }
          .tsg-left-notif.info    { border-left:4px solid #3498db; }
          .tsg-left-notif.success { border-left:4px solid #2ecc71; }
          .tsg-left-notif.warning { border-left:4px solid #f1c40f; }
          .tsg-left-notif.error   { border-left:4px solid #e74c3c; }
          .tsg-icon { width:18px; }
          .tsg-body { flex:1; }
          .tsg-ts { font-size:11px; opacity:.8; margin-top:2px; }
          html,body{ background:transparent; margin:0; padding:0; }
        </style>"""
        html = f"<!doctype html><html><head>{css}</head><body><div class='tsg-root'>{''.join(items)}</div></body></html>"
        st_html(html, height=1, scrolling=False)

    class NotificationCenter:
        def __init__(self): self.seen = SS.setdefault("_notifs_seen", set())
        def push(self, level, title, text, ts, ttl=6.0, key=None):
            k = key or f"{level}:{title}:{text}:{str(ts)[:13]}"
            if k in self.seen: return
            self.seen.add(k)
            import time as _t
            SS["_notif_items"].append({"level":level,"text":f"<b>{title}</b> — {text}",
                                       "ts": str(ts), "until": _t.time()+float(ttl)})
            _render_left_notifs_component()

    NOTIF = NotificationCenter()
    def n_info(ti, tx, ts, key=None): NOTIF.push("info","Awaiting late data" if "Awaiting" in ti else ti, tx, ts, key=key)
    def n_ok(ti, tx, ts, key=None):   NOTIF.push("success", ti, tx, ts, key=key)
    def n_warn(ti, tx, ts, key=None): NOTIF.push("warning", ti, tx, ts, key=key)
    def n_err(ti, tx, ts, key=None):  NOTIF.push("error", ti, tx, ts, key=key)

    # -------------------- Contraintes (spatial/temporal) --------------------
    constraints_list = SS.get("constraints", []) or []
    spatial_cfg = None
    temporal_cfgs = []
    for c in constraints_list:
        if c.get("type") == "Spatial": spatial_cfg = c
        elif c.get("type") == "Temporal": temporal_cfgs.append(c)

    MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    def month_name(ts: pd.Timestamp) -> str:
        try: return MONTHS[int(ts.month)-1]
        except: return ""

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        dφ, dλ = math.radians(lat2-lat1), math.radians(lon2-lon1)
        a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return 2*R*math.asin(math.sqrt(a))

    def temporal_ok(ts: pd.Timestamp, val) -> (bool, str|None):
        if val is None or (isinstance(val,float) and np.isnan(val)): return False, "no value"
        mname = month_name(ts)
        rules = [c for c in temporal_cfgs if c.get("month")==mname]
        if not rules: return True, None
        v = float(val)
        for r in rules:
            opt = (r.get("constraint_option") or r.get("option") or "").lower()
            thr = float(r.get("temp_threshold", np.nan))
            if np.isnan(thr): continue
            if "greater" in opt and not (v > thr): return False, f"{mname}: {v:.2f} !> {thr}"
            if "less"    in opt and not (v < thr): return False, f"{mname}: {v:.2f} !< {thr}"
        return True, None

    def spatial_ok(sensor_id: str, ts: pd.Timestamp, val, latlng_df: pd.DataFrame, values_row: dict) -> (bool, list[str]):
        if spatial_cfg is None or val is None or (isinstance(val,float) and np.isnan(val)): return True, []
        max_km  = float(spatial_cfg.get("distance in km", 0)) or 0.0
        max_diff = float(spatial_cfg.get("diff", 0)) if spatial_cfg.get("diff") is not None else 0.0
        if max_km <= 0 or max_diff <= 0: return True, []
        row = latlng_df.loc[latlng_df["data_col"] == sensor_id]
        if row.empty: return True, []
        lat_s, lon_s = float(row.iloc[0]["latitude"]), float(row.iloc[0]["longitude"])
        bad = []
        for _, nb in latlng_df.iterrows():
            nb_id = nb["data_col"]
            if nb_id == sensor_id: continue
            dkm = haversine_km(lat_s, lon_s, float(nb["latitude"]), float(nb["longitude"]))
            if dkm <= max_km + 1e-9:
                nb_val = values_row.get(nb_id, np.nan)
                if nb_val is None or (isinstance(nb_val,float) and np.isnan(nb_val)): continue
                if abs(float(val)-float(nb_val)) > max_diff:
                    bad.append(f"{nb_id} (Δ={abs(float(val)-float(nb_val)):.2f} > {max_diff}, {dkm:.2f} km)")
        return (len(bad)==0), bad

    # -------------------- Sélection / positions --------------------
    all_sensor_cols = [c for c in sim_df.columns if c != "datetime"]
    graph_size = int(SS.get("graph_size", DEFAULT_VALUES["graph_size"]))
    sensor_cols = [str(c) for c in all_sensor_cols[:graph_size]]
    col_to_idx = {c: i for i, c in enumerate(sensor_cols)}

    latlng_raw = positions_to_df_safe(positions).copy()
    latlng_raw["sensor_id"] = latlng_raw["sensor_id"].astype(str).str.strip()
    latlng_raw["latitude"]  = pd.to_numeric(latlng_raw["latitude"],  errors="coerce")
    latlng_raw["longitude"] = pd.to_numeric(latlng_raw["longitude"], errors="coerce")
    latlng_raw = latlng_raw.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    pos_ids = latlng_raw["sensor_id"].tolist()

    map_exact  = {pid: pid for pid in pos_ids if pid in sensor_cols}
    map_pad6   = {pid: zpad6(pid) for pid in pos_ids if zpad6(pid) in sensor_cols}
    map_strip0 = {}
    for pid in pos_ids:
        s = strip0(pid); s6 = zpad6(s)
        if s in sensor_cols: map_strip0[pid] = s
        elif s6 in sensor_cols: map_strip0[pid] = s6
    map_index = {}
    if all(str(p).isdigit() for p in pos_ids):
        nums = sorted(int(p) for p in pos_ids)
        if nums and nums[0]==0 and nums[-1]==len(nums)-1:
            for i, pid in enumerate(sorted(pos_ids, key=lambda x: int(x))):
                if i < len(sensor_cols): map_index[pid] = sensor_cols[i]

    best_map = max([map_exact, map_pad6, map_strip0, map_index], key=lambda m: len(m))
    if len(best_map) == 0: st.info("No matching positions for selected sensors."); return

    latlng = latlng_raw.copy()
    latlng["data_col"] = latlng["sensor_id"].map(best_map)
    latlng = latlng[latlng["data_col"].notna()].copy()
    order_index = {c: i for i, c in enumerate(sensor_cols)}
    latlng["__ord"] = latlng["data_col"].map(order_index)
    latlng = latlng.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    sensor_cols = [c for c in sensor_cols if c in set(latlng["data_col"])]
    if not sensor_cols: st.info("After mapping, no sensors remain to plot."); return
    col_to_idx = {c: i for i, c in enumerate(sensor_cols)}

    # -------------------- Temps / alignement --------------------
    sim_df     = ensure_datetime_column(sim_df, "sim_df")
    missing_df = ensure_datetime_column(missing_df, "missing_df")
    sim_df["datetime"]     = pd.to_datetime(sim_df["datetime"], errors="coerce").dt.floor("h")
    missing_df["datetime"] = pd.to_datetime(missing_df["datetime"], errors="coerce").dt.floor("h")
    sim_df     = sim_df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"])
    missing_df = missing_df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"])
    sim_df.set_index("datetime", inplace=True)
    missing_df.set_index("datetime", inplace=True)

    if "orig_missing_baseline" not in SS:
        SS.orig_missing_baseline = missing_df.isna().copy()
    else:
        if (not SS.orig_missing_baseline.index.equals(missing_df.index) or
            list(SS.orig_missing_baseline.columns) != list(missing_df.columns)):
            SS.orig_missing_baseline = missing_df.isna().copy()
    if "orig_missing_values" not in SS:
        SS.orig_missing_values = missing_df.copy()

    common_index = missing_df.index
    if common_index.empty or latlng.empty: st.info("No matching timeline or positions/sensors."); return

    # -------------------- États persistants --------------------
    uid = init_once("sim_uid", f"sim_{uuid.uuid4().hex[:8]}")
    init_once("sim_iter", 0)
    init_once("sim_ptr", 0)
    init_once("sliding_window_df", pd.DataFrame(columns=["datetime"] + list(sensor_cols)))
    init_once("global_df", pd.DataFrame(columns=["datetime"] + list(sensor_cols)))
    if "_missing_streak_hours" not in SS:
        SS["_missing_streak_hours"] = {c: 0.0 for c in sensor_cols}
    else:
        for c in sensor_cols: SS["_missing_streak_hours"].setdefault(c, 0.0)
    if "_prev_baseline_missing" not in SS:
        SS["_prev_baseline_missing"] = {c: None for c in sensor_cols}

    # -------------------- UI (identique) --------------------
    if "_ui_inited" not in SS:
        SS["_ui_inited"] = True
        col_left, col_right = st.columns([3, 1], gap="small")
        with col_left:
            st.markdown("### Sensor Visualization")
            hdr_l, hdr_r = st.columns([5, 2], gap="small")
            with hdr_l: SS["ph_time"] = st.markdown("**Current Time:** —")
            with hdr_r: SS["ph_fitbtn"] = st.empty()
            SS["ph_map"] = st.empty()
        with col_right:
            st.markdown("<div style='text-align:center;font-weight:700;margin:0.25rem 0'>Data Quality & Activity</div>", unsafe_allow_html=True)
            SS["ph_counts_active"]  = st.empty()
            SS["ph_counts_missing"] = st.empty()
            SS["ph_gauge"] = st.empty()
        st.markdown("---")
        row2_l, row2_r = st.columns([3, 2], gap="small")
        with row2_l: SS["ph_global"] = st.empty()
        with row2_r: SS["ph_snap10"] = st.empty()
        st.markdown("---")

    with SS["ph_fitbtn"]:
        if st.button("Fit map to sensors", use_container_width=True, key=f"{uid}_fitbtn"):
            SS["_fit_event"] = True

    # -------------------- deck.gl --------------------
    GREEN  = [46,204,113,200]; AMBER = [241,196,15,200]; RED = [231,76,60,200]
    # global ICON_SPEC
    # if "ICON_SPEC" not in globals() or ICON_SPEC is None:
    #     ICON_SPEC = {"url": "", "width": 1, "height": 1, "anchorX": 0, "anchorY": 0}
    def make_bg_layer(df):
        return pdk.Layer("ScatterplotLayer", data=df,
                         get_position=["longitude","latitude"],
                         get_fill_color="bg_color", get_radius="bg_radius",
                         radius_scale=1, radius_min_pixels=6, radius_max_pixels=22,
                         stroked=True, get_line_color=[255,255,255,180], line_width_min_pixels=1,
                         pickable=False)
    def make_icon_layer(df):
        return pdk.Layer("IconLayer", data=df, get_icon="icon",
                         get_position=["longitude","latitude"], get_size="icon_size",
                         size_scale=8, size_min_pixels=14, size_max_pixels=28, pickable=True)
    def fit_view_simple(df: pd.DataFrame, padding_deg=0.02) -> pdk.ViewState:
        if df is None or df.empty: return pdk.ViewState(latitude=0, longitude=0, zoom=2, bearing=0, pitch=0)
        lat_min = float(df["latitude"].min());  lat_max = float(df["latitude"].max())
        lon_min = float(df["longitude"].min()); lon_max = float(df["longitude"].max())
        lat_c = (lat_min+lat_max)/2.0; lon_c = (lon_min+lon_max)/2.0
        span = max(lat_max-lat_min, lon_max-lon_min) + padding_deg; span = max(span, 1e-3)
        zoom = max(1.0, min(16.0, np.log2(360.0/span)))
        return pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom, bearing=0, pitch=0)

    if "deck_obj" not in SS:
        base_df = latlng.copy()
        base_df["sensor_id"] = base_df["sensor_id"].astype(str)
        base_df["value"] = "NA"
        base_df["status"] = "Predicted"
        base_df["bg_color"] = [[231, 76, 60, 200] for _ in range(len(base_df))]
        base_df["bg_radius"] = 10
        base_df["icon"] = [ICON_SPEC] * len(base_df)  # ← utilise l'instance stateful
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

    if SS.get("_fit_event", False):
        df_to_fit = SS.get("_fit_base_df", latlng)
        new_view = fit_view_simple(df_to_fit)
        SS.deck_obj = pdk.Deck(layers=SS.deck_obj.layers, initial_view_state=new_view,
                               map_style=getattr(SS.deck_obj,"map_style","mapbox://styles/mapbox/light-v11"),
                               tooltip=SS.get("deck_tooltip", {"text":"Sensor {sensor_id}\nValue: {value}\nStatus: {status}"}))
        SS["_fit_event"] = False
    SS["ph_map"].pydeck_chart(SS.deck_obj, use_container_width=True)
    _render_left_notifs_component()

    # -------------------- Palette PLOTS --------------------
    base_palette = ["#000000","#003366","#009999","#006600","#66CC66","#FF9933","#FFD700","#708090","#4682B4","#99FF33"]
    sensor_color_map = {c: base_palette[i % len(base_palette)] for i, c in enumerate(sensor_cols)}

    # -------------------- Modèle --------------------
    use_model = model is not None
    if use_model:
        try: model.eval()
        except Exception: pass

    # -------------------- Δtᵢ --------------------
    sigma_minutes = SS.get("sigma_threshold", DEFAULT_VALUES["sigma_threshold"])
    DELTA_HOURS = float(sigma_minutes) / 60.0

    # -------------------- Mask builder (pour PLOTS) --------------------
    def build_imputed_mask_from_store(missing_df: pd.DataFrame,
                                      baseline_missing: pd.DataFrame,
                                      sensor_cols: list[str]) -> pd.DataFrame:
        base = baseline_missing.reindex(index=missing_df.index, columns=sensor_cols, fill_value=False).astype(bool)
        now_has_val = missing_df.reindex(columns=sensor_cols).notna()
        return (base & now_has_val).astype(bool)

    # -------------------- Fallback prédicteur --------------------
    def predict_single_missing_value_fallback(historical_window, target_sensor_index, **kwargs):
        try:
            col = int(target_sensor_index)
            col_series = pd.Series(historical_window[:, col])
            col_series = pd.to_numeric(col_series, errors="coerce").dropna()
            return float(col_series.iloc[-1]) if not col_series.empty else np.nan
        except Exception:
            return np.nan

    # -------------------- LOOP --------------------
    SNAP10 = 10
    ptr = int(SS["sim_ptr"])
    total_steps = len(common_index)

    while ptr < total_steps:
        ts = pd.Timestamp(common_index[ptr])
        SS["sim_iter"] = SS.get("sim_iter", 0) + 1
        iter_key = SS["sim_iter"]
        SS["ph_time"].markdown(f"<div style='font-weight:600'>Current Time: {ts}</div>", unsafe_allow_html=True)

        baseline_row_ts = SS.orig_missing_baseline.reindex(index=[ts], columns=sensor_cols).iloc[0]

        # Fenêtre historique < ts
        hist_end = ts - pd.Timedelta(hours=1)
        if hist_end in missing_df.index:
            hist_idx = missing_df.loc[:hist_end].index[-window_hours:]
        else:
            hist_idx = missing_df.index[missing_df.index < ts][-window_hours:]
        hist_win = missing_df.loc[hist_idx, sensor_cols] if len(hist_idx) > 0 else pd.DataFrame()

        # --- Imputation avec attente Δtᵢ ---
        svals, status = [], []  # status: "Real" | "Delayed" | "Predicted"
        for col in sensor_cols:
            originally_missing = bool(baseline_row_ts.get(col, False))
            val_now = np.nan

            if originally_missing:
                # streak / délai
                SS["_missing_streak_hours"][col] = SS["_missing_streak_hours"].get(col, 0.0) + 1.0
                delay_h = SS["_missing_streak_hours"][col]

                if delay_h < DELTA_HOURS:
                    # S1: attendre (ne pas imputer)
                    status.append("Delayed")
                    svals.append(np.nan)
                    if SS.get("alerts_armed", False):
                        n_info("Awaiting late data", f"Sensor {col}: missing for {int(delay_h)}h (Δt={DELTA_HOURS:.1f}h).", ts,
                              key=f"s1:{col}:{int(delay_h)}")
                else:
                    # S2/S3: tenter d’imputer
                    pred_val = np.nan
                    if use_model and not hist_win.empty:
                        try:
                            pred_val = predict_single_missing_value(   # ta fonction si dispo
                                historical_window=np.asarray(hist_win.values, dtype=np.float32),
                                target_sensor_index=col_to_idx[col],
                                model=model, scaler=scaler, inv_scaler=inv_scaler, device=device
                            )
                        except Exception:
                            pred_val = np.nan
                    if (not np.isfinite(pred_val)) or pd.isna(pred_val):
                        pred_val = predict_single_missing_value_fallback(
                            historical_window=np.asarray(hist_win.values, dtype=np.float32) if not hist_win.empty else np.empty((0, len(sensor_cols))),
                            target_sensor_index=col_to_idx[col]
                        )

                    if np.isfinite(pred_val) and not pd.isna(pred_val):
                        try: missing_df.at[ts, col] = float(pred_val)
                        except Exception: pass
                        status.append("Predicted")
                        svals.append(float(pred_val))

                        # Contraintes
                        ok_t, msg_t = temporal_ok(ts, pred_val)
                        # Prépare la ligne des voisins au même ts
                        row_vals = {c: (missing_df.at[ts, c] if (ts in missing_df.index and c in missing_df.columns) else np.nan) for c in sensor_cols}
                        row_vals[col] = pred_val
                        ok_s, bad_neighbors = spatial_ok(col, ts, pred_val, latlng, row_vals)

                        if SS.get("alerts_armed", False):
                            if (not ok_t) and msg_t:
                                n_err("Real-time alert", f"Sensor {col}: temporal constraint failed — {msg_t}.", ts)
                            elif (not ok_s) and bad_neighbors:
                                n_err("Real-time alert", f"Sensor {col}: spatial inconsistency → {', '.join(bad_neighbors[:5])}{'…' if len(bad_neighbors)>5 else ''}.", ts)
                            else:
                                n_ok("Imputation completed", f"Sensor {col}: reconstructed; plausibility OK.", ts)
                    else:
                        # S3: aucune estimation fiable
                        status.append("Delayed")
                        svals.append(np.nan)
                        if SS.get("alerts_armed", False):
                            n_err("System/Sensor fault?", f"Sensor {col}: no reliable estimate from historical patterns.", ts)
            else:
                # Donnée réelle présente à l'origine
                SS["_missing_streak_hours"][col] = 0.0
                try:
                    val_now = missing_df.at[ts, col]
                except Exception:
                    val_now = np.nan
                svals.append(val_now)
                status.append("Real")

                # Restauration si précédemment “missing”
                prev = SS["_prev_baseline_missing"].get(col, None)
                if SS.get("alerts_armed", False) and prev is True:
                    n_ok("Data restored", f"Sensor {col}: telemetry available at {ts}.", ts)

            SS["_prev_baseline_missing"][col] = originally_missing

        # --- Buffers rendu (sliding & global) ---
        row = {"datetime": ts}
        for i, c in enumerate(sensor_cols): row[c] = svals[i]
        SS.sliding_window_df.loc[len(SS.sliding_window_df)] = row
        SS.global_df.loc[len(SS.global_df)] = row
        if len(SS.sliding_window_df) > 36: SS.sliding_window_df = SS.sliding_window_df.tail(36)

        # --- Carte (JAUNE < Δtᵢ, ROUGE ≥ Δtᵢ ou Predicted) ---
        tick_df = latlng.copy()
        vals_by_col = dict(zip(sensor_cols, svals))
        status_by_col = dict(zip(sensor_cols, status))
        def color_for(col):
            stt = status_by_col.get(col, "Real")
            if stt == "Real": return [46,204,113,200]           # vert
            if stt == "Predicted": return [231,76,60,200]       # rouge
            # Delayed: jaune si < Δtᵢ, rouge si >= Δtᵢ
            delay_h = SS["_missing_streak_hours"].get(col, 0.0)
            return [241,196,15,200] if delay_h < DELTA_HOURS else [231,76,60,200]

        tick_df["value"] = tick_df["data_col"].map(vals_by_col).fillna("NA")
        tick_df["status"] = tick_df["data_col"].map(lambda c: status_by_col.get(c, "Real"))
        tick_df["bg_color"] = tick_df["data_col"].map(color_for)
        tick_df["bg_radius"] = 10

        tick_df["icon"] = [ICON_SPEC] * len(tick_df)  # ← pas de global
        tick_df["icon_size"] = 1.0

        # fit éventuel
        if SS.get("_fit_event", False):
            new_view = fit_view_simple(tick_df)
            SS.deck_obj = pdk.Deck(layers=[make_bg_layer(tick_df), make_icon_layer(tick_df)],
                                   initial_view_state=new_view,
                                   map_style="mapbox://styles/mapbox/light-v11",
                                   tooltip=SS.get("deck_tooltip", {"text":"Sensor {sensor_id}\nValue: {value}\nStatus: {status}"}))
            SS["_fit_event"] = False
        else:
            SS.deck_obj.layers = [make_bg_layer(tick_df), make_icon_layer(tick_df)]
        SS["_fit_base_df"] = tick_df.copy()
        SS["ph_map"].pydeck_chart(SS.deck_obj, use_container_width=True)
        _render_left_notifs_component()  # garde l’overlay vivant

        # --- Gauge & compteurs ---
        baseline_mask_to_now = SS.orig_missing_baseline.loc[:ts, sensor_cols]
        cumulative_missed = int(baseline_mask_to_now.values.sum())
        total_cells_to_now = baseline_mask_to_now.size
        pct_missed_to_now = (cumulative_missed / total_cells_to_now * 100.0) if total_cells_to_now else 0.0
        missed_now = int(baseline_row_ts.sum())
        active_now = len(sensor_cols) - missed_now
        SS["ph_counts_active"].markdown(f"Active sensors now: **{active_now}**")
        SS["ph_counts_missing"].markdown(f"Delayed sensors now: **{missed_now}**")

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pct_missed_to_now, title={"text":"Missed Data (%)"},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"red" if pct_missed_to_now >= DEFAULT_VALUES["gauge_red_max"] else "green"},
                   "steps":[
                       {"range":[DEFAULT_VALUES["gauge_green_min"],DEFAULT_VALUES["gauge_green_max"]], "color":"lightgreen"},
                       {"range":[DEFAULT_VALUES["gauge_yellow_min"],DEFAULT_VALUES["gauge_yellow_max"]], "color":"yellow"},
                       {"range":[DEFAULT_VALUES["gauge_red_min"],DEFAULT_VALUES["gauge_red_max"]], "color":"red"}]}))
        gauge_fig.update_layout(title="", margin=dict(l=10,r=10,t=30,b=10))
        lightify_safe(gauge_fig)
        SS["ph_gauge"].plotly_chart(gauge_fig, use_container_width=True, key=f"{uid}_gauge_{iter_key}")

        # --- PLOTS (même logique que chez toi) ---
        # masque reconstruit depuis la mémoire → segments rouges cohérents
        mask_plot = build_imputed_mask_from_store(missing_df, SS.orig_missing_baseline, sensor_cols)

        # Global TS
        # (reindex pour coller à l’ordre temporel de global_df)
        mask_for_global = mask_plot.reindex(index=pd.to_datetime(SS.global_df["datetime"]), columns=sensor_cols, fill_value=False)
        full_ts_fig = draw_full_time_series_with_mask_gap(SS.global_df.copy(), mask_for_global, sensor_cols, sensor_color_map)
        lightify_safe(full_ts_fig)
        SS["ph_global"].plotly_chart(full_ts_fig, use_container_width=True, key=f"{uid}_global_{iter_key}")

        # Snapshot 10
        snap10 = SS.sliding_window_df.tail(SNAP10)
        snap10_fig = go.Figure()
        for col in sensor_cols:
            base_color = sensor_color_map[col]
            sub = (snap10[["datetime", col]].dropna().sort_values("datetime").rename(columns={col:"value"}))
            if sub.empty: continue
            def is_imp(t, c=col):
                try: return bool(mask_plot.loc[pd.Timestamp(t), c])
                except: return False
            add_imputed_segments(snap10_fig, sub, is_imp, base_color, gap_hours=6)
        snap10_fig.update_layout(title="Snapshot (last 10)",
                                 xaxis_title="Time", yaxis_title="Value",
                                 margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        lightify_safe(snap10_fig)
        SS["ph_snap10"].plotly_chart(snap10_fig, use_container_width=True, key=f"{uid}_snap10_{iter_key}")

        # --- avance + armement alertes ---
        ptr += 1
        SS["sim_ptr"] = ptr
        if not SS.get("alerts_armed", False):
            time.sleep(float(intro_delay_seconds))
            SS["alerts_armed"] = True
        else:
            time.sleep(1.0)


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
    import os, time, hashlib
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    # ---------------- helpers ----------------
    try:
        from utils.config import DEFAULT_VALUES
    except Exception:
        DEFAULT_VALUES = {"graph_size": 36}

    def _last_observed(hist_df: pd.DataFrame, col: str) -> float:
        if hist_df.empty or col not in hist_df.columns: return np.nan
        s = pd.to_numeric(hist_df[col].dropna(), errors="coerce")
        return float(s.iloc[-1]) if len(s) else np.nan

    def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """Conserver date+heure exactes (aucun arrondi/‘floor’)."""
        if "datetime" in df.columns:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"]).set_index("datetime")
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()]
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df[~df.index.duplicated(keep="first")].sort_index()

    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [(str(c).strip().zfill(6) if str(c).isdigit() else str(c).strip()) for c in out.columns]
        return out

    def lightify(fig: go.Figure) -> None:
        fig.update_layout(template="plotly_white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                          margin=dict(l=10, r=10, t=28, b=10))
        fig.update_xaxes(showgrid=True, gridwidth=1, griddash="dot", type="date", tickformat="%Y-%m-%d %H:%M")
        fig.update_yaxes(showgrid=True, gridwidth=1, griddash="dot")

    def add_segmented_curve(fig, xs, ys, imp_flags, base_color: str, gap_hours: float = 6.0):
        """xs: list[datetime], ys: array, imp_flags: array[bool] → rouge=True pour segments"""
        if len(xs) < 2:
            return
        seg_x = [xs[0]]; seg_y = [ys[0]]; seg_imp = bool(imp_flags[0])
        for i in range(1, len(xs)):
            gap = (xs[i] - xs[i-1]).total_seconds() / 3600.0
            flipped = (bool(imp_flags[i]) != seg_imp)
            if (gap > gap_hours) or flipped:
                if len(seg_x) >= 2:
                    fig.add_trace(dict(
                        type="scatter", mode="lines",
                        x=list(seg_x), y=list(seg_y),
                        line=dict(width=2, color=("red" if seg_imp else base_color)),
                        showlegend=False
                    ))
                seg_x, seg_y = [xs[i-1], xs[i]], [ys[i-1], ys[i]]
                seg_imp = bool(imp_flags[i])
            else:
                seg_x.append(xs[i]); seg_y.append(ys[i])
        if len(seg_x) >= 2:
            fig.add_trace(dict(
                type="scatter", mode="lines",
                x=list(seg_x), y=list(seg_y),
                line=dict(width=2, color=("red" if seg_imp else base_color)),
                showlegend=False
            ))

    # signature **stable** de la timeline (indépendante des valeurs / régularisation)
    def timeline_signature(idx: pd.DatetimeIndex, cols: list) -> str:
        h = hashlib.md5()
        # index -> int64 ns (asi8), ordre conservé
        try:
            h.update(np.asarray(idx.asi8).tobytes())
        except Exception:
            h.update("|".join(map(str, idx.astype("datetime64[ns]"))).encode())
        h.update(("|".join(map(str, cols))).encode())
        return h.hexdigest()

    # ---------------- normalisation ----------------
    st.markdown("## TSGuard vs PriSTI — Comparison")
    SS = st.session_state

    sim_df     = _norm_cols(ensure_datetime(sim_df))
    missing_df = _norm_cols(ensure_datetime(missing_df))
    if missing_df.index.empty:
        st.info("No timeline to compare."); return

    # ---------------- PriSTI presence ----------------
    PRISTI_ROOT = "./PriSTI"
    CONFIG_PATH = f"{PRISTI_ROOT}/config/base.yaml"
    WEIGHTS_PATH = f"{PRISTI_ROOT}/save/aqi36/model.pth"
    MEANSTD_PK   = f"{PRISTI_ROOT}/data/pm25/pm25_meanstd.pk"
    pristi_ready = (
        len(missing_df.columns) >= 36
        and os.path.exists(CONFIG_PATH)
        and os.path.exists(WEIGHTS_PATH)
        and os.path.exists(MEANSTD_PK)
    )
    if not pristi_ready:
        st.error("PriSTI backend not available (need config/weights/meanstd and ≥36 sensors).")
        return

    def ensure_pristi_artifacts():
        need_reload = ("pristi_model" not in SS) or (SS.get("pristi_device") != str(device))
        if need_reload:
            pm, pmu, psd = load_pristi_artifacts(CONFIG_PATH, WEIGHTS_PATH, MEANSTD_PK, device=device)
            SS["pristi_model"]  = pm
            SS["pristi_mean"]   = pmu
            SS["pristi_std"]    = psd
            SS["pristi_device"] = str(device)
    ensure_pristi_artifacts()

    # ---------------- bases ----------------
    base_df           = missing_df.copy()      # valeurs originales (NaN aux trous)
    orig_missing_mask = base_df.isna()
    sim_df            = sim_df.reindex(base_df.index)

    all_sensor_cols = list(sim_df.columns)
    graph_size      = int(SS.get("graph_size", DEFAULT_VALUES["graph_size"]))
    sensor_cols     = [str(c) for c in all_sensor_cols[:graph_size]]    # TSGuard agit ici
    pristi_cols     = list(base_df.columns)[:36]                         # PriSTI: 36 capteurs

    # couleurs
    base_palette = ["#000000","#003366","#009999","#006600","#66CC66","#FF9933","#FFD700","#708090",
                    "#4682B4","#99FF33","#1F77B4","#5DA5DA","#1E90FF","#00BFFF","#00CED1","#17BECF",
                    "#40E0D0","#20B2AA","#16A085","#1ABC9C","#2ECC71","#3CB371","#2CA02C","#00FA9A",
                    "#7FFFD4","#ADFF2F","#F1C40F","#F4D03F","#B7950B","#4B0082","#6A5ACD","#7B68EE",
                    "#483D8B","#3F51B5","#2E4057","#A9A9A9"]
    color_map = {c: base_palette[i % len(base_palette)] for i, c in enumerate(pristi_cols)}

    # ---------------- store PERSISTANT par timeline ----------------
    TL_SIG = timeline_signature(base_df.index, pristi_cols)
    SS.setdefault("impute_store", {})
    store = SS["impute_store"].get(TL_SIG)
    if store is None:
        store = {
            "tsg_imp": pd.DataFrame(index=base_df.index, columns=pristi_cols, dtype=float),
            "pri_imp": pd.DataFrame(index=base_df.index, columns=pristi_cols, dtype=float),
            # ensembles de timestamps imputés (par capteur) conservés en pd.Timestamp
            "tsg_idx": {c: set() for c in pristi_cols},
            "pri_idx": {c: set() for c in pristi_cols},
            # état d'avancement
            "t_times": pd.Series(np.nan, index=base_df.index, dtype=float),
            "p_times": pd.Series(np.nan, index=base_df.index, dtype=float),
            "i_done": -1,
            "first_window_shown": False,
            "sum_lastN_pri_ms": 0.0,
            "sum_lastN_tsg_ms": 0.0,
        }
        SS["impute_store"][TL_SIG] = store

    # ---------------- affichage ----------------
    def build_display(ts_index, which: str):
        base_block = base_df.reindex(ts_index)[pristi_cols]
        if which == "tsg":
            imp_vals = store["tsg_imp"].reindex(ts_index)[pristi_cols]
            idx_sets = store["tsg_idx"]
        else:
            imp_vals = store["pri_imp"].reindex(ts_index)[pristi_cols]
            idx_sets = store["pri_idx"]
        display = base_block.copy()
        have_imp = imp_vals.notna()
        display[have_imp] = imp_vals[have_imp]
        return display, idx_sets

    def plot_block(display_df, idx_sets, cols_show, title):
        import numpy as np
        import pandas as pd
        fig = go.Figure()

        for col in cols_show:
            if col not in display_df.columns: continue
            base_color = color_map.get(col, "#444")
            s = display_df[col]
            notna_mask = s.notna()
            if not notna_mask.any(): continue

            x_idx  = display_df.index[notna_mask]               # DatetimeIndex exact
            y_vals = s.loc[x_idx].to_numpy(dtype=float)

            # construire l'index des timestamps imputés (robuste)

            # FLAGS ROUGES: Index.isin -> ndarray (PAS de .to_numpy ici)
            # x_idx : DatetimeIndex exact de la série tracée
            imp_times = list(idx_sets.get(col, set()))
            imp_idx = pd.DatetimeIndex(pd.to_datetime(imp_times, errors="coerce")).dropna()

            # Membership robuste : compare en nanosecondes (évite tz/dtype)
            imp_flags = np.isin(x_idx.asi8, imp_idx.asi8)

            x_vals = x_idx.to_pydatetime().tolist()
            add_segmented_curve(fig, x_vals, y_vals, imp_flags, base_color, gap_hours=6.0)

            red_pos = np.where(imp_flags)[0]
            if red_pos.size:
                fig.add_trace(go.Scatter(
                    x=[x_vals[j] for j in red_pos],
                    y=y_vals[red_pos],
                    mode="markers",
                    marker=dict(size=6, color="red"),
                    name="Imputed (points)",
                    showlegend=False
                ))

        # légende
        for col in cols_show:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                     marker=dict(size=8, color=color_map.get(col, "#444")),
                                     showlegend=True, name=f"{col}"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                 line=dict(width=2, color="red"),
                                 showlegend=True, name="Imputed segment"))

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value")
        lightify(fig)
        return fig

    # ---------------- UI sliders ----------------
    sliders = st.container()
    cache_key_short = TL_SIG[:10]  # compact
    def _ui_only_changed(): st.session_state["_ui_only_rerun"] = True

    first_window_ready = bool(store["first_window_shown"]) or (store["i_done"] >= (eval_len - 1))
    if first_window_ready:
        st.session_state.setdefault(f"cmp_sensors_{cache_key_short}", min(6, len(pristi_cols)))
        st.session_state.setdefault(f"cmp_steps_{cache_key_short}",   min(36, eval_len))
        with sliders:
            c1, c2 = st.columns(2)
            cmp_sensors = c1.slider("Sensors to display", 1, min(36, len(pristi_cols)),
                                    st.session_state[f"cmp_sensors_{cache_key_short}"],
                                    key=f"cmp_sensors_{cache_key_short}", on_change=_ui_only_changed)
            cmp_steps   = c2.slider("Timestamps to display (≤36, last)", 6, 36,
                                    st.session_state[f"cmp_steps_{cache_key_short}"],
                                    key=f"cmp_steps_{cache_key_short}", on_change=_ui_only_changed)
    else:
        with sliders:
            st.caption("⏳ Computing the first 36-step window… sliders will appear here once ready.")
        cmp_sensors = min(6, len(pristi_cols))
        cmp_steps   = min(36, eval_len)

    def render_from_cache(cmp_sensors_val, cmp_steps_val):
        # fenêtre qui SE TERMINE au dernier timestamp effectivement calculé
        end_i = store.get("i_done", -1)
        if end_i < 0:
            # rien de calculé → on affiche simplement le début
            start_i = 0
            end_i = min(len(base_df.index) - 1, cmp_steps_val - 1)
        else:
            end_i = int(min(end_i, len(base_df.index) - 1))
            start_i = max(0, end_i - cmp_steps_val + 1)

        ts_index = base_df.index[start_i:end_i + 1]

        disp_tsg, tsg_sets = build_display(ts_index, "tsg")
        disp_pri, pri_sets = build_display(ts_index, "pri")
        cols_show = pristi_cols[:cmp_sensors_val]

        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_block(disp_tsg, tsg_sets, cols_show, "TSGuard"),
                        use_container_width=True, key="plot_tsg")
        c2.plotly_chart(plot_block(disp_pri, pri_sets, cols_show, "PriSTI"),
                        use_container_width=True, key="plot_pri")

        st.caption(
            f"Red pts in window — TSGuard: "
            f"{sum(len(set(tsg_sets[c]) & set(ts_index)) for c in cols_show)} • "
            f"PriSTI: {sum(len(set(pri_sets[c]) & set(ts_index)) for c in cols_show)}"
        )
        st.markdown(
            f"**Imputation time — PriSTI: {store['sum_lastN_pri_ms']:.1f} ms • "
            f"TSGuard: {store['sum_lastN_tsg_ms']:.1f} ms**"
        )

    if st.session_state.get("_ui_only_rerun"):
        st.session_state["_ui_only_rerun"] = False
        render_from_cache(st.session_state[f"cmp_sensors_{cache_key_short}"],
                          st.session_state[f"cmp_steps_{cache_key_short}"])
        return

    # ---------------- compute / resume ----------------
    if not store["first_window_shown"]:
        status_box = st.status("Preparing first 36-step window…", expanded=True)
        prog = st.progress(0)
    else:
        status_box = None; prog = None

    live_l, live_r = st.columns(2)
    index_all = list(base_df.index); n_total = len(index_all)
    col_to_idx = {c: i for i, c in enumerate(sensor_cols)}
    if model is not None: model.eval()

    for i in range(store["i_done"] + 1, n_total):
        ts = pd.Timestamp(index_all[i])

        # ---- TSGuard ----
        hist_end = ts - pd.Timedelta(hours=1)
        if hist_end in base_df.index:
            hist_idx = base_df.loc[:hist_end].index[-window_hours:]
        else:
            hist_idx = base_df.index[base_df.index < ts][-window_hours:]
        hist_win = base_df.loc[hist_idx, sensor_cols].copy() if len(hist_idx) else pd.DataFrame(columns=sensor_cols)

        t0 = time.perf_counter()
        miss_row = orig_missing_mask.reindex(index=[ts], columns=sensor_cols).iloc[0]
        for col in sensor_cols:
            if bool(miss_row[col]):
                if not hist_win.empty and (model is not None):
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
                if pd.notna(pred_val) and col in store["tsg_imp"].columns:
                    store["tsg_imp"].at[ts, col] = pred_val
                    store["tsg_idx"][col].add(ts)  # pd.Timestamp
        store["t_times"].at[ts] = time.perf_counter() - t0

        # ---- PriSTI ----
        end_loc = base_df.index.get_loc(ts)
        if isinstance(end_loc, (int, np.integer)) and (end_loc + 1) >= eval_len:
            start_loc = end_loc - (eval_len - 1)
            time_index = base_df.index[start_loc:end_loc + 1]
            t0p = time.perf_counter()
            updated_df, info = impute_window_with_pristi(
                missing_df=base_df.copy(), sensor_cols=pristi_cols,
                target_timestamp=ts, model=SS["pristi_model"], device=device,
                eval_len=eval_len, nsample=100
            )
            store["p_times"].at[ts] = time.perf_counter() - t0p
            if info == "ok":
                pri_mask_win = orig_missing_mask.reindex(index=time_index, columns=pristi_cols).fillna(False)
                upd_win      = updated_df.reindex(index=time_index, columns=pristi_cols)
                filled_win   = pri_mask_win & upd_win.notna()

                current = store["pri_imp"].loc[time_index, pristi_cols]
                store["pri_imp"].loc[time_index, pristi_cols] = current.where(~filled_win, upd_win)

                for col in pristi_cols:
                    hits = filled_win.index[filled_win[col]]
                    if len(hits):
                        store["pri_idx"][col].update(hits)  # hits: DatetimeIndex → pd.Timestamp

        # temps cumulés (≤ eval_len derniers pas)
        t_idx = store["t_times"].dropna().index
        if len(t_idx):
            lastNt = t_idx[-min(eval_len, len(t_idx)):]
            store["sum_lastN_tsg_ms"] = float(np.nansum(store["t_times"].reindex(lastNt).values) * 1000.0)
        p_idx = store["p_times"].dropna().index
        if len(p_idx):
            lastNp = p_idx[-min(eval_len, len(p_idx)):]
            store["sum_lastN_pri_ms"] = float(np.nansum(store["p_times"].reindex(lastNp).values) * 1000.0)

        store["i_done"] = i
        SS["impute_store"][TL_SIG] = store

        # première fenêtre (36) → live
        if not store["first_window_shown"]:
            pct = min(100, int((i + 1) * 100 / max(eval_len, 1)))
            if prog: prog.progress(pct)
            if (i + 1) >= eval_len:
                end_i   = i
                start_i = max(0, end_i - 36 + 1)
                ts_idx  = base_df.index[start_i:end_i + 1]

                disp_tsg, tsg_sets = build_display(ts_idx, "tsg")
                disp_pri, pri_sets = build_display(ts_idx, "pri")
                cols_live = pristi_cols[:min(6, len(pristi_cols))]
                live_l.plotly_chart(plot_block(disp_tsg, tsg_sets, cols_live, "TSGuard (live)"), use_container_width=True)
                live_r.plotly_chart(plot_block(disp_pri, pri_sets, cols_live, "PriSTI (live)"),  use_container_width=True)

                if prog: prog.empty()
                if status_box: status_box.update(label="✅ First 36-step window ready.", state="complete")
                store["first_window_shown"] = True
                SS["impute_store"][TL_SIG] = store

                with sliders:
                    c1, c2 = st.columns(2)
                    c1.slider("Sensors to display", 1, min(36, len(pristi_cols)),
                              min(6, len(pristi_cols)), key=f"cmp_sensors_{cache_key_short}", on_change=_ui_only_changed)
                    c2.slider("Timestamps to display (≤36, last)", 6, 36,
                              36, key=f"cmp_steps_{cache_key_short}", on_change=_ui_only_changed)
                st.session_state["_ui_only_rerun"] = True
                st.stop()

    # ---------------- final render ----------------
    render_from_cache(
        st.session_state.get(f"cmp_sensors_{cache_key_short}", min(6, len(pristi_cols))),
        st.session_state.get(f"cmp_steps_{cache_key_short}",   min(36, eval_len))
    )



