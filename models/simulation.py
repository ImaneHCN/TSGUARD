import json
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import models.sim_helper as helper
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils.visualization import draw_graph, draw_dashboard
from utils.config import DEFAULT_VALUES
import time
import helpers as Help
import pydeck as pdk
import plotly.graph_objects as go
import pandas as pd
import numpy as np

SENSOR_SYMBOL = "circle"  # valid Scattermapbox symbol

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
        marker=dict(size=18, opacity=0.95, color=["#2ecc71"] * len(lats), symbol="circle"),
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
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # The weight matrix transforms features AFTER aggregation.
        # The input to this transformation has the same dimension as the number of nodes.
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x shape: (batch_size, num_nodes)
        # adj shape: (num_nodes, num_nodes)

        # Correct order of operations: (Â * X) * W
        # 1. Aggregate features from neighbors (Â * X)
        # To handle the batch, we compute Â * X^T and then transpose the result.
        aggregated_features = torch.spmm(adj, x.t()).t()  # (N,N) @ (N,B) -> (N,B) -> (B,N)

        # 2. Transform the aggregated features (result * W)
        output = torch.mm(aggregated_features, self.weight)  # (B,N) @ (N, F_out) -> (B, F_out)

        return output


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
    def __init__(self, X, y, mask, seq_len=24):
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

# -------------------------
# Train GCN Model
# -------------------------
# def train_model(train_file, positions_file, model_path='model.pth'):
#     if st.session_state.get('graph_size'):
#         graph_size = st.session_state['graph_size']
#     else:
#         graph_size = DEFAULT_VALUES["graph_size"]
#
#     sensor_cols = train_file.columns[1:(graph_size + 1)]
#
#     # Build fake positions or use the real ones
#     sensor_positions = {i: (i % 5, i // 5) for i in range(graph_size)}
#     edge_index = helper.build_edge_index(sensor_positions)
#     model = GraphConvolution(in_features=1,  out_features=1)
#     #model = helper.TSGUARD(input_dim=1, hidden_dim=32, output_dim=1)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     loss_fn = nn.MSELoss()
#     # Transpose to get features per sensor over time
#     all_features = torch.tensor(train_file[sensor_cols].values.T, dtype=torch.float)  # Shape: [10 nodes, time_steps]
#     progress_bar = st.progress(0)  # Initialize progress bar
#     status_container = st.container()
#     avg_loss =0.0 # Container to keep all updates
#     model.train()
#     for epoch in range(100):
#         total_loss = 0.0
#         for t in range(all_features.shape[1]):
#             x = all_features[:, t].unsqueeze(1)
#             y = x.clone()
#             optimizer.zero_grad()
#             out = model(x, edge_index)
#             loss = loss_fn(out, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         avg_loss += total_loss/all_features.shape[1]
#
#         progress = epoch + 1
#         if progress % 10 == 0 or epoch == 0:
#             loss_value = (total_loss / all_features.shape[1])
#             print(f"Epoch {progress}, Loss: {loss_value:.4f}")
#             progress_bar.progress(progress)
#
#             with status_container:
#                 st.write(f"🔹 **Epoch {progress}** | 📉 **Loss:** `{loss_value:.4f}`")
#
#     torch.save(model.state_dict(), model_path)
#     print("✅ Model saved to", model_path)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

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
    dist_matrix = np.zeros((num_sensors, num_sensors))

    for i in range(num_sensors):
        for j in range(i, num_sensors):
            lat1, lon1 = latlng_df.iloc[i]['latitude'], latlng_df.iloc[i]['longitude']
            lat2, lon2 = latlng_df.iloc[j]['latitude'], latlng_df.iloc[j]['longitude']
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    if threshold_type == 'gaussian':
        # Use a Gaussian kernel to compute weights
        sigma_sq = dist_matrix.std()**2 * sigma_sq_ratio
        adj_matrix = np.exp(-dist_matrix**2 / sigma_sq)
    else: # Simple binary threshold
        threshold = np.mean(dist_matrix) * 0.5
        adj_matrix = (dist_matrix <= threshold).astype(float)

    # Add self-loops
    np.fill_diagonal(adj_matrix, 1)

    # Symmetric normalization
    D = np.diag(np.sum(adj_matrix, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    adj_matrix_normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    return torch.FloatTensor(adj_matrix_normalized)


def train_model(train_file, missing_file, positions_file, epochs, model_path):
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import streamlit as st
    from torch.utils.data import DataLoader
    from io import BytesIO
    import json

    # ---- helpers ------------------------------------------------------------
    def norm_id_colname(c: object) -> str:
        """Normalize column names: keep 'datetime' as is; for all-digit names, zero-pad to 6."""
        s = str(c).strip()
        if s.lower() == "datetime":
            return "datetime"
        return s.zfill(6) if s.isdigit() else s

    def norm_df_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={c: norm_id_colname(c) for c in df.columns})

    def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        return df

    def load_positions_df(obj) -> pd.DataFrame:
        """
        Return DataFrame with columns: sensor_id, latitude, longitude.
        If obj is a dict like {0:(lon,lat), ...}, we convert it.
        """
        if isinstance(obj, dict):
            # dict assumed as {idx: (lon, lat)}
            df = (pd.DataFrame.from_dict(obj, orient="index", columns=["longitude", "latitude"])
                    .rename_axis("sensor_id").reset_index())
        elif isinstance(obj, pd.DataFrame):
            df = obj.copy()
        elif hasattr(obj, "read"):  # Streamlit UploadedFile
            raw = obj.read()
            # try CSV first
            try:
                df = pd.read_csv(BytesIO(raw))
            except Exception:
                data = json.loads(raw.decode("utf-8"))
                if isinstance(data, dict):
                    df = (pd.DataFrame.from_dict(data, orient="index",
                                                 columns=["longitude", "latitude"])
                            .rename_axis("sensor_id").reset_index())
                else:
                    df = pd.DataFrame(data)
        elif isinstance(obj, str):
            # path
            try:
                df = pd.read_csv(obj)
            except Exception:
                with open(obj, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    df = (pd.DataFrame.from_dict(data, orient="index",
                                                 columns=["longitude", "latitude"])
                            .rename_axis("sensor_id").reset_index())
                else:
                    df = pd.DataFrame(data)
        else:
            raise TypeError("positions_file must be dict, DataFrame, path, or file-like.")

        # normalize column names and aliases
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"lon": "longitude", "lng": "longitude", "lat": "latitude"})
        if "sensor_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "sensor_id"})

        required = {"sensor_id", "latitude", "longitude"}
        if not required.issubset(df.columns):
            raise ValueError(f"Positions file must contain {required}. Found: {list(df.columns)}")

        # Don't force zero-padding yet; we may need to remap 0..N-1 to real IDs later.
        df["sensor_id"] = df["sensor_id"].astype(str).str.strip()
        return df[["sensor_id", "latitude", "longitude"]]

    def remap_positions_ids_if_indexed(latlng: pd.DataFrame, ground_df: pd.DataFrame) -> pd.DataFrame:
        """
        If positions have sensor_id like '0','1',...,'N-1', remap them to the actual sensor
        column names from ground_df (e.g., '001001'...'001036') preserving order.
        """
        # columns present in ground_df (already normalized later)
        gcols = [c for c in ground_df.columns]
        # positions ids that look like integers 0..N-1
        if latlng["sensor_id"].str.fullmatch(r"\d+").all():
            ids = latlng["sensor_id"].astype(int)
            if ids.min() == 0 and ids.max() == len(latlng) - 1:
                # sort by the numeric id to ensure consistent order
                latlng = latlng.sort_values("sensor_id", key=lambda s: s.astype(int)).reset_index(drop=True)
                # map to ground_df columns by order
                if len(gcols) < len(latlng):
                    raise ValueError(
                        f"Positions rows ({len(latlng)}) exceed available sensor columns in ground_df ({len(gcols)})."
                    )
                latlng["sensor_id"] = gcols[:len(latlng)]
        else:
            # if already like '001001', leave as-is but normalize to 6-digit digit-only when appropriate
            latlng["sensor_id"] = latlng["sensor_id"].apply(
                lambda s: s if not s.isdigit() else s.zfill(6)
            )
        return latlng

    # ---- graph size & edges -------------------------------------------------
    graph_size = st.session_state.get('graph_size', DEFAULT_VALUES["graph_size"])
    sensor_positions = {i: (i % 5, i // 5) for i in range(graph_size)}
    edge_index = helper.build_edge_index(sensor_positions)

    # ---- load / normalize inputs -------------------------------------------
    # Expect train_file & missing_file already as DataFrames (from your Streamlit pipeline)
    ground_df = train_file.copy()
    missing_df = missing_file.copy()

    # Normalize column names: keep datetime, zero-pad purely digit names
    ground_df = norm_df_columns(ground_df)
    missing_df = norm_df_columns(missing_df)

    # Ensure datetime index
    ground_df = ensure_datetime_index(ground_df)
    missing_df = ensure_datetime_index(missing_df)

    # Load and normalize positions
    latlng = load_positions_df(positions_file)  # columns: sensor_id, latitude, longitude
    # If positions are 0..N-1, remap to actual sensor names from ground_df
    latlng = remap_positions_ids_if_indexed(latlng, ground_df)

    # Final normalization: ensure positions IDs match ground/missing style (zero-padded if digits)
    latlng["sensor_id"] = latlng["sensor_id"].apply(lambda s: s if not s.isdigit() else s.zfill(6))

    # ---- align columns ------------------------------------------------------
    sensor_cols = latlng["sensor_id"].tolist()  # e.g., ['001001', '001002', ...]
    # Sanity checks
    missing_in_ground = sorted(set(sensor_cols) - set(map(str, ground_df.columns)))
    missing_in_missing = sorted(set(sensor_cols) - set(map(str, missing_df.columns)))
    if missing_in_ground or missing_in_missing:
        raise KeyError(
            "Some expected sensor columns are missing.\n"
            f"- Missing in ground_df: {missing_in_ground}\n"
            f"- Missing in missing_df: {missing_in_missing}\n"
            f"- Sample ground_df columns: {list(ground_df.columns)[:8]}"
        )

    # Reindex columns to the exact order from positions
    ground_df = ground_df[sensor_cols]
    missing_df = missing_df[sensor_cols]
    # remove duplicate timestamps if any
    ground_df = ground_df[~ground_df.index.duplicated(keep="first")]
    missing_df = missing_df[~missing_df.index.duplicated(keep="first")]

    # keep only the timestamps present in BOTH
    common_index = ground_df.index.intersection(missing_df.index)

    if len(common_index) == 0:
        raise ValueError(
            "No overlapping timestamps between ground and missing data. "
            f"ground_df span: {ground_df.index.min()} -> {ground_df.index.max()}, "
            f"missing_df span: {missing_df.index.min()} -> {missing_df.index.max()}"
        )

    # align & sort

    ground_df = ground_df.loc[common_index].sort_index()
    missing_df = missing_df.loc[common_index].sort_index()

    # safety checks
    train_month_list = [1, 2, 4, 5, 7, 8, 10, 11]
    SEQ_LEN = 24
    if len(common_index) < 2 * SEQ_LEN:
        raise ValueError(f"Not enough overlapping timestamps after alignment: {len(common_index)} rows.")
    if not np.any(np.isin(missing_df.index.month, train_month_list)):
        raise ValueError("Train split is empty. Check your train_month_list or the data's date range.")

    # ---- numpy conversion & simple imputation -------------------------------
    ground_data = ground_df.to_numpy(dtype=np.float32)
    missing_data = missing_df.to_numpy(dtype=np.float32)

    # simple ffill/bfill for model inputs; evaluation still compares to ground_data
    imputed_df = missing_df.fillna(method='ffill').fillna(method='bfill')
    imputed_data = imputed_df.to_numpy(dtype=np.float32)

    # ---- adjacency & model --------------------------------------------------
    adj_matrix = create_adjacency_matrix(latlng, sigma_sq_ratio=0.1)

    NUM_NODES = missing_data.shape[1]
    GCN_HIDDEN = 64
    LSTM_HIDDEN = 64

    model = GCNLSTMImputer(
        adj=adj_matrix.to(device),
        num_nodes=NUM_NODES,
        in_features=NUM_NODES,
        gcn_hidden=GCN_HIDDEN,
        lstm_hidden=LSTM_HIDDEN,
        out_features=NUM_NODES
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # ---- split, scaling -----------------------------------------------------

    months = missing_df.index.month

    # mask where we have ground truth for NaNs in missing_data
    loss_mask = np.where(np.isnan(missing_data) & ~np.isnan(ground_data), 1.0, 0.0).astype(np.float32)
    train_slice = np.isin(months, train_month_list)

    train_imputed_data = imputed_data[train_slice]
    min_val = np.nanmin(train_imputed_data)
    max_val = np.nanmax(train_imputed_data)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaler = lambda x: (x - min_val) / denom
    inv_scaler = lambda x: x * denom + min_val
    scaler_params = {'min_val': float(min_val), 'max_val': float(max_val)}

    imputed_data_normalized = scaler(imputed_data)
    ground_data_normalized = scaler(ground_data)
    ground_data_normalized = np.nan_to_num(ground_data_normalized, nan=0.0)

    X_train = imputed_data_normalized[train_slice]
    y_train = ground_data_normalized[train_slice]
    mask_train = loss_mask[train_slice]


    BATCH_SIZE = 32
    train_dataset = SpatioTemporalDataset(X_train, y_train, mask_train, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ---- training loop ------------------------------------------------------
    progress_bar = st.progress(0.0)
    status_container = st.container()

    avg_train_loss = 0.0
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0
        num_batches = 0

        for inputs, targets, mask in train_loader:
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = masked_loss(outputs, targets, mask)
            if not torch.isnan(loss) and torch.sum(mask) > 0:
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                num_batches += 1

        epoch_loss = (total_train_loss / max(num_batches, 1))
        avg_train_loss += epoch_loss

        progress_bar.progress(int((epoch + 1) * 100 / max(epochs, 1)))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with status_container:
                st.write(f"🔹 **Epoch {epoch + 1}** | 📉 **Loss:** `{epoch_loss:.4f}`")

    avg_train_loss /= max(epochs, 1)

    # ---- save model ---------------------------------------------------------
    MODEL_SAVE_PATH = model_path if model_path else "gcn_lstm_imputer.pth"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f}")


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
    fig.update_layout(
        title="Global Time Series",
        xaxis_title="Time",
        yaxis_title="Sensor Value",
        margin=dict(l=20, r=20, t=40, b=20),
        legend_title="Sensors"
    )
    return fig

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

def start_simulation(sim_file, positions, graph_placeholder, sliding_chart_placeholder, gauge_placeholder):
    # 1) graph size & columns
    graph_size = st.session_state.get('graph_size', DEFAULT_VALUES["graph_size"])
    sensor_cols = sim_file.columns[1:(graph_size + 1)]

    # 2) align times / indexes
    imputed_df = pd.read_csv("pm25_imputed_live.csv")
    imputed_df["datetime"] = pd.to_datetime(imputed_df["datetime"], errors="coerce")
    sim_file["datetime"]    = pd.to_datetime(sim_file["datetime"], errors="coerce")
    imputed_df["datetime"]  = imputed_df["datetime"].dt.floor("h")
    sim_file["datetime"]    = sim_file["datetime"].dt.floor("h")
    imputed_df.set_index("datetime", inplace=True)
    sim_file.set_index("datetime", inplace=True)

    # 3) header + map in one container (title/time above the map)
    map_container = st.container()
    with map_container:
        title_slot = st.markdown("### Sensor Simulation Graph")
        time_slot  = st.empty()
        map_placeholder = st.empty()

    # 4) positions (once)
    latlng = positions_to_df(positions).copy()
    latlng["latitude"]  = pd.to_numeric(latlng["latitude"], errors="coerce")
    latlng["longitude"] = pd.to_numeric(latlng["longitude"], errors="coerce")
    sid_order = [str(s) for s in latlng["sensor_id"].tolist()]

    # 5) init map once
    if "sensor_map_fig" not in st.session_state:
        st.session_state.sensor_map_fig = init_sensor_map(latlng)

    # throttle map renders (~5 fps)
    if "last_map_render" not in st.session_state:
        st.session_state.last_map_render = 0.0
    MIN_INTERVAL = 0.30

    # cache last payload to avoid redundant renders
    if "last_map_payload" not in st.session_state:
        st.session_state.last_map_payload = None

    # 6) layout below the map
    st.markdown("---")
    line3_col1, line3_col2 = st.columns([2, 2])
    with line3_col1:
        st.subheader("Global Time Series")
        global_dashboard_placeholder = st.empty()
    with line3_col2:
        st.subheader("10-Step Snapshot")
        sliding_chart_placeholder = st.empty()

    st.markdown("---")
    st.subheader("Missed Data (%)")
    gauge_placeholder = st.empty()

    # 7) buffers for charts
    sliding_window_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))
    global_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))

    # palette for time-series lines
    sensor_custom_colors = [
        "#000000", "#003366", "#009999", "#006600", "#66CC66",
        "#FF9933", "#FFD700", "#708090", "#4682B4", "#99FF33"
    ]
    sensor_color_map = {col: sensor_custom_colors[i % len(sensor_custom_colors)] for i, col in enumerate(sensor_cols)}

    # static mapping from data columns to map sensor IDs (same order)
    col_to_sid = {col: sid_order[i] for i, col in enumerate(sensor_cols)}


    # 8) main loop (single loop only)
    for current_time, row in sim_file.iterrows():
        # (optional) skip limits / exact hour as you had
        if current_time >= pd.Timestamp("2014-05-09 09:00:00"):
            break
        if current_time.time().strftime("%H:%M:%S") == "00:00:00":
            continue
        if current_time not in imputed_df.index:
            continue

        # --- build svals / sstates FIRST ---
        imputed_row = imputed_df.loc[current_time]
        svals, sstates = [], []
        for col in sensor_cols:
            val = row[col]
            imputed_col = col.lstrip("0")  # mapping used in your data
            if pd.isna(val):
                svals.append(imputed_row.get(imputed_col))
                sstates.append(False)   # imputed
            else:
                svals.append(val)
                sstates.append(True)    # real

        # enforce size
        svals   = (svals + [None] * graph_size)[:graph_size]
        sstates = (sstates + [False] * graph_size)[:graph_size]

        # append to buffers for time-series figures
        row_data = {"datetime": current_time}
        for i, col in enumerate(sensor_cols):
            row_data[col] = svals[i]
        sliding_window_df = pd.concat([sliding_window_df, pd.DataFrame([row_data])], ignore_index=True)
        global_df        = pd.concat([global_df,        pd.DataFrame([row_data])], ignore_index=True)
        if len(sliding_window_df) > graph_size:
            sliding_window_df = sliding_window_df.tail(graph_size)

        # Build values/states keyed by columns
        vals_by_col = {col: (svals[i] if i < len(svals) else None) for i, col in enumerate(sensor_cols)}
        real_by_col = {col: (sstates[i] if i < len(sstates) else False) for i, col in enumerate(sensor_cols)}

        # Column -> SID mapping (precomputed above as col_to_sid)
        vals_by_sid = {col_to_sid[col]: vals_by_col[col] for col in sensor_cols if col in col_to_sid}
        real_by_sid = {col_to_sid[col]: real_by_col[col] for col in sensor_cols if col in col_to_sid}

        # Pack customdata rows (sid, value, status)
        customdata = [
            [sid, vals_by_sid.get(sid, "NA"), "Real" if real_by_sid.get(sid, False) else "Imputed"]
            for sid in sid_order
        ]
        colors = ["#2ecc71" if row[2] == "Real" else "#e74c3c" for row in customdata]

        fig = st.session_state.sensor_map_fig
        if not fig.data:
            fig.add_trace(go.Scattermapbox(mode="markers+text"))

        tr = fig.data[0]

        # ⚠️ ONLY these two lines change every tick:
        tr.customdata = customdata
        tr.marker.color = colors

        # Throttle + render when changed
        payload = (tuple(colors), tuple((row[1], row[2]) for row in customdata))
        now = time.time()
        should_render = (
                payload != st.session_state.last_map_payload
                and (now - st.session_state.last_map_render) >= MIN_INTERVAL
        )
        if should_render:
            time_slot.markdown(f"**Current Time:** {current_time}")
            map_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )
            st.session_state.last_map_render = now
            st.session_state.last_map_payload = payload

        if now - st.session_state.last_map_render >= MIN_INTERVAL:
            time_slot.markdown(f"**Current Time:** {current_time}")
            map_placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


        # ---- Sliding window (10-step) chart ----
        sstate_by_col = {}
        for col in sensor_cols:
            sstate_by_col[col] = []
            for j in range(len(sliding_window_df)):
                ts = sliding_window_df.iloc[j]["datetime"]
                real = not pd.isna(sim_file.loc[ts, col]) if ts in sim_file.index else False
                sstate_by_col[col].append(real)

        sliding_fig = go.Figure()
        for i, col in enumerate(sensor_cols):
            color = sensor_color_map[col]
            x_vals = list(sliding_window_df["datetime"])
            y_vals = list(sliding_window_df[col])
            states = sstate_by_col[col]

            segment_x, segment_y, segment_state = [], [], []
            for x, y, state in zip(x_vals, y_vals, states):
                if pd.isna(y):
                    continue
                segment_x.append(x); segment_y.append(y); segment_state.append(state)
                if len(segment_x) >= 2:
                    seg_color = "red" if any(not s for s in segment_state) else color
                    sliding_fig.add_trace(go.Scatter(
                        x=segment_x, y=segment_y, mode="lines+markers",
                        name=f"Sensor {col}",
                        line=dict(color=seg_color), marker=dict(size=6, color=seg_color),
                        showlegend=False
                    ))
                    segment_x, segment_y, segment_state = [segment_x[-1]], [segment_y[-1]], [segment_state[-1]]

            if len(segment_x) >= 2:
                seg_color = "red" if any(s is False for s in segment_state) else color
                sliding_fig.add_trace(go.Scatter(
                    x=segment_x, y=segment_y, mode="lines+markers",
                    name=f"Sensor {col}",
                    line=dict(color=seg_color), marker=dict(size=6, color=seg_color),
                    showlegend=False
                ))

        for col in sensor_cols:
            sliding_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=8, color=sensor_color_map[col]),
                legendgroup=col, showlegend=True, name=f"Sensor {col}"
            ))
        sliding_fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=8, color="red"),
            legendgroup="imputed", showlegend=True, name="Imputed Segment"
        ))
        sliding_fig.update_layout(
            title="10-Step Snapshot",
            xaxis_title="Time", yaxis_title="Sensor Value",
            margin=dict(l=20, r=20, t=40, b=20),
            legend_title="Sensors"
        )
        sliding_chart_placeholder.plotly_chart(sliding_fig, use_container_width=True, key=f"sliding_{current_time}")

        # ---- Global time series + Gauge ----
        full_ts_fig = draw_full_time_series(global_df.copy(), sim_file, sensor_cols, sensor_color_map)
        with line3_col1:
            global_dashboard_placeholder.plotly_chart(full_ts_fig, use_container_width=True, key=f"global_{current_time}")
        with line3_col2:
            gauge_fig = draw_gauge_figure(sim_file, current_time, sensor_cols)
            gauge_placeholder.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_{current_time}")

        time.sleep(1)
