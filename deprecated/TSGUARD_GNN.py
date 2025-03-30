# TSGuard_demo.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np
import time
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from datetime import datetime, timedelta
import base64
# -------------------------
# Define Simple GCN Model
# -------------------------
class TSGuard(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TSGuard, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# -------------------------
# Build Adjacency Matrix (KNN or geographical distance-based)
# -------------------------
def build_edge_index(sensor_positions):
    edge_list = []
    sensor_ids = list(sensor_positions.keys())
    for i in range(len(sensor_ids)):
        for j in range(len(sensor_ids)):
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

# -------------------------
# Train GCN Model
# -------------------------
def train_model(train_file, positions_file, model_path='model.pth'):
    df = pd.read_csv(train_file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.dropna()

    sensor_cols = df.columns[1:11]  # Assuming 10 sensors

    # Build fake positions or use the real ones
    sensor_positions = {i: (i % 5, i // 5) for i in range(10)}
    edge_index = build_edge_index(sensor_positions)

    model = TSGuard(input_dim=1, hidden_dim=32, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Transpose to get features per sensor over time
    all_features = torch.tensor(df[sensor_cols].values.T, dtype=torch.float)  # Shape: [10 nodes, time_steps]

    for epoch in range(100):
        total_loss = 0
        for t in range(all_features.shape[1]):
            x = all_features[:, t].unsqueeze(1)  # Shape: [10 nodes, 1 feature]
            y = x.clone()

            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / all_features.shape[1]:.4f}")

    torch.save(model.state_dict(), model_path)
    print("✅ Model saved to", model_path)


# -------------------------
# Streaming + Imputation Logic
# -------------------------

def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

def draw_graph(graph_size, sensor_values, sensor_states, positions, current_time):
    """
    Create a Plotly graph for 10 sensors.
    - Circle marker size=10
    - Very small icon overlay (sizex=0.03, sizey=0.03)
    """
    # This is used to persist the on hover text
    # when the graph is rendered.
    node_x, node_y, colors, texts, hover_texts, custom_data = [], [], [], [], [], []
    for sensor_idx in range(graph_size):
        x, y = positions[sensor_idx]
        node_x.append(x)
        node_y.append(y)
        colors.append("#90ee90" if sensor_states[sensor_idx] else "red")
        val = sensor_values[sensor_idx]
        texts.append("miss" if pd.isna(val) else str(val))

        sensor_json = {
            "Sensor ID": sensor_idx,
            "Value": None if pd.isna(val) else val,
            "State": "Active" if sensor_states[sensor_idx] else "Inactive",
            "Position": {"x": x, "y": y}
        }
        hover_texts.append(json.dumps(sensor_json, indent=2))
        custom_data.append(json.dumps(sensor_json))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=graph_size, color=colors, line=dict(color="black", width=2)),
        text=texts,
        textposition="top center",
        hoverinfo="text",
        hovertext=hover_texts,  # On hover
        customdata=custom_data,  # On click
    ))

    icon_base64 = get_image_base64("captor_icon.png")
    for sensor_idx in range(graph_size):
        x, y = positions[sensor_idx]
        fig.add_layout_image(
            dict(
                source=icon_base64,
                xref="x",
                yref="y",
                x=x,
                y=y,
                # Make these very small to shrink the icon
                sizex=0.03,
                sizey=0.03,
                xanchor="center",
                yanchor="middle",
                opacity=1
            )
        )

    fig.update_layout(
        title=f"Sensor Graph - Current Time: {current_time}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
DEFAULT_VALUES = {
    "sigma_threshold": 10,
    "gauge_green_min": 0,
    "gauge_green_max": 20,
    "gauge_yellow_min": 20,
    "gauge_yellow_max": 50,
    "gauge_red_min": 50,
    "gauge_red_max": 100,
    "graph_size": 10,
}

def draw_dashboard(df, current_time, sensor_cols):
    # Get missing values thresholds
    green_min, green_max = DEFAULT_VALUES["gauge_green_min"], DEFAULT_VALUES["gauge_green_max"]
    yellow_min, yellow_max = DEFAULT_VALUES["gauge_yellow_min"], DEFAULT_VALUES["gauge_yellow_max"]
    red_min, red_max = DEFAULT_VALUES["gauge_red_min"], DEFAULT_VALUES["gauge_red_max"]
    if st.session_state.get('missing_value_thresholds'):
        thresholds = st.session_state['missing_value_thresholds']
        green_min, green_max = thresholds.get("Green", (None, None))
        yellow_min, yellow_max = thresholds.get("Yellow", (None, None))
        red_min, red_max = thresholds.get("Red", (None, None))

    df_filtered = df[df["datetime"] <= current_time].copy()
    df_line = df_filtered.set_index("datetime")[sensor_cols]

    total = df_filtered[sensor_cols].size
    missed = df_filtered[sensor_cols].isna().sum().sum()
    pmiss = (missed / total) * 100 if total > 0 else 0

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pmiss,
        title={"text": "Overall Missed Data (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if pmiss > 20 else "green"},
            "steps": [
                {"range": [green_min, green_max], "color": "lightgreen"},
                {"range": [yellow_min, yellow_max], "color": "yellow"},
                {"range": [red_min, red_max], "color": "red"}
            ]
        }
    ))
    gauge_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return df_line, gauge_fig

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from datetime import timedelta

def simulate_streaming(test_file, global_dashboard_placeholder, sliding_chart_placeholder, positions, model_path='model.pth'):
    df = pd.read_csv(test_file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    sensor_cols = df.columns[1:11]  # Assuming 10 sensors

    graph_size = 10
    edge_index = build_edge_index({i: (i % 5, i // 5) for i in range(graph_size)})
    model = TSGuard(input_dim=1, hidden_dim=32, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    last_received_time = {i: None for i in range(graph_size)}
    time_deltas = {i: [] for i in range(graph_size)}
    sigma_threshold = st.session_state.get('sigma_threshold', 10)

    global_window_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))
    sliding_window_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))

    for i, row in df.iterrows():
        current_time = row['datetime']
        inputs = torch.full((graph_size, 1), float('nan'))
        svals = [row[col] if col in df.columns else None for col in sensor_cols]
        sstates = [False] * graph_size
        imputed_this_step = {}

        for j in range(graph_size):
            val = svals[j]
            if not pd.isna(val):
                if last_received_time[j] is not None:
                    delta = (current_time - last_received_time[j]).total_seconds() / 60.0  # minutes
                    time_deltas[j].append(delta)
                last_received_time[j] = current_time
                inputs[j, 0] = val
                sstates[j] = True

        # Run imputation for delayed sensors only
        for j in range(graph_size):
            if pd.isna(inputs[j, 0]):
                if last_received_time[j] is None:
                    continue
                delta_t = (current_time - last_received_time[j]).total_seconds() / 60.0
                if len(time_deltas[j]) >= 2:
                    mu = np.mean(time_deltas[j])
                    sigma = np.std(time_deltas[j])
                    if delta_t > mu + sigma_threshold * sigma:
                        with torch.no_grad():
                            filled_input = inputs.clone()
                            filled_input[j, 0] = 0  # dummy init
                            imputed = model(filled_input, edge_index)
                            value = imputed[j, 0].item()
                            inputs[j, 0] = value
                            imputed_this_step[j] = value
                            sstates[j] = True

        row_data = {"datetime": current_time}
        for j, col in enumerate(sensor_cols):
            row_data[col] = inputs[j, 0].item() if not torch.isnan(inputs[j, 0]) else np.nan

        global_window_df = pd.concat([global_window_df, pd.DataFrame([row_data])], ignore_index=True)
        sliding_window_df = pd.concat([sliding_window_df, pd.DataFrame([row_data])], ignore_index=True)
        if len(sliding_window_df) > graph_size:
            sliding_window_df = sliding_window_df.tail(graph_size)

        global_dashboard_placeholder.line_chart(global_window_df.set_index("datetime"))

        # Plot sliding chart with imputed markers
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.get_cmap("tab10", len(sensor_cols))
        for j, col in enumerate(sensor_cols):
            series = sliding_window_df[col].astype(float).values
            ax.plot(sliding_window_df["datetime"], series, label=col, color=colors(j), linestyle='-')
            if j in imputed_this_step:
                ax.plot(sliding_window_df["datetime"].iloc[-1], series[-1], 'ro', label=f"Imputed {col}")
        ax.set_title("Sensor Values with Imputation Highlights")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Sensor Value")
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True)
        sliding_chart_placeholder.pyplot(fig)

        time.sleep(2)


def simulate_streaming_new(model, edge_index, inputs, last_received_time, current_time):
    imputed_values = {}
    for j in range(inputs.shape[1]):
        if pd.isna(inputs[0, j]):
            if last_received_time[j] is None or current_time - last_received_time[j] > timedelta(seconds=3):
                with torch.no_grad():
                    filled_input = inputs.clone()
                    filled_input[0, j] = 0  # dummy init
                    imputed = model(filled_input, edge_index)
                    value = imputed[0, j].item()
                    imputed_values[j] = value
                    last_received_time[j] = current_time

    return imputed_values, last_received_time

