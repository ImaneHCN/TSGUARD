import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np
from datetime import datetime
import base64
import json
from TSGUARD_GNN import train_model
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class TDGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TDGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# ----------------------------
# Global Default Values
# ----------------------------
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
# ----------------------------
# Page Configuration (Must Be First)
# ----------------------------
st.set_page_config(page_title="📡 Sensor Dashboard", layout="wide")


# ----------------------------
# Data Loading (Cached)
# ----------------------------
@st.cache_data
def load_training_data(file):
    """Load training data from a .txt file (CSV format)."""
    df = pd.read_csv(file)
    if "datetime" not in df.columns:
        for candidate in ["timestamp", "date"]:
            if candidate in df.columns:
                df.rename(columns={candidate: "datetime"}, inplace=True)
                break
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    return df


@st.cache_data
def load_sensor_data(file):
    """Load sensor data from a .txt file (CSV format)."""
    df = pd.read_csv(file)
    if "datetime" not in df.columns:
        for candidate in ["timestamp", "date"]:
            if candidate in df.columns:
                df.rename(columns={candidate: "datetime"}, inplace=True)
                break
    print(df.head())
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    return df


@st.cache_data
def load_positions_data(file):
    """Load sensor positions from a CSV file."""
    df = pd.read_csv(file)
    positions = {}
    for i, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        positions[i] = (lon, lat)  # x=lon, y=lat for the plotly graph
    return positions


# ----------------------------
# Setting Management
# ----------------------------
def add_setting_panel():
    with st.expander("⚙️ Settings", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["📌 Constraints", "📈 Threshold", "📊 Missing values", "🕸️ Graph Options"])

        with tab1:
            add_constraints_panel()
            st.json(st.session_state.get('constraints', []))

        with tab2:
            add_threshold_panel()

        with tab3:
            add_missing_value_panel()

        with tab4:
            add_graph_opt_panel()


# ----------------------------
# Constraints Management
# ----------------------------
def add_constraints_panel():
    if 'constraints' not in st.session_state:
        st.session_state['constraints'] = []
    ctype = st.radio("Select Constraint Type", options=["📍 Spatial", "⏳ Temporal"], key="constraint_type")
    if "Spatial" in ctype:
        st.markdown("#### 📍 Spatial Constraints")
        # Distance with unit selection
        col1, col2 = st.columns([2, 1])
        with col1:
            spatial_distance = st.number_input("📏 Distance Threshold", value=2.0, step=0.1, key="spatial_distance")
        with col2:
            distance_unit = st.selectbox("Unit", ["km", "miles"], key="distance_unit")

        # Convert miles to km for standardization
        spatial_distance_km = 0
        spatial_distance_miles = 0
        if distance_unit == "miles":
            spatial_distance_km = round(spatial_distance * 1.60934, 2)  # 1 mile = 1.60934 km
            spatial_distance_miles = spatial_distance
        else:
            spatial_distance_km = spatial_distance
            spatial_distance_miles = round(spatial_distance / 1.60934, 2)  # 1 mile = 1.60934 km

        spatial_diff = st.number_input("📊 Max Sensor Difference", value=5.0, step=0.1, key="spatial_diff")
        if st.button("Add Spatial Constraint", key="add_spatial"):
            st.session_state['constraints'].append(
                {"type": "Spatial", "distance in km": spatial_distance_km, "distance in miles": spatial_distance_miles,
                 "diff": spatial_diff})
            st.success("Spatial constraint added.")
    else:
        st.markdown("#### ⏳ Temporal Constraints")
        month = st.selectbox("🌦️ Month",
                             options=["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], key="month")
        constraint_option = st.selectbox("📉 Constraint Option", options=["Greater than", "Less than"],
                                         key="constraint_option")
        temp_threshold = st.number_input("📈 Threshold Value", value=50.0, step=0.1, key="temp_threshold")
        if st.button("Add Temporal Constraint", key="add_temporal"):
            st.session_state['constraints'].append(
                {"type": "Temporal", "month": month, "option": constraint_option, "temp_threshold": temp_threshold})
            st.success("Temporal constraint added.")


# ----------------------------
# Missing value Management
# ----------------------------
def add_missing_value_panel():
    if 'missing_value_thresholds' not in st.session_state:
        st.session_state['missing_value_thresholds'] = []

    st.markdown("### 🛠 Define Missing Value Thresholds")
    st.markdown(
        "Please specify the missing value percentage ranges for different risk states (Green: Low, Yellow: Medium, Red: High).")

    col1, col2 = st.columns(2)
    with col1:
        green_min = st.number_input("🟢 Green Min", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_green_min"],
                                    step=1)
        yellow_min = st.number_input("🟡 Yellow Min", min_value=0, max_value=100,
                                     value=DEFAULT_VALUES["gauge_yellow_min"], step=1)
        red_min = st.number_input("🔴 Red Min", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_red_min"],
                                  step=1)

    with col2:
        green_max = st.number_input("🟢 Green Max", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_green_max"],
                                    step=1)
        yellow_max = st.number_input("🟡 Yellow Max", min_value=0, max_value=100,
                                     value=DEFAULT_VALUES["gauge_yellow_max"], step=1)
        red_max = st.number_input("🔴 Red Max", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_red_max"],
                                  step=1)

    if st.button("✅ Save Thresholds"):
        if not (green_min <= green_max <= yellow_min <= yellow_max <= red_min <= red_max):
            st.error("🚨 Invalid threshold ranges. Ensure consistency between min/max values.")
        else:
            st.session_state['missing_value_thresholds'] = {
                "Green": (green_min, green_max),
                "Yellow": (yellow_min, yellow_max),
                "Red": (red_min, red_max)
            }
            st.success("✅ Missing value thresholds saved successfully.")


# ----------------------------
# Threshold Management
# ----------------------------
def add_threshold_panel():
    if 'sigma_threshold' not in st.session_state:
        st.session_state['sigma_threshold'] = DEFAULT_VALUES["sigma_threshold"]

    st.markdown("Please specify the allowed delay threshold before a sensor is considered as having a missing value.")
    st.markdown("The default value is **" + str(DEFAULT_VALUES["sigma_threshold"]) + " minutes**.")
    col1, col2 = st.columns([2, 1])
    with col1:
        threshold = st.number_input("📈 Threshold Value", value=DEFAULT_VALUES["sigma_threshold"], step=1,
                                    key="threshold")
    with col2:
        time_unit = st.selectbox("Unit", ["minutes", "hours"], key="time_unit")
    if st.button("Set the delay threshold", key="set_sigma_threshold"):
        st.session_state['sigma_threshold'] = threshold
        st.success("Delay 'Sigma' threshold set to : **" + str(threshold) + " " + time_unit + "**.")


# ----------------------------
# Graph Management
# ----------------------------
def add_graph_opt_panel():
    if 'graph_size' not in st.session_state:
        st.session_state['graph_size'] = DEFAULT_VALUES["graph_size"]

    st.markdown("### Configure Graph Size")
    st.markdown("Specify the number of sensors (nodes) in the graph.")
    st.markdown(f"**Default:** {DEFAULT_VALUES['graph_size']} sensors")

    g_size = st.number_input("📶 Graph Size", value=DEFAULT_VALUES["graph_size"], step=1, key="g_size")

    if st.button("Save", key="set_graph_size"):
        st.session_state['graph_size'] = g_size
        st.success("The graph size set to : **" + str(g_size) + " sensors**.")


# ----------------------------
# Sensor Graph with Icons
# ----------------------------
def draw_sensor_graph(graph_size, sensor_values, sensor_states, positions, current_time):
    node_x, node_y, colors, texts = [], [], [], []
    for sensor_idx in range(graph_size):
        x, y = positions[sensor_idx]
        node_x.append(x)
        node_y.append(y)
        colors.append("#90ee90" if sensor_states[sensor_idx] else "red")
        val = sensor_values[sensor_idx]
        texts.append("miss" if pd.isna(val) else str(val))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=graph_size, color=colors, line=dict(color="black", width=2)),
        text=texts,
        textposition="top center",
        hoverinfo="none"
    ))
    fig.update_layout(
        title=f"Sensor Graph - Current Time: {current_time}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# ----------------------------
# 3) Helper: encode sensor icon
# ----------------------------
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()


# ----------------------------
# 4) Draw Sensor Graph
# ----------------------------
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


# ----------------------------
# 5) Build the global time series + gauge
# ----------------------------
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


# ----------------------------
# Main Layout
# ----------------------------
def main():
    st.markdown("""
        <h1 style='text-align: center;'>📡 TSGuard Sensor Streaming Simulation</h1>
        <hr style='border: 1px solid #ccc;'>
    """, unsafe_allow_html=True)

    # --- Initialize session state ---
    for key, default in {
        "train_data": None,
        "sim_data": None,
        "running": False,
        "training": False,
        "constraints": [],
        "missing_value_thresholds": [],
        "graph_size": DEFAULT_VALUES["graph_size"],
        "sigma_threshold": DEFAULT_VALUES["sigma_threshold"],
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Upload inputs with UNIQUE keys ---
    training_data_file = st.sidebar.file_uploader(
        "🧠 Upload Training Data (.csv or .txt)", type=["csv", "txt"], key="file_uploader_training"
    )
    sensor_data_file = st.sidebar.file_uploader(
        "📂 Upload Sensor Data (.csv or .txt)", type=["csv", "txt"], key="file_uploader_sensor"
    )
    positions_file = st.sidebar.file_uploader(
        "📍 Upload Sensor Positions (.csv or .txt)", type=["csv", "txt"], key="file_uploader_positions"
    )

    # --- Ensure all files are uploaded ---
    if not sensor_data_file or not positions_file or not training_data_file:
        st.warning("Please upload **training**, **sensor**, and **position** data files to continue.")
        return

    # --- Load data ---
    tr = load_training_data(training_data_file)
    df = load_sensor_data(sensor_data_file)
    positions = load_positions_data(positions_file)

    # --- Store in session state ---
    st.session_state.train_data = tr.copy()
    st.session_state.sim_data = df.copy()

    # --- Settings Panel ---
    add_setting_panel()

    # --- Control Buttons ---
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Start Simulation", use_container_width=True):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Stop Simulation", use_container_width=True):
            st.session_state.running = False
    with col3:
        if st.button("🧠 Start Training", use_container_width=True):
            st.session_state.training = True

    # --- UI Placeholders ---
    graph_placeholder = st.empty()
    gauge_placeholder = st.empty()
    sliding_chart_placeholder = st.empty()
    time_placeholder = st.empty()
    global_dashboard_placeholder = st.empty()

    # --- Run Simulation ---
    if st.session_state.running:
        st.success("✅ Simulation is running. Click 'Stop Simulation' to end it.")

        graph_size = st.session_state.graph_size
        sensor_cols = df.columns[1:(graph_size + 1)]

        # Sliding window for charts
        sliding_window_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))

        for idx, row in df.iterrows():
            current_time = row["datetime"]
            svals = [row[col] if col in df.columns else None for col in sensor_cols]
            sstates = [False if pd.isna(v) else True for v in svals]

            svals = (svals + [None] * graph_size)[:graph_size]
            sstates = (sstates + [False] * graph_size)[:graph_size]

            row_data = {"datetime": current_time}
            for i, col in enumerate(sensor_cols):
                row_data[col] = svals[i]
            sliding_window_df = pd.concat([sliding_window_df, pd.DataFrame([row_data])], ignore_index=True)
            if len(sliding_window_df) > graph_size:
                sliding_window_df = sliding_window_df.tail(graph_size)

            # Draw Graph
            fig = draw_graph(graph_size, svals, sstates, positions, current_time)
            time_placeholder.write(f"**Current Time:** {current_time}")
            graph_placeholder.plotly_chart(fig, use_container_width=True, key=f"graph_{idx}")

            # Charts
            sliding_chart_placeholder.line_chart(sliding_window_df.set_index("datetime"))
            df_line, gauge_fig = draw_dashboard(df, current_time, sensor_cols)
            global_dashboard_placeholder.line_chart(df_line)
            gauge_placeholder.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_{idx}")

            time.sleep(1)

    # --- Run Training ---
    if st.session_state.training:
        st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
        st.title("⏳ Training is running... Please wait.")
        with st.spinner("Training GNN model..."):
            try:
                model_path = "model_tdgnn.pth"
                train_model(training_data_file, positions_file, model_path=model_path)
                st.success(f"✅ Training completed. Model saved to `{model_path}`")
            except Exception as e:
                st.error(f"Training failed: {e}")
            st.session_state.training = False


def run_streaming_with_imputation(df, positions, model, edge_index, graph_size, sensor_cols,
                                   sliding_chart_placeholder, global_dashboard_placeholder,
                                   graph_placeholder, gauge_placeholder, time_placeholder):
    sliding_window_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))
    global_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))
    last_seen = {col: None for col in sensor_cols}
    delta_threshold = pd.Timedelta(minutes=st.session_state.get('sigma_threshold', 10))

    colors_map = {col: f"C{i}" for i, col in enumerate(sensor_cols)}  # consistent colors

    for idx, row in df.iterrows():
        current_time = row["datetime"]
        svals = []
        sstates = []
        imputed_flags = []

        x = torch.full((graph_size, 1), float('nan'))

        for i, col in enumerate(sensor_cols):
            val = row[col] if col in df.columns else None
            print(val)
            if pd.isna(val):
                # Missing, will check if imputation is needed
                if last_seen[col] is None or current_time - last_seen[col] > delta_threshold:
                    sstates.append(False)
                    imputed_flags.append(True)
                    svals.append(None)
                else:
                    sstates.append(False)
                    imputed_flags.append(False)
                    svals.append(None)
            else:
                sstates.append(True)
                imputed_flags.append(False)
                last_seen[col] = current_time
                svals.append(val)
                x[i, 0] = val
                print(f"[{current_time}] Sensor {col} — value from file: {val}")

        # Imputation
        for i, (missing, should_impute) in enumerate(zip(pd.isna(x[:, 0]), imputed_flags)):
            if missing and should_impute:
                x_impute = x.clone()
                x_impute[i, 0] = 0
                with torch.no_grad():
                    out = model(x_impute, edge_index)
                    imputed_value = out[i, 0].item()
                    x[i, 0] = imputed_value
                    svals[i] = imputed_value
                    sstates[i] = False
                    print(f"[{current_time}] Sensor {sensor_cols[i]} — imputed value: {imputed_value:.2f}")

        # Sensor graph
        fig = draw_graph(graph_size, svals, sstates, positions, current_time)
        for i, imp in enumerate(imputed_flags):
            if imp:
                val = svals[i]
                fig.add_annotation(
                    x=positions[i][0], y=positions[i][1] + 0.02,
                    text=f"<b>{val:.1f}</b>", showarrow=False, font=dict(color="red", size=12)
                )
        time_placeholder.write(f"**Current Time:** {current_time}")
        graph_placeholder.plotly_chart(fig, use_container_width=True)

        # Update DFs
        row_data = {"datetime": current_time}
        for i, col in enumerate(sensor_cols):
            row_data[col] = svals[i]
        global_df = pd.concat([global_df, pd.DataFrame([row_data])], ignore_index=True)
        sliding_window_df = pd.concat([sliding_window_df, pd.DataFrame([row_data])], ignore_index=True)
        if len(sliding_window_df) > 10:
            sliding_window_df = sliding_window_df.tail(10)

        # Charts
        global_line = global_df.set_index("datetime")
        sliding_line = sliding_window_df.set_index("datetime")

        global_dashboard_placeholder.line_chart(global_line, use_container_width=True)
        sliding_chart_placeholder.line_chart(sliding_line, use_container_width=True)

        # Gauge
        df_line, gauge_fig = draw_dashboard(global_df, current_time, sensor_cols)
        gauge_placeholder.plotly_chart(gauge_fig, use_container_width=True)

        time.sleep(1)

if __name__ == '__main__':
    main()