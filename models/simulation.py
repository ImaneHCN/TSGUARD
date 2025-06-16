import streamlit as st
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import models.sim_helper as helper
import utils.visualization as vz

from utils.config import DEFAULT_VALUES

import plotly.graph_objects as go
import plotly.express as px
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
def train_model(train_file, positions_file, model_path='model.pth'):
    if st.session_state.get('graph_size'):
        graph_size = st.session_state['graph_size']
    else:
        graph_size = DEFAULT_VALUES["graph_size"]

    sensor_cols = train_file.columns[1:(graph_size + 1)]

    # Build fake positions or use the real ones
    sensor_positions = {i: (i % 5, i // 5) for i in range(graph_size)}
    edge_index = helper.build_edge_index(sensor_positions)

    model = helper.TSGUARD(input_dim=1, hidden_dim=32, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Transpose to get features per sensor over time
    all_features = torch.tensor(train_file[sensor_cols].values.T, dtype=torch.float)  # Shape: [10 nodes, time_steps]

    progress_bar = st.progress(0)  # Initialize progress bar
    status_container = st.container()  # Container to keep all updates

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

        progress = epoch + 1
        if progress % 10 == 0 or epoch == 0:
            loss_value = (total_loss / all_features.shape[1])
            print(f"Epoch {progress}, Loss: {loss_value:.4f}")
            progress_bar.progress(progress)

            with status_container:
                st.write(f"🔹 **Epoch {progress}** | 📉 **Loss:** `{loss_value:.4f}`")

    torch.save(model.state_dict(), model_path)
    print("✅ Model saved to", model_path)


def start_simulation(sim_file, positions, graph_placeholder, sliding_chart_placeholder, gauge_placeholder):
    import plotly.graph_objects as go
    from utils.visualization import draw_graph, draw_dashboard
    import time

    if st.session_state.get('graph_size'):
        graph_size = st.session_state['graph_size']
    else:
        graph_size = DEFAULT_VALUES["graph_size"]

    sensor_cols = sim_file.columns[1:(graph_size + 1)]

    # === Load imputed data ===
    imputed_df = pd.read_csv("pm25_imputed_live.csv")
    imputed_df["datetime"] = pd.to_datetime(imputed_df["datetime"], errors="coerce")
    sim_file["datetime"] = pd.to_datetime(sim_file["datetime"], errors="coerce")

    imputed_df["datetime"] = imputed_df["datetime"].dt.floor("H")
    sim_file["datetime"] = sim_file["datetime"].dt.floor("H")

    imputed_df.set_index("datetime", inplace=True)
    sim_file.set_index("datetime", inplace=True)

    # === Placeholders ===
    st.subheader("Sensor Simulation Graph")
    graph_placeholder = st.empty()
    time_placeholder = st.empty()

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

    sliding_window_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))
    global_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))
    station_colors = px.colors.qualitative.Plotly

    for current_time, row in sim_file.iterrows():
        if current_time >= pd.Timestamp("2014-05-09 09:00:00"):
            break
        # Skip timestamps at 00:00:00
        if current_time.time().strftime("%H:%M:%S") == "00:00:00":
            continue
        if current_time not in imputed_df.index:
            st.warning(f"No imputed data found for time {current_time}. Skipping.")
            continue

        imputed_row = imputed_df.loc[current_time]
        svals, sstates = [], []

        for col in sensor_cols:
            val = row[col]
            if pd.isna(val):
                imputed_val = imputed_row.get(col)
                svals.append(imputed_val)
                sstates.append(False)  # imputed
            else:
                svals.append(val)
                sstates.append(True)  # real

        svals = (svals + [None] * graph_size)[:graph_size]
        sstates = (sstates + [False] * graph_size)[:graph_size]

        row_data = {"datetime": current_time}
        for i, col in enumerate(sensor_cols):
            row_data[col] = svals[i]

        # === Update dataframes ===
        sliding_window_df = pd.concat([sliding_window_df, pd.DataFrame([row_data])], ignore_index=True)
        global_df = pd.concat([global_df, pd.DataFrame([row_data])], ignore_index=True)

        if len(sliding_window_df) > graph_size:
            sliding_window_df = sliding_window_df.tail(graph_size)

        # === Draw sensor graph ===
        fig = draw_graph(graph_size, svals, sstates, positions, current_time)
        time_placeholder.write(f"**Current Time**: {current_time}")
        graph_placeholder.plotly_chart(fig, use_container_width=True, key=f"graph_{current_time}")

        # === Build sstates per column for color-coding lines ===
        sstate_by_col = {}
        for col in sensor_cols:
            sstate_by_col[col] = []
            for j in range(len(sliding_window_df)):
                timestamp = sliding_window_df.iloc[j]["datetime"]
                real = not pd.isna(sim_file.loc[timestamp, col]) if timestamp in sim_file.index else False
                sstate_by_col[col].append(real)

        # === Plot sliding window chart (custom) ===
        sliding_fig = go.Figure()
        for i, col in enumerate(sensor_cols):
            color = station_colors[i % len(station_colors)]
            x_vals = sliding_window_df["datetime"]
            y_vals = sliding_window_df[col]
            states = sstate_by_col[col]

            segment_x, segment_y, segment_color = [], [], []
            for x, y, real in zip(x_vals, y_vals, states):
                if pd.isna(y): continue
                segment_x.append(x)
                segment_y.append(y)
                segment_color.append("red" if not real else color)

            sliding_fig.add_trace(go.Scatter(
                x=segment_x,
                y=segment_y,
                mode="lines+markers",
                name=f"Sensor {col}",
                line=dict(color=color),
                marker=dict(color=segment_color, size=6)
            ))

        sliding_fig.update_layout(
            title="10-Step Snapshot",
            xaxis_title="Time",
            yaxis_title="Sensor Value",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        sliding_chart_placeholder.plotly_chart(sliding_fig, use_container_width=True, key=f"sliding_{current_time}")

        # === Global dashboard and gauge ===
        df_line, gauge_fig = draw_dashboard(global_df.copy(), current_time, sensor_cols)
        with line3_col1:
            global_dashboard_placeholder.line_chart(df_line)
        with line3_col2:
            gauge_placeholder.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_{current_time}")


        time.sleep(1)


