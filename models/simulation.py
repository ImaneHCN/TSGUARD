import streamlit as st
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import models.sim_helper as helper
import utils.visualization as vz

from utils.config import DEFAULT_VALUES

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

        progress = epoch +1
        if progress % 10 == 0 or epoch == 0:
            loss_value = (total_loss / all_features.shape[1])
            print(f"Epoch {progress}, Loss: {loss_value:.4f}")
            progress_bar.progress(progress)
    
            with status_container:
                st.write(f"🔹 **Epoch {progress}** | 📉 **Loss:** `{loss_value:.4f}`")
            

    torch.save(model.state_dict(), model_path)
    print("✅ Model saved to", model_path)

def start_simulation(sim_file, positions, graph_placeholder, sliding_chart_placeholder, gauge_placeholder):

    if st.session_state.get('graph_size'):
        graph_size = st.session_state['graph_size']
    else:
        graph_size = DEFAULT_VALUES["graph_size"]

    sensor_cols = sim_file.columns[1:(graph_size + 1)]

    st.subheader("Sensor Simulation Graph")
    graph_placeholder = st.empty()
    time_placeholder = st.empty()

    # Sliding window for charts
    sliding_window_df = pd.DataFrame(columns=["datetime"] + list(sensor_cols))

     # === LINE 3: global time series (col1), missed data gauge (col2) ===
    st.markdown("---")
    line3_col1, line3_col2 = st.columns([2,2])
    with line3_col1:
        st.subheader("Global Time Series")
        global_dashboard_placeholder = st.empty()
    with line3_col2:
        st.subheader("10-Step Snapshot")
        sliding_chart_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("Missed Data (%)")
    gauge_placeholder = st.empty()

    for idx, row in sim_file.iterrows():
        current_time = row["datetime"]
        svals = [row[col] if col in sim_file.columns else None for col in sensor_cols]
        sstates = [False if pd.isna(v) else True for v in svals]

        svals = (svals + [None] * graph_size)[:graph_size]
        sstates = (sstates + [False] * graph_size)[:graph_size]

        row_data = {"datetime": current_time}
        for i, col in enumerate(sensor_cols):
            row_data[col] = svals[i]
        sliding_window_df = pd.concat([sliding_window_df, pd.DataFrame([row_data])], ignore_index=True)
        if len(sliding_window_df) > graph_size:
            sliding_window_df = sliding_window_df.tail(graph_size)
        
        sliding_data = sliding_window_df.set_index("datetime")
         #with line:
        sliding_chart_placeholder.line_chart(sliding_data)

        # 2) Sensor simulation graph
        fig = vz.draw_graph(graph_size, svals, sstates, positions, current_time)
        #with line:
        time_placeholder.write(f"**Current Time**: {current_time}")
        graph_placeholder.plotly_chart(fig, use_container_width=True, key=f"graph_{idx}")

        # 3) Global time series + gauge
        df_line, gauge_fig = vz.draw_dashboard(sim_file, current_time, sensor_cols)
        with line3_col1:
            global_dashboard_placeholder.line_chart(df_line)
        with line3_col2:
            gauge_placeholder.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_{idx}")

        time.sleep(1)