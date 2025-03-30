import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import helpers as helper
from utils.config import COLOR_MAP
from utils.config import DEFAULT_VALUES

# ----------------------------
# Draw Sensor Graph
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

    icon_base64 = helper.get_image_base64("images/captor_icon.png")
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
# Build the global time series + gauge
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
    pmiss = (missed / total)*100 if total>0 else 0

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pmiss,
        title={"text":"Overall Missed Data (%)"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"red" if pmiss>20 else "green"},
            "steps":[
                {"range":[green_min,green_max], "color":"lightgreen"},
                {"range":[yellow_min,yellow_max],"color":"yellow"},
                {"range":[red_min,red_max],"color":"red"}
            ]
        }
    ))
    gauge_fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    return df_line, gauge_fig
# ----------------------------