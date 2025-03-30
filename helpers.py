import streamlit as st
import pandas as pd
import base64

def get_image_base64(image_path):
    """Convert image to base64 for HTML embedding"""
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    
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
    df = df.dropna()
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
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df = df.dropna()
    return df

@st.cache_data
def load_positions_data(file):
    """Load sensor positions from a CSV file."""
    df = pd.read_csv(file)
    df = df.dropna()
    positions = {}
    for i, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        positions[i] = (lon, lat)  # x=lon, y=lat for the plotly graph
    return positions

def init_files(training_data_file, sensor_data_file, positions_file):
    tr = load_training_data(training_data_file)
    df = load_sensor_data(sensor_data_file)
    pf = load_positions_data(positions_file)
    return tr, df, pf

def init_states(training_data_file, sensor_data_file, positions_file):
    if 'train_data' not in st.session_state:
        st.session_state.train_data = training_data_file.copy()
    if 'sim_data' not in st.session_state:
        st.session_state.sim_data = sensor_data_file.copy()
    if 'positions_date' not in st.session_state:
        st.session_state.positions_date = positions_file.copy()
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'training' not in st.session_state:
        st.session_state.training = False