import streamlit as st

def init():

    training_data_file = st.sidebar.file_uploader("🧠 Upload Training Data (.csv or .txt)", type=["csv", "txt"])
    sensor_data_file = st.sidebar.file_uploader("🛰️ Upload Sensor Data (.csv or .txt)", type=["csv", "txt"])
    positions_file = st.sidebar.file_uploader("🗺️ Upload Sensor Positions (.csv or .txt)", type=["csv", "txt"])

    return training_data_file, sensor_data_file, positions_file