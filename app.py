import streamlit as st
import pandas as pd
import time
import helpers as helper
import utils.visualization as vz
import components.sidebar as sidebar
import components.settings as settings
import components.buttons as buttons
import components.containers as containers
import models.simulation  as sim

from utils.config import DEFAULT_VALUES

# ----------------------------
# Page Configuration (Must Be First)
# ----------------------------
st.set_page_config(page_title="📡 TSGuard Sensor Dashboard", layout="wide")

# ----------------------------
# Main Layout
# ----------------------------
def main():
    st.markdown("""
        <h1 style='text-align: center;'>📡 TSGuard Sensor Streaming Simulation</h1>
        <hr style='border: 1px solid #ccc;'>
    """, unsafe_allow_html=True)
    
    training_data_file, sensor_data_file, positions_file = sidebar.init()
    
    if not training_data_file or not sensor_data_file or not positions_file:
        st.warning("Please upload training data, sensor data and position files to continue.")
        return

    # Load required files
    tr, df, pf = helper.init_files(training_data_file, sensor_data_file, positions_file)

    # init streamlit state variables
    helper.init_states(tr, df, pf)
    
    # Display settings
    settings.add_setting_panel()

    # Display buttons
    buttons.add_buttons()

    # Init graph containers
    graph_placeholder, gauge_placeholder, sliding_chart_placeholder = containers.init_containers()

    if st.session_state.training:
        # Training loop
        st.title("⏳ Training is running... Please wait.")
        with st.spinner("Training GNN model..."):
            model_path = DEFAULT_VALUES["training_file_path"]
            sim.train_model(tr, pf, model_path=model_path)
            st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
            st.success(f"✅ Training completed. Model saved to '`{model_path}`'")
            st.session_state.training = False
    
    if st.session_state.running:
        st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
        st.success("✅ Simulation is running. Click 'Stop Simulation' to end it.")

        sim.start_simulation(df, pf, graph_placeholder, sliding_chart_placeholder, gauge_placeholder)

if __name__ == '__main__':
    main()