import streamlit as st
import pandas as pd
import time
import helpers as helper
import utils.visualization as vz
import components.sidebar as sidebar
import components.settings as settings
import components.buttons as buttons
import components.containers as containers
import json
import torch
from utils.config import DEFAULT_VALUES
from models.simulation import run_simulation_with_live_imputation, train_model, GCNLSTMImputer  # your new function
from pathlib import Path
import os, json, torch

def resolve_paths(model_path: str = None, scaler_path: str = None):
    """
    Make sure we point to the same files you wrote in training.
    Example training save:
      model -> 'generated/model_TSGuard.pth'
      scaler -> 'generated/model_TSGuard_scaler.json'
    """
    if model_path is None:
        # default to your training output
        model_path = "generated/model_TSGuard.pth"
    if scaler_path is None:
        # same stem + _scaler.json by convention
        p = Path(model_path)
        scaler_path = str(p.with_name(p.stem + "_scaler.json"))

    mp = Path(model_path).expanduser().resolve()
    sp = Path(scaler_path).expanduser().resolve()

    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp}\nCWD: {Path.cwd()}")
    if not sp.exists():
        raise FileNotFoundError(f"Scaler file not found: {sp}\nCWD: {Path.cwd()}")

    return str(mp), str(sp)

def load_scaler_from_json(scaler_json_path: str):
    with open(scaler_json_path, "r") as f:
        params = json.load(f)
    min_val = float(params["min_val"])
    max_val = float(params["max_val"])
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0

    def scaler(x):     return (x - min_val) / denom
    def inv_scaler(x): return x * denom + min_val

    return scaler, inv_scaler

def load_state_dict_into(model: torch.nn.Module, model_path: str, device: torch.device):
    state = torch.load(model_path, map_location=device)
    # handle both {'state_dict': ...} and raw state_dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model

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
    tr, df, pf, sensor_list = helper.init_files(training_data_file, sensor_data_file, positions_file)

    # init streamlit state variables
    helper.init_states(tr, df, pf)
    
    # Display settings
    settings.add_setting_panel()

    # Display buttons
    buttons.add_buttons()

    # Init graph containers
    graph_placeholder, gauge_placeholder, sliding_chart_placeholder = containers.init_containers()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = None

    if st.session_state.training:
        # Training loop
        st.title("⏳ Training is running... Please wait.")
        with st.spinner("Training GNN model..."):
            model_path = DEFAULT_VALUES["training_file_path"]
            model= train_model(tr,df, pf, 10, model_path=model_path)
            st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
            st.success(f"✅ Training completed. Model saved to '`{model_path}`'")
            st.session_state.training = False
    
    if st.session_state.running:
        st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
        st.success("✅ Simulation is running. Click 'Stop Simulation' to end it.")

        # Resolve files to EXACT ones used in training
        MODEL_PATH, SCALER_PATH = resolve_paths(
            model_path="generated/model_TSGuard.pth",  # <-- your real saved file
            scaler_path="generated/model_TSGuard_scaler.json"  # <-- your real saved file
        )

        # Build the model the SAME WAY as in training (NO adjacency creation here if you don’t need it)
        #model = GCNLSTMImputer(...).to(device)  # your exact ctor args from training

        # Load weights
        #load_state_dict_into(model, MODEL_PATH, device)

        # Load scaler funcs
        scaler, inv_scaler = load_scaler_from_json(SCALER_PATH)

        # Now call your simulation/imputation using model, scaler, inv_scaler
        # sim.run_simulation_with_live_imputation(train_df, missing_df, pf, model, scaler, inv_scaler, device, seq_len=24)

        # --- run (note: use window_hours=24 NOT 'int=24') ---


        run_simulation_with_live_imputation(
            sim_df=tr,  # dataframe you play through (with NaNs)
            missing_df=df,  # this will be UPDATED in-place as we impute
            positions=pf,  # positions dict/df
            model=model,
            scaler=scaler,
            inv_scaler=inv_scaler,
            device=device,
            graph_placeholder=graph_placeholder,
            sliding_chart_placeholder=sliding_chart_placeholder,
            gauge_placeholder=gauge_placeholder,
            window_hours=24  # the history length for the predictor
        )
if __name__ == '__main__':
    main()