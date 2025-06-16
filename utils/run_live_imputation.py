# run_live_imputation.py

import os
import json
import time
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from IPython.display import display


# Geographical Helper Functions
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def create_adjacency_matrix(latlng_df, sigma_sq_ratio=0.1):
    num_sensors = len(latlng_df)
    dist_matrix = np.zeros((num_sensors, num_sensors))
    for i in range(num_sensors):
        for j in range(i, num_sensors):
            lat1, lon1 = latlng_df.iloc[i]['latitude'], latlng_df.iloc[i]['longitude']
            lat2, lon2 = latlng_df.iloc[j]['latitude'], latlng_df.iloc[j]['longitude']
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    sigma_sq = dist_matrix.std()**2 * sigma_sq_ratio
    adj_matrix = np.exp(-dist_matrix**2 / sigma_sq)
    np.fill_diagonal(adj_matrix, 1)

    D = np.diag(np.sum(adj_matrix, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    adj_matrix_normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    return torch.FloatTensor(adj_matrix_normalized)

#  Model Architecture Definitions
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        aggregated_features = torch.spmm(adj, x.t()).t()
        output = torch.mm(aggregated_features, self.weight)
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
        batch_size, seq_len, num_nodes = x.shape
        gcn_outputs = []
        for t in range(seq_len):
            gcn_out = self.relu(self.gcn(x[:, t, :], self.adj))
            gcn_outputs.append(gcn_out.unsqueeze(1))

        gcn_sequence = torch.cat(gcn_outputs, dim=1)
        lstm_out, _ = self.lstm(gcn_sequence)
        output = self.fc(lstm_out)
        return output

#LIVE IMPUTATION AND VISUALIZATION FUNCTIONS

def run_live_imputation(
    model_path,
    missing_csv_path,
    latlng_path,
    scaler_params_path,
    output_data_path="pm25_imputed_live.csv",
    output_mask_path="pm25_imputation_mask.csv",
    simulation_speed=0.01,
    simulation_limit=None
):
    """
    Runs a standalone live imputation simulation.
    """
    print("--- Starting Standalone Live Imputation ---")

    # --- Step 1: Load all prerequisites from files ---
    if not all(os.path.exists(p) for p in [model_path, missing_csv_path, latlng_path, scaler_params_path]):
        raise FileNotFoundError("One or more required input files are missing.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load coordinates and determine sensor order
    latlng_df = pd.read_csv(latlng_path)
    sensor_cols = latlng_df['sensor_id'].astype(str).tolist()

    # Load scaler parameters and create scaler functions
    with open(scaler_params_path, 'r') as f:
        scaler_params = json.load(f)
    min_val, max_val = scaler_params['min_val'], scaler_params['max_val']
    scaler = lambda x: (x - min_val) / (max_val - min_val)
    inv_scaler = lambda x: x * (max_val - min_val) + min_val

    # Initialization of the Model and Adjacency Matrix ---
    adj_matrix = create_adjacency_matrix(latlng_df)

    # These parameters must match the model that was trained
    model_config = {'NUM_NODES': len(sensor_cols), 'GCN_HIDDEN': 64, 'LSTM_HIDDEN': 64, 'EVAL_LENGTH': 36}

    model = GCNLSTMImputer(
        adj=adj_matrix.to(device),
        num_nodes=model_config['NUM_NODES'],
        in_features=model_config['NUM_NODES'],
        gcn_hidden=model_config['GCN_HIDDEN'],
        lstm_hidden=model_config['LSTM_HIDDEN'],
        out_features=model_config['NUM_NODES']
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model and prerequisites loaded successfully.")

    # Prepare Data and Output Files ---
    original_missing_df = pd.read_csv(missing_csv_path, index_col='datetime', parse_dates=True)
    # Remove leading zeros from all column names
    original_missing_df.columns = [col.lstrip('0') if col.isdigit() else col for col in original_missing_df.columns]

    #  Fix column order
    original_missing_df = original_missing_df[sensor_cols]

    if simulation_limit is None:
        simulation_limit = len(original_missing_df)

    imputed_df_live = original_missing_df.copy().fillna(method='ffill').fillna(method='bfill')
    imputation_mask = pd.DataFrame(False, index=imputed_df_live.index, columns=imputed_df_live.columns)

    # Initialize output files with headers
    imputed_df_live.head(0).to_csv(output_data_path)
    imputation_mask.head(0).to_csv(output_mask_path)
    print(f"Output files will be saved to:\n -> {output_data_path}\n -> {output_mask_path}")

    # --- Step 4: Run Simulation and Incremental Save ---
    for t in tqdm(range(simulation_limit), desc="Live Imputation Progress"):
        # Check if imputation is needed and possible
        if t >= model_config['EVAL_LENGTH'] and original_missing_df.iloc[t].isnull().any():
            history_df = imputed_df_live.iloc[t - model_config['EVAL_LENGTH'] : t]
            history_normalized = scaler(history_df.to_numpy())
            input_tensor = torch.FloatTensor(history_normalized).unsqueeze(0).to(device)

            with torch.no_grad():
                output_sequence = model(input_tensor)

            last_step_prediction = inv_scaler(output_sequence[0, -1, :].cpu().numpy())

            missing_cols_indices = np.where(original_missing_df.iloc[t].isnull())[0]
            for col_idx in missing_cols_indices:
                imputed_df_live.iat[t, col_idx] = last_step_prediction[col_idx]
                imputation_mask.iat[t, col_idx] = True

        # Append the current row (original or imputed) to the files
        imputed_df_live.iloc[[t]].to_csv(output_data_path, mode='a', header=False)
        imputation_mask.iloc[[t]].to_csv(output_mask_path, mode='a', header=False)

        time.sleep(simulation_speed)

    print("\nSimulation finished.")
    return output_data_path, output_mask_path


def visualize_imputation_from_files(data_path, mask_path, rows_to_display=150):
    """
    Reads imputation results from files and displays a styled DataFrame.
    This version correctly applies styles to the entire frame at once.
    """
    print(f"Reading data from '{data_path}' and mask from '{mask_path}'...")
    try:
        imputed_df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        imputation_mask = pd.read_csv(mask_path, index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: Files not found. Please run the imputation function first.")
        return

    print("Styling the output DataFrame. Imputed values are highlighted in red.")


    # Slice both the data and the mask to the desired display size
    display_df = imputed_df.head(rows_to_display).copy().round(2)
    display_mask = imputation_mask.head(rows_to_display)

    # Create a new DataFrame
    style_df = pd.DataFrame('', index=display_df.index, columns=display_df.columns)

    # Use the boolean `display_mask` to set the style string only for cells

    style_df[display_mask] = 'background-color: lightcoral; color: white'

    # Apply the DataFrame of styles to the data DataFrame.

    styled_output = display_df.style.apply(lambda x: style_df, axis=None)



    display(styled_output)

    display_df = imputed_df.head(rows_to_display).copy().round(2)
    styler = display_df.style.apply(lambda v: style_cells(v, v.name[0], v.name[1], imputation_mask), axis=None)
    display(styler)

# EXECUTION

if __name__ == '__main__':
    # Configuration, paths to imputs files

    MODEL_FILE = './models/gcn_lstm_imputer.pth'
    MISSING_DATA_FILE = './pm25/SampleData/pm25_missing.txt'
    LATLNG_FILE = './pm25/SampleData/pm25_latlng.txt'
    SCALER_PARAMS_FILE = './models/scaler_params.json'

    #  Run the Imputation ---This will generate the output CSV files.
    final_data, final_mask = run_live_imputation(
        model_path=MODEL_FILE,
        missing_csv_path=MISSING_DATA_FILE,
        latlng_path=LATLNG_FILE,
        scaler_params_path=SCALER_PARAMS_FILE,
        simulation_limit=200,  # Set to None to run on the whole file
        simulation_speed=1  # Set to 0 for max speed
    )

    #  Visualize the Results (Optional) ---
    # This function is called after the simulation is complete.
    """print("\n--- Visualizing Results ---")
    visualize_imputation_from_files(
        data_path=final_data, 
        mask_path=final_mask,
        rows_to_display=200
    )"""
