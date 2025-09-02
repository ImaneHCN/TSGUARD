import streamlit as st
import pandas as pd
import base64
import numpy as np

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
def load_sensor_data(path: str) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(path)

    # Normalize datetime column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "time" in df.columns:
        df = df.rename(columns={"time": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "datetime"})
    else:
        raise KeyError("No datetime-like column found in the file.")

    # Drop rows without valid datetime
    df = df.dropna(subset=["datetime"]).copy()

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

def _id_norm(s: object) -> str:
    """Keep 'datetime' as is; zero-pad pure digits to 6 chars (e.g., '123'->'000123')."""
    s = str(s).strip()
    if s.lower() == "datetime":
        return "datetime"
    return s.zfill(6) if s.isdigit() else s

def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _id_norm(c) for c in df.columns})

def normalize_positions_df(pos) -> pd.DataFrame:
    """
    Normalize positions into a DataFrame with columns: sensor_id, latitude, longitude.
    Accepts dict, DataFrame, or path-like.
    """

    import pandas as pd

    # --- case: dict {idx: (lon, lat)} ---
    if isinstance(pos, dict):
        pos = (
            pd.DataFrame.from_dict(pos, orient="index", columns=["longitude", "latitude"])
            .rename_axis("sensor_id")
            .reset_index()
        )

    # --- case: already DataFrame ---
    elif isinstance(pos, pd.DataFrame):
        pos = pos.copy()

    else:
        raise TypeError(f"Unsupported type for positions: {type(pos)}")

    # --- normalize column names ---
    pos = pos.rename(columns={"lat": "latitude", "lng": "longitude", "lon": "longitude"}).copy()
    if "sensor_id" not in pos.columns:
        pos = pos.reset_index().rename(columns={"index": "sensor_id"})

    # --- clean types ---
    pos["sensor_id"] = pos["sensor_id"].astype(str).str.strip()
    pos["sensor_id"] = pos["sensor_id"].apply(lambda x: x.zfill(6) if x.isdigit() else x)

    pos["latitude"] = pd.to_numeric(pos["latitude"], errors="coerce")
    pos["longitude"] = pd.to_numeric(pos["longitude"], errors="coerce")

    return pos[["sensor_id", "latitude", "longitude"]]

def remap_positions_if_indexed_0_to_n_minus_1(pos: pd.DataFrame, sensor_cols_in_data: list[str]) -> pd.DataFrame:
    """
    If positions have ids like '0','1',...,'N-1', replace them by real sensor ids
    from the *data* (exact order). Otherwise, just return normalized positions.
    """
    sid = pos["sensor_id"]
    if sid.str.fullmatch(r"\d+").all():
        nums = sid.astype(int)
        if nums.min() == 0 and nums.max() == len(pos) - 1:
            # reorder by numeric id then map to the data’s first len(pos) sensor ids
            pos = pos.iloc[np.argsort(nums)].reset_index(drop=True).copy()
            wanted = [c for c in sensor_cols_in_data][:len(pos)]
            pos["sensor_id"] = wanted
    return pos

def init_files(training_data_file, sensor_data_file, positions_file):
    tr = load_training_data(training_data_file)
    df = load_sensor_data(sensor_data_file)
    pf = load_positions_data(positions_file)
    # Normalize training/simulation dfs
    ground_df = normalize_df_columns(tr)  # or train_file
    missing_df = normalize_df_columns(df)  # or sim_file

    # Positions
    positions_df = normalize_positions_df(pf)  # your positions source as DF
    # The sensor columns used in the data (order matters!)
    sensor_cols_in_data = [c for c in ground_df.columns if c != "datetime"]

    # If positions look like 0..N-1, remap them to the real ids by order:
    positions_df = remap_positions_if_indexed_0_to_n_minus_1(positions_df, sensor_cols_in_data)

    # Now, verify & align 1:1
    pos_ids = positions_df["sensor_id"].tolist()
    data_ids = sensor_cols_in_data[:len(pos_ids)]

    missing_in_pos = sorted(set(data_ids) - set(pos_ids))
    extra_in_pos = sorted(set(pos_ids) - set(data_ids))

    if missing_in_pos or extra_in_pos:
        raise KeyError(
            "Sensor ID mismatch after normalization.\n"
            f"- Missing in positions: {missing_in_pos}\n"
            f"- Unexpected in positions: {extra_in_pos}\n"
            f"- Example data ids: {data_ids[:8]}\n"
            f"- Example pos ids: {pos_ids[:8]}"
        )

    # Reindex dataframes to EXACT positions order, so mapping is stable
    ordered_ids = positions_df["sensor_id"].tolist()
    # NEW (preserves 'datetime' if present)
    cols_ground = (['datetime'] + ordered_ids) if 'datetime' in ground_df.columns else ordered_ids
    cols_missing = (['datetime'] + ordered_ids) if 'datetime' in missing_df.columns else ordered_ids
    ground_df = ground_df[cols_ground]
    missing_df = missing_df[cols_missing]
    return ground_df, missing_df, pf, ordered_ids


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