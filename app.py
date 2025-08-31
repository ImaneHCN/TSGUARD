# app.py
# -----------------------------------------------------------------------------
# TSGuard – Streamlit front-end (refined header + popover about + world demo map)
# - Uploads in sidebar (centered vertically)
# - First view: TSGuard header + ❓ popover + blue alert + worldwide pydeck sensors
# - Auto-hide sidebar when 3 datasets loaded
# - Neutral action buttons (like Browse files)
# - Fixed footer logos
# -----------------------------------------------------------------------------

from pathlib import Path
import os
import json
import torch
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.io as pio

# --- project modules (unchanged) ---------------------------------------------
import helpers as helper
import utils.visualization as vz
import components.sidebar as sidebar
import components.settings as settings
import components.buttons as buttons
import components.containers as containers
from utils.config import DEFAULT_VALUES
from models.simulation import (
    run_simulation_with_live_imputation,
    train_model,
    run_tsguard_vs_pristi_comparison
)

# Plotly default theme
pio.templates.default = "plotly_white"

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="📡 TSGuard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Global CSS (professional + your tweaks)
# -----------------------------------------------------------------------------
# ---- About TSGuard (Markdown content) ---------------------------------------
ABOUT_TS_GUARD_MD = """
### About TSGuard

**TSGuard** is a lightweight, real-time system for detecting and imputing missing values in streaming environmental time series.

**What it does**
- Monitors large networks of sensors and repairs delayed/missing readings on the fly.
- Preserves physical plausibility through **domain range constraints** applied to every imputed value.
- Keeps the interface responsive so operators can see the state of the network as it evolves.

**How it works (under the hood)**
- **Spatial structure** is captured with a **Graph Neural Network (GNN)** that relates each station to its neighbors.
- **Temporal continuity** is captured with an **LSTM**, modeling smooth changes over time.
- The hybrid GNN+LSTM design lets TSGuard impute values for a sensor using both nearby stations **and** recent history.
- A simple validator enforces **domain-specific bounds** (e.g., acceptable ranges for air-quality metrics) before values are displayed.

**What you’re seeing in the demo**
- **Live Map (left)**: each station updates in real time — **green = real** reading, **red = imputed** reading at that timestamp.
- **Gauge (right of map)**: **cumulative percentage of originally-missing data** seen so far on the displayed sensors.
- **Global Time Series (below)**: full history of the displayed sensors; red segments highlight intervals containing imputed points.
- **Snapshot (last 10 timestamps)**: a compact, synchronized view of the most recent window for quick trend checks.
- **Active / Delayed counters**: numbers shown next to the gauge reflect, at the *current timestamp*, how many sensors were observed vs. missing originally.

**Comparison mode (TSGuard vs PriSTI)**
- PriSTI is a strong baseline model but **heavier to compute**; the comparison page runs it separately.
- PriSTI can alter some observed values in its internal reconstructions.  
  In this demo, we show PriSTI **only on cells that were originally missing**, and we keep the **original values** elsewhere — so comparisons are fair.
- The comparison uses **non-overlapping 36-step windows**; the **first window needs to fill** before plots appear.
- Sliders let you control how many **sensors** and **timestamps** to display in the side-by-side charts.

**Why it matters**
- Environmental networks face **delays, outages, and sensor drift**. TSGuard recovers missing values quickly while respecting domain limits.
- The approach is **online** and **lightweight**, making it suitable for operational dashboards with minimal latency.

**Dataset in the demo**
- We illustrate the system on **AQI-36**, a network of 36 air-quality stations, to show real-time recovery and plausibility monitoring.

**Notes & limitations**
- Imputation quality depends on **graph topology**, **window length**, and **data coverage**.
- Until the first comparison window is ready, the PriSTI panel shows a warm-up message.
- If a GPU isn’t available, PriSTI may take longer to compute; TSGuard stays responsive.

If you reference this demo, please cite our **TSGuard demo paper**.
"""

st.markdown("""
<style>
/* App background + space for fixed footer */
.stApp{ background:#F3F6FD !important; padding-bottom:140px !important; }

/* Sidebar background (swapped colors) */
[data-testid="stSidebar"]{ background:#EAF2FF !important; }
[data-testid="stSidebar"] > div{ height:100%; } /* full height container */

/* Center the sidebar content vertically (and fallback on short screens) */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  height:100%;
  display:flex; flex-direction:column; justify-content:center;
}
@media (max-height: 760px){
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
    display:block; height:auto;
  }
}

/* Typo & links */
html, body, .stApp, [data-testid="stMarkdownContainer"], label, p, li, span {
  color:#112233 !important;
}
a { color:#0B5FFF; text-decoration:none; } 
a:hover{ text-decoration:underline; }

/* -------- Header row alignment -------- */
.ts-hero{
  display:flex; align-items:center; gap:10px; margin:4px 0 6px;
}
.ts-hero .icon{ font-size:34px; line-height:1; }
.ts-hero .title{
  margin:0; font-weight:800; letter-spacing:.2px;
  font-size: clamp(24px, 2.1vw, 34px); color:#0B1F33;
}
/* underline slightly lower than the row */
.ts-rule{
  height:3px; width:100%; margin-top:12px; border-radius:3px;
  background: linear-gradient(90deg,#0B5FFF,#00A3FF); opacity:.9;
}

/* Inline popover for About (no JS, just <details>) */
.ts-about{
  position: relative;
  display: inline-block;
  margin-left: 8px;
}
.ts-about > summary{
  list-style: none;
  cursor: pointer;
  font-size: 20px;
  line-height: 1;
  color: #0B5FFF;             /* same blue as underline */
  display: inline-flex;
  align-items: center;
}
.ts-about > summary::-webkit-details-marker{ display:none; } /* Safari/Chrome hide marker */
.ts-about .about-note{
  display: none;
  position: absolute;
  top: 28px; left: 0;
  width: min(360px, 90vw);
  background: #ffffff;
  border: 1.5px solid #0B5FFF;
  border-radius: 10px;
  padding: 12px 14px;
  color: #0B1F33;
  box-shadow: 0 4px 12px rgba(0,0,0,.15);
  z-index: 9999;
}
.ts-about[open] .about-note{ display:block; }

/* Blue alert (upload required) – text color = same blue as border */
.ts-alert{
  background:#EAF2FF;           /* light blue */
  border:2px solid #0B5FFF;     /* dark blue border */
  color:#0B5FFF;                /* text same blue */
  border-radius:10px;
  padding:12px 14px;
  font-weight:600;
  margin:12px 0 10px 0;
  opacity:1;
}

/* Action buttons: neutral, like Browse files */
#action-bar .stButton > button{
  background:#fff !important; color:#0B1F33 !important;
  border:1px solid #D0D7E2 !important; border-radius:10px !important;
  font-weight:600 !important; padding:0.55rem 1rem !important;
  box-shadow:0 1px 2px rgba(0,0,0,.04) !important;
}
#action-bar .stButton > button:hover{
  background:#F6F8FC !important; box-shadow:0 2px 6px rgba(0,0,0,.06) !important;
}
#action-bar .stButton > button:focus:not(:active){
  box-shadow:0 0 0 3px rgba(11,95,255,.18) inset !important;
}
#action-bar .stButton > button:disabled{ opacity:.55 !important; cursor:not-allowed; }

/* Fixed footer (logos), always on top */
.ts-footer{
  position:fixed; left:0; right:0; bottom:0; width:100%;
  background:#fff; border-top:1px solid rgba(27,99,161,.15);
  box-shadow:0 -2px 10px rgba(18,61,101,.06);
  z-index:2147483647; pointer-events:none;
}
.ts-footer-inner{ padding:10px 16px; }
.ts-footer-row{ display:flex; align-items:center; justify-content:space-between; min-height:68px; }
.ts-footer-cell{ display:flex; align-items:center; justify-content:center; }
.ts-footer-img{ height:64px; width:auto; object-fit:contain; display:block; pointer-events:none; }
.ts-footer-cell--ul  .ts-footer-img{ transform: scale(1.10); } /* small balancing */
.ts-footer-cell--esi .ts-footer-img{ transform: scale(0.94); }

@media (max-width: 900px){
  .ts-footer-img{ height:52px; }
  .stApp{ padding-bottom:115px !important; }
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  padding-top: 60px !important;   /* ajuste la valeur selon la hauteur désirée */
}
/* Sidebar structure */
[data-testid="stSidebar"] > div:first-child {
    display: flex;
    flex-direction: column;
    height: 100%;
}

/* Logos grid at bottom */
.sidebar-logos {
    margin-top: auto;
    padding: 20px 0;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 18px;
    justify-items: center;
    border-top: 1px solid #ddd;
}

/* Logo container */
.sidebar-logos a {
    position: relative;
    display: inline-block;
}

/* Logo image */
.sidebar-logos img {
    max-height: 70px;
    cursor: pointer;
}

/* Tooltip text */
.sidebar-logos a::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 110%; /* above the logo */
    left: 50%;
    transform: translateX(-50%);
    background: #0B1F33;
    color: #fff;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
    z-index: 9999;
}

/* Show tooltip on hover */
.sidebar-logos a:hover::after {
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header (TSGuard + ❓ popover)
# -----------------------------------------------------------------------------
def render_header():
    st.markdown("""
    <div class="ts-hero">
      <div class="icon">📡</div>
      <div>
        <div style="display:flex; align-items:center; gap:8px;">
          <h1 class="title" style="margin:0;">TSGuard</h1>
          <details class="ts-about">
            <summary title="About TSGuard">❓</summary>
            <div class="about-note">
              <b>About TSGuard (quick)</b><br/>
              Real-time detection and imputation of missing values in streaming environmental data, driven by a hybrid GNN+LSTM model with domain range checks.
              <br/><br/>
              <i>Open the “About TSGuard” section below for the full description.</i>
            </div>
          </details>
        </div>
        <div class="ts-rule"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Full "About" content right under the header
    with st.expander("About TSGuard", expanded=False):
        st.markdown(ABOUT_TS_GUARD_MD)


# -----------------------------------------------------------------------------
# First view (before uploads): header + alert + worldwide demo sensors
# -----------------------------------------------------------------------------
def render_first_view():
    render_header()

    # Blue alert right under the header
    st.markdown(
        "<div class='ts-alert'>Please upload training data, sensor data, and sensor positions to continue.</div>",
        unsafe_allow_html=True
    )

    # Worldwide demo sensors (pydeck)
    NUM_SENSORS = 1320
    rng = np.random.default_rng(12)
    regions = [
        ("North America",  39.0,  -98.0, 0.18),
        ("South America", -15.0,  -60.0, 0.10),
        ("Europe",         50.0,   10.0, 0.20),
        ("North Africa",   28.0,   10.0, 0.06),
        ("Sub-Saharan",     0.0,   20.0, 0.08),
        ("Middle East",    28.0,   45.0, 0.07),
        ("South Asia",     22.0,   78.0, 0.12),
        ("East Asia",      35.0,  105.0, 0.12),
        ("Southeast Asia", 10.0,  105.0, 0.05),
        ("Oceania",       -25.0,  134.0, 0.05),
        ("Japan/Korea",    36.0,  135.0, 0.04),
        ("Polar",          70.0,    0.0, 0.03),
    ]
    rows = []
    for name, lat, lon, weight in regions:
        n = max(1, int(NUM_SENSORS * weight))
        sig_lat = 7 + 6 * rng.random()
        sig_lon = 10 + 10 * rng.random()
        lats = lat + rng.normal(0, sig_lat, n)
        lons = lon + rng.normal(0, sig_lon, n)
        rows.append(pd.DataFrame({"lat": lats, "lon": lons, "region": name}))
    world_points = pd.concat(rows, ignore_index=True)
    world_points["lat"] = world_points["lat"].clip(-85, 85)
    world_points["lon"] = ((world_points["lon"] + 180) % 360) - 180

    st.caption("🌍 Prototype Visualization of Global Sensors")
    # --- Vue MONDE unique, sans répétition horizontale -----------------------
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=world_points,
        get_position="[lon, lat]",
        get_radius=35000,                # taille des points (ajuste si besoin)
        get_fill_color=[230, 65, 30, 160],
        pickable=False,
        radius_min_pixels=2,
        radius_max_pixels=24,
    )

    # Vue monde centrée, sans wrap
    view_state = pdk.ViewState(
        latitude=20, longitude=0, zoom=0.8, min_zoom=0.4, max_zoom=5, pitch=0, bearing=0
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        views=[pdk.View("MapView", repeat=False)],  # <- empêche la répétition des continents
        map_style=None,                              # garde un fond neutre (pas besoin de token)
        height=560                                   # hauteur de la carte (augmente si tu veux)
    )

    st.pydeck_chart(deck, use_container_width=True)


# -----------------------------------------------------------------------------
# Footer (logos)
# -----------------------------------------------------------------------------
LOGO_UL = "images/UNI logo.jpeg"
LOGO_ESI = "images/ESI_Logo.png"
LOGO_CH = "images/Logo-UHBC.png"
LOGO_UP = "images/Logo_UniParis.png"

import base64, mimetypes
def _img_to_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def render_sidebar_logos():
    def safe(p):
        try:
            return _img_to_data_uri(p)
        except Exception:
            return ""

    ul_b64  = safe(LOGO_UL)
    esi_b64 = safe(LOGO_ESI)
    ch_b64  = safe(LOGO_CH)
    up_b64  = safe(LOGO_UP)

    logos_html = '<div class="sidebar-logos">'
    if ul_b64:
        logos_html += f'<a href="https://www.univ-lorraine.fr" target="_blank" data-tooltip="Université de Lorraine"><img src="{ul_b64}" /></a>'
    if esi_b64:
        logos_html += f'<a href="https://www.esi.dz" target="_blank" data-tooltip="École Nationale Supérieure d’Informatique (ESI)"><img src="{esi_b64}" /></a>'
    if ch_b64:
        logos_html += f'<a href="https://www.univ-chlef.dz" target="_blank" data-tooltip="Université Hassiba Benbouali de Chlef (UHBC)"><img src="{ch_b64}" /></a>'
    if up_b64:
        logos_html += f'<a href="https://u-paris.fr" target="_blank" data-tooltip="Université de Paris"><img src="{up_b64}" /></a>'
    logos_html += "</div>"

    st.sidebar.markdown(logos_html, unsafe_allow_html=True)
# ----------------------------
# -----------------------------------------------------------------------------
# Auto-hide the sidebar once files are ready
# -----------------------------------------------------------------------------
def hide_sidebar_when_ready(files_ready: bool):
    if files_ready:
        st.markdown("""
        <style>
          [data-testid="stSidebar"]{
            position: fixed !important;
            left: -24rem !important; top:0; bottom:0;
            transition: left .25s ease-in-out;
          }
          .main .block-container{ padding-left: 2rem !important; }
        </style>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utilities (unchanged)
# -----------------------------------------------------------------------------
def resolve_paths(model_path: str = None, scaler_path: str = None):
    if model_path is None:
        model_path = "generated/model_TSGuard.pth"
    if scaler_path is None:
        p = Path(model_path)
        scaler_path = str(p.with_name(p.stem + "_scaler.json"))
    return str(Path(model_path).expanduser().resolve()), str(Path(scaler_path).expanduser().resolve())

def load_scaler_from_json(scaler_json_path: str):
    with open(scaler_json_path, "r") as f:
        params = json.load(f)
    min_val = float(params["min_val"]); max_val = float(params["max_val"])
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    def scaler(x):     return (x - min_val) / denom
    def inv_scaler(x): return x * denom + min_val
    return scaler, inv_scaler

def load_state_dict_into(model: torch.nn.Module, model_path: str, device: torch.device):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state); model.eval(); return model

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main():
    # Uploaders in the sidebar (your existing component)
    training_data_file, sensor_data_file, positions_file = sidebar.init()

    files_ready = bool(training_data_file and sensor_data_file and positions_file)

    if not files_ready:
        render_first_view()
        render_sidebar_logos()
        return

    # Once ready, hide the sidebar and show the main UI
    hide_sidebar_when_ready(files_ready)

    render_header()

    # Settings (open by default: set expanded=True in your settings component)
    settings.add_setting_panel()

    # Wrap action buttons so our CSS targets them (#action-bar)
    st.markdown('<div id="action-bar">', unsafe_allow_html=True)
    buttons.add_buttons()
    # st.write({"page": st.session_state.get("page"), "running": st.session_state.get("running")})
    st.markdown('</div>', unsafe_allow_html=True)

    # Init data
    tr, df, pf, sensor_list = helper.init_files(training_data_file, sensor_data_file, positions_file)

    # Clean ground truth
    gref = tr.copy()
    if "datetime" not in gref.columns:
        gref = gref.reset_index().rename(columns={"index": "datetime"})
    gref["datetime"] = pd.to_datetime(gref["datetime"], errors="coerce").dt.floor("h")
    gref = gref.dropna(subset=["datetime"]).set_index("datetime")
    gref.columns = [c if not str(c).isdigit() else str(c).zfill(6) for c in gref.columns]
    st.session_state["ground_ref"] = gref

    helper.init_states(tr, df, pf)
    graph_ph, gauge_ph, sliding_ph = containers.init_containers()

    # Device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = None

    # Training
    if st.session_state.training:
        st.title("⏳ Training is running... Please wait.")
        with st.spinner("Training GNN model..."):
            model_path = DEFAULT_VALUES["training_file_path"]
            model = train_model(tr, df, pf, 10, model_path=model_path)
            st.success(f"✅ Training completed. Model saved to '{model_path}'")
            st.session_state.training = False

    # --- Router (comparison must NOT depend on `running`) ---
    page = st.session_state.get("page", "sim")

    # Device + scaler (prépare une fois)
    MODEL_PATH, SCALER_PATH = resolve_paths(
        model_path="generated/model_TSGuard.pth",
        scaler_path="generated/model_TSGuard_scaler.json",
    )
    if os.path.exists(SCALER_PATH):
        scaler, inv_scaler = load_scaler_from_json(SCALER_PATH)
    else:
        scaler = lambda x: x
        inv_scaler = lambda x: x

    if page == "cmp":
        # Page comparaison — indépendante de `running`
        st.info("📊 Comparison page (TSGuard vs PriSTI).")
        run_tsguard_vs_pristi_comparison(
            sim_df=tr, missing_df=df, model=model, scaler=scaler, inv_scaler=inv_scaler, device=device
        )

    else:  # page == "sim"
        if st.session_state.get("running", False):
            st.success("✅ Simulation is running. Click '📊 TSGuard vs PTISTI' to open the comparison page.")
            run_simulation_with_live_imputation(
                sim_df=tr,
                missing_df=df,
                positions=pf,
                model=model,
                scaler=scaler,
                inv_scaler=inv_scaler,
                device=device,
                graph_placeholder=graph_ph,
                sliding_chart_placeholder=sliding_ph,
                gauge_placeholder=gauge_ph,
                window_hours=24,
            )
        else:
            st.info(
                "Choose an action: 🧠 **Start TSGuard training** to train the model, ▶️ **Start TSGuard Simulation** to start the simulation, or 📊 **TSGuard vs PriSTI** to open the comparison."

            )

    render_sidebar_logos()

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
