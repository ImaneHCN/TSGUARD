import streamlit as st
from utils.config import DEFAULT_VALUES

# ----------------------------
# Setting Management
# ----------------------------
def add_setting_panel():
    with st.expander("⚙️ Settings", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["📌 Constraints", "📈 Threshold", "📊 Missing values", "🕸️ Graph Options"])
        
        with tab1:
            add_constraints_panel()
            st.json(st.session_state.get('constraints', []))
        
        with tab2:
            add_threshold_panel()
        
        with tab3:
            add_missing_value_panel()
        
        with tab4:
            add_graph_opt_panel()

# ----------------------------
# Constraints Management
# ----------------------------
def add_constraints_panel():
    if 'constraints' not in st.session_state:
        st.session_state['constraints'] = []
    ctype = st.radio("Select Constraint Type", options=["📍 Spatial", "⏳ Temporal"], key="constraint_type")
    if "Spatial" in ctype:
        st.markdown("#### 📍 Spatial Constraints")
        # Distance with unit selection
        col1, col2 = st.columns([2, 1])
        with col1:
            spatial_distance = st.number_input("📏 Distance Threshold", value=2.0, step=0.1, key="spatial_distance")
        with col2:
            distance_unit = st.selectbox("Unit", ["km", "miles"], key="distance_unit")

        # Convert miles to km for standardization 
        spatial_distance_km = 0
        spatial_distance_miles = 0
        if distance_unit == "miles":
            spatial_distance_km = round(spatial_distance * 1.60934, 2)  # 1 mile = 1.60934 km
            spatial_distance_miles = spatial_distance
        else:
            spatial_distance_km = spatial_distance
            spatial_distance_miles = round(spatial_distance / 1.60934, 2)  # 1 mile = 1.60934 km

        spatial_diff = st.number_input("📊 Max Sensor Difference", value=5.0, step=0.1, key="spatial_diff")
        if st.button("Add Spatial Constraint", key="add_spatial"):
            st.session_state['constraints'].append({"type": "Spatial", "distance in km": spatial_distance_km, "distance in miles": spatial_distance_miles, "diff": spatial_diff})
            st.success("Spatial constraint added.")
    else:
        st.markdown("#### ⏳ Temporal Constraints")
        month = st.selectbox("🌦️ Month", options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], key="month")
        constraint_option = st.selectbox("📉 Constraint Option", options=["Greater than", "Less than"], key="constraint_option")
        temp_threshold = st.number_input("📈 Threshold Value", value=50.0, step=0.1, key="temp_threshold")
        if st.button("Add Temporal Constraint", key="add_temporal"):
            st.session_state['constraints'].append({"type": "Temporal", "month": month, "option": constraint_option, "temp_threshold": temp_threshold})
            st.success("Temporal constraint added.")
# ----------------------------
# Missing value Management
# ----------------------------
def add_missing_value_panel():
    if 'missing_value_thresholds' not in st.session_state:
        st.session_state['missing_value_thresholds'] = []
    
    st.markdown("### 🛠 Define Missing Value Thresholds")
    st.markdown("Please specify the missing value percentage ranges for different risk states (Green: Low, Yellow: Medium, Red: High).")
    
    col1, col2 = st.columns(2)    
    with col1:
        green_min = st.number_input("🟢 Green Min", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_green_min"], step=1)
        yellow_min = st.number_input("🟡 Yellow Min", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_yellow_min"], step=1)
        red_min = st.number_input("🔴 Red Min", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_red_min"], step=1)
        
    with col2:
        green_max = st.number_input("🟢 Green Max", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_green_max"], step=1)
        yellow_max = st.number_input("🟡 Yellow Max", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_yellow_max"], step=1)
        red_max = st.number_input("🔴 Red Max", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_red_max"], step=1)
        
    if st.button("✅ Save Thresholds"):
        if not (green_min <= green_max <= yellow_min <= yellow_max <= red_min <= red_max):
            st.error("🚨 Invalid threshold ranges. Ensure consistency between min/max values.")
        else:
            st.session_state['missing_value_thresholds'] = {
                "Green": (green_min, green_max),
                "Yellow": (yellow_min, yellow_max),
                "Red": (red_min, red_max)
            }
            st.success("✅ Missing value thresholds saved successfully.")
        

# ----------------------------
# Threshold Management
# ----------------------------
def add_threshold_panel():
    if 'sigma_threshold' not in st.session_state:
        st.session_state['sigma_threshold'] = DEFAULT_VALUES["sigma_threshold"]
    
    st.markdown("Please specify the allowed delay threshold before a sensor is considered as having a missing value.")
    st.markdown("The default value is **" + str(DEFAULT_VALUES["sigma_threshold"]) + " minutes**.")
    col1, col2 = st.columns([2, 1])
    with col1:
        threshold = st.number_input("📈 Threshold Value Delay", value=DEFAULT_VALUES["sigma_threshold"], step=1, key="threshold")
    with col2:
        time_unit = st.selectbox("Unit", ["minutes", "hours"], key="time_unit")
    if st.button("Set the delay threshold", key="set_sigma_threshold"):
        st.session_state['sigma_threshold'] = threshold
        st.success("Delay 'Sigma' threshold set to : **"+ str(threshold)+ " "+ time_unit+"**.")

# ----------------------------
# Graph Management
# ----------------------------
def add_graph_opt_panel():
    if 'graph_size' not in st.session_state:
        st.session_state['graph_size'] = DEFAULT_VALUES["graph_size"]
    
    st.markdown("### Configure Graph Size")  
    st.markdown("Specify the number of sensors (nodes) in the graph.")  
    st.markdown(f"**Default:** {DEFAULT_VALUES['graph_size']} sensors")

    g_size = st.number_input("📶 Graph Size", value=DEFAULT_VALUES["graph_size"], step=1, key="g_size")

    if st.button("Save", key="set_graph_size"):
        st.session_state['graph_size'] = g_size
        st.success("The graph size set to : **"+ str(g_size)+" sensors**.")
