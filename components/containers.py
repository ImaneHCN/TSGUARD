import streamlit as st

# ----------------------------
# Containers Management
# ----------------------------
def init_containers():
    with st.container():
        graph_placeholder = st.empty()
        gauge_placeholder = st.empty()
        sliding_chart_placeholder = st.empty()
    return graph_placeholder, gauge_placeholder, sliding_chart_placeholder
