import streamlit as st

# ----------------------------
# Buttons Management
# ----------------------------
def add_buttons():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🧠 Start training", use_container_width=True):
            st.session_state.training = True
    with col2:
        if st.button("▶️ Start Simulation", use_container_width=True):
            st.session_state.running = True
    with col3:
        if st.button("⏹ Stop Simulation", use_container_width=True):
            st.session_state.running = False

