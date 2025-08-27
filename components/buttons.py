import streamlit as st

# ----------------------------
# Buttons Management
# ----------------------------
def add_buttons():
    if "page" not in st.session_state:
        st.session_state.page = "sim"   # "sim" or "cmp"

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🧠 Start training", use_container_width=True):
            st.session_state.training = True

    with col2:
        if st.button("▶️ Start TSGuard Simulation", use_container_width=True):
            st.session_state.page = "sim"
            st.session_state.running = True

    with col3:
        # keep your exact label request (“PTISTI”)
        if st.button("📊 TSGuard vs PTISTI", use_container_width=True):
            st.session_state.page = "cmp"
            st.session_state.running = False   # stop the sim when going to comparison