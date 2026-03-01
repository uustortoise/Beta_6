import streamlit as st


def apply_global_layout():
    """Apply consistent full-width content layout across all Streamlit pages."""
    st.markdown(
        """
        <style>
        div[data-testid="stMainBlockContainer"] {
            max-width: 100%;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }
        @media (min-width: 1200px) {
            div[data-testid="stMainBlockContainer"] {
                padding-left: 2.0rem;
                padding-right: 2.0rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the global sidebar across all Streamlit pages."""
    apply_global_layout()
    from services.export_service import get_residents
    
    with st.sidebar:
        st.title("Beta 6 Ops Studio")
        st.markdown("---")
        
        # Global Resident Selector
        residents = get_residents()
        
        if "global_resident" not in st.session_state:
            st.session_state["global_resident"] = residents[0] if residents else "All"
            
        def on_resident_change():
            pass # Handle clear caches if necessary
            
        selected = st.selectbox(
            "Resident Context",
            options=["All"] + residents if residents else ["All"],
            index=(["All"] + residents).index(st.session_state["global_resident"]) if st.session_state["global_resident"] in (["All"] + residents) else 0,
            key="selected_resident_ui",
            on_change=on_resident_change
        )
        
        # Check if changed
        if selected != st.session_state["global_resident"]:
            st.session_state["global_resident"] = selected

        st.markdown("---")
        st.caption("v2.0 Greenfield UI")
