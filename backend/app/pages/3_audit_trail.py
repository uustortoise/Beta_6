import datetime
import sys
from pathlib import Path

_backend = str(Path(__file__).resolve().parent.parent.parent)
if _backend not in sys.path: sys.path.extend([_backend, str(Path(_backend).parent)])

import pandas as pd
import streamlit as st

import services.audit_service as audit_service
from app._sidebar import render_sidebar


def render():
    render_sidebar()
    st.header("📝 Correction Audit Trail")
    
    elder_id = st.session_state.get("global_resident", "All")
    
    # 1. Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        days = st.selectbox("Timeframe", options=[1, 7, 30, 90, 365, 3650], 
                            format_func=lambda x: f"Last {x} days" if x < 3650 else "All Time", 
                            index=2)
    with col2:
        from config import get_room_config
        rooms = ["All"] + [r for r in get_room_config().get_all_rooms().keys() if r != "default"]
        room = st.selectbox("Room filter", rooms)
    with col3:
        corrected_by = st.selectbox("Corrected By", ["All", "operator", "admin_ui", "ml_pipeline"])
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        show_deleted = st.checkbox("Show Rolled Back", value=False)
        
    # 2. Fetch Data
    with st.spinner("Loading audit trail..."):
        trail_df = audit_service.fetch_correction_trail(
            elder_id=elder_id,
            room=room,
            corrected_by=corrected_by,
            days=days,
            include_deleted=show_deleted
        )
        
        evaluation_df = audit_service.fetch_evaluation_history(
            elder_id=elder_id,
            days=days
        )
        
    # 3. Summary Panel
    st.markdown("---")
    sc1, sc2, sc3 = st.columns(3)
    
    total_corrections = len(trail_df) if not trail_df.empty else 0
    sc1.metric("Corrections Logged", total_corrections)
    
    if not evaluation_df.empty:
        total_evals = len(evaluation_df)
        passed_evals = len(evaluation_df[evaluation_df['decision'].isin(['PASS', 'PASS_WITH_FLAG'])])
        sc2.metric("Retrain Evaluations", total_evals)
        sc3.metric("Successful Retrains", passed_evals)
    else:
        sc2.metric("Retrain Evaluations", 0)
        sc3.metric("Successful Retrains", 0)
        
    st.markdown("---")
    
    # 4. Data Table and Rollback
    if trail_df.empty:
        st.info("No corrections found matching these filters.")
        return
        
    # Display table
    display_cols = ['id', 'corrected_at', 'elder_id', 'room', 'duration_str', 'old_activity', 'new_activity', 'rows_affected', 'corrected_by']
    if show_deleted:
        display_cols.append('is_deleted')
        
    if not show_deleted:
        st.dataframe(
            trail_df[display_cols].style.apply(
                lambda x: ['background: #ffebee' if x.name == 'old_activity' 
                           else 'background: #e8f7ee' if x.name == 'new_activity' else '' for i in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("### ⏪ Rollback Correction")
        rb_col1, rb_col2 = st.columns([1, 4])
        with rb_col1:
            rollback_id = st.number_input("Correction ID", min_value=0, value=0, step=1)
        with rb_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Execute Rollback", type="primary") and rollback_id > 0:
                success, msg = audit_service.rollback_correction(
                    correction_id=rollback_id,
                    rolled_back_by="admin_ui"
                )
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.dataframe(trail_df[display_cols], use_container_width=True, hide_index=True)
        
    # Download
    csv = trail_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Audit CSV",
        csv,
        "audit_trail.csv",
        "text/csv",
        key='download-audit-csv'
    )


if __name__ == "__main__":
    if "global_resident" not in st.session_state:
        st.session_state["global_resident"] = "All"
    render()
