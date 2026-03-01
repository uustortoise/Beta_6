import sys
from pathlib import Path

_backend = str(Path(__file__).resolve().parent.parent.parent)
if _backend not in sys.path: sys.path.extend([_backend, str(Path(_backend).parent)])

import pandas as pd
import streamlit as st

import services.ops_service as ops_service
from app._sidebar import render_sidebar


def _format_metric(value) -> str:
    """Render numeric metrics safely for UI display."""
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def render():
    render_sidebar()
    st.header("📊 Operations & Health Dashboard")
    
    elder_id = st.session_state.get("global_resident", "All")
    if elder_id == "All":
        st.warning("Select a specific Resident Context from the sidebar.")
        return
        
    view_mode = st.radio("Display Mode", options=["Ops View (Status)", "ML View (Deep Dive)"], horizontal=True)
    is_ml_mode = "ML View" in view_mode
    
    st.markdown("---")
    
    # 1. Daily Operations (Data Flow)
    st.subheader("Section A: Daily Operations Health")
    with st.spinner("Fetching ops..."):
        ops_sum = ops_service.get_daily_ops_summary(elder_id)
        
    colA1, colA2, colA3 = st.columns(3)
    
    stale = ops_sum.get("is_stale", True)
    hrs = ops_sum.get("hours_since_last_ingestion")
    health = "🔴 Stale/Missing" if stale else "🟢 Healthy"
    
    colA1.metric("Ingestion Status", health)
    colA2.metric("Last Data (Hours Ago)", f"{hrs:.1f}h" if hrs is not None else "N/A", 
                 delta="Action Needed" if stale else "Normal", delta_color="inverse")
    if ops_sum.get("last_ingestion_time") is not None:
        colA2.caption(f"Last timestamp: {pd.to_datetime(ops_sum.get('last_ingestion_time')).strftime('%Y-%m-%d %H:%M:%S')}")
    colA3.metric("Open Alerts", ops_sum.get("open_alerts", 0),
                 delta="Review" if ops_sum.get("open_alerts", 0) > 0 else "Clear", delta_color="inverse")
                 
    # 2. Sample Collection
    st.markdown("---")
    st.subheader("Section B: Sample Collection (Beta 6 Requirements)")
    with st.spinner("Analyzing archives..."):
        samples = ops_service.get_sample_collection_status(elder_id)
        
    if not samples:
        st.info("No archive data found.")
    else:
        import altair as alt

        target = samples.get("target", 21)
        counts = samples.get("counts", {})
        
        # Build simple dataframe for bar chart
        df_samples = pd.DataFrame(list(counts.items()), columns=["Room", "Days Recorded"])
        
        colB1, colB2 = st.columns([1, 2])
        with colB1:
            ready = len(samples.get("ready_rooms", []))
            total = len(counts)
            st.metric("Rooms Ready for ML", f"{ready} / {total}")
            st.caption(f"Target: {target} days")
            
        with colB2:
            # Altair bar chart with threshold line
            bars = alt.Chart(df_samples).mark_bar().encode(
                x='Days Recorded:Q',
                y=alt.Y('Room:N', sort='-x'),
                color=alt.condition(
                    alt.datum['Days Recorded'] >= target,
                    alt.value('#1b5e20'), # Green if met
                    alt.value('#f57c00')  # Orange if under
                )
            )
            rule = alt.Chart(pd.DataFrame({'Target': [target]})).mark_rule(color='red').encode(x='Target:Q')
            st.altair_chart((bars + rule).properties(height=250), use_container_width=True)


    # 3. Model Maintenance
    st.markdown("---")
    st.subheader("Section C: Model Performance")
    with st.spinner("Fetching model status..."):
        ml_status = ops_service.get_model_status(elder_id)
        
    overall = ml_status.get("status", {}).get("overall", "unknown")
    
    if is_ml_mode:
        st.write(f"**Overall Status:** `{overall}`")
        if "rooms" in ml_status:
            for r in ml_status["rooms"]:
                metrics = r.get("metrics") or {}
                st.markdown(f"**{str(r.get('room')).title()}**")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Candidate F1", _format_metric(metrics.get("candidate_macro_f1_mean")))
                champion = metrics.get("champion_macro_f1_mean")
                prev_run = metrics.get("previous_run_macro_f1_mean")
                if champion is None and prev_run is not None:
                    mc2.metric("Prev Run F1", _format_metric(prev_run))
                else:
                    mc2.metric("Champion F1", _format_metric(champion))
                mc3.metric("Status", r.get("status"))
    else:
        # Ops view - simpler traffic lights
        st.write("Model performance is evaluated nightly via Walk-Forward validation.")
        
        status_map = {
            "healthy": "🟢 Healthy",
            "watch": "🟡 Needs Monitoring",
            "action_needed": "🔴 Action Needed",
            "not_available": "⚪ Not Available/Pending"
        }
        
        st.metric("Global Health", status_map.get(overall, "⚪ Unknown"))
        
        if "rooms" in ml_status:
            room_texts = []
            for r in ml_status["rooms"]:
                r_status = r.get("status")
                room_texts.append(f"- **{str(r.get('room')).title()}**: {status_map.get(r_status, 'Unknown')}")
            for text in room_texts:
                st.write(text)
                
                
    # 4. Hard Negatives
    st.markdown("---")
    st.subheader("Section D: Active Learning Queue")
    with st.spinner("Fetching queue..."):
        hn_sum = ops_service.get_hard_negative_summary(elder_id)
        
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("Open Hard Negatives", hn_sum.get("open_count", 0))
    hc2.metric("Recently Resolved", hn_sum.get("recent_resolved", 0))
    hc3.metric("Last Miner Run", str(hn_sum.get("last_run"))[:16] if hn_sum.get("last_run") else "Never")

if __name__ == "__main__":
    if "global_resident" not in st.session_state:
        st.session_state["global_resident"] = "All"
    render()
