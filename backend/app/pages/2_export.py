import datetime
import importlib
import sys
from pathlib import Path

_backend = str(Path(__file__).resolve().parent.parent.parent)
if _backend not in sys.path: sys.path.extend([_backend, str(Path(_backend).parent)])

import streamlit as st

import services.export_service as export_service
from app._sidebar import render_sidebar


def render():
    render_sidebar()
    st.header("📦 Export Data")
    
    elder_id = st.session_state.get("global_resident", "All")
    
    # 1. Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.date.today() - datetime.timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.date.today())
    with col3:
        export_type = st.selectbox("Export Type", ["Raw ADL History", "Predicted Results", "Review Candidates"])
        
    include_no_predictions = st.toggle("Include residents with no predictions", value=False)
    
    # 2. Generate
    if st.button("Generate Export", type="primary"):
        with st.spinner("Generating..."):
            try:
                if export_type == "Raw ADL History":
                    df = export_service.export_raw_adl(elder_id, start_date, end_date)
                elif export_type == "Predicted Results":
                    df = export_service.export_predicted_results(elder_id, start_date, end_date)
                else:
                    df = export_service.export_review_candidates(elder_id, start_date, end_date)
                    
                if df.empty:
                    st.warning("No data found matching the criteria.")
                    return
                    
                st.success(f"Generated {len(df)} rows.")
                
                # Preview
                st.dataframe(df.head(50), use_container_width=True, hide_index=True)
                
                # Download
                excel_data = export_service.to_excel_bytes(df)
                filename = f"{elder_id}_{export_type.lower().replace(' ','_')}_{start_date}_{end_date}.xlsx"
                
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")


if __name__ == "__main__":
    if "global_resident" not in st.session_state:
        st.session_state["global_resident"] = "All"
    render()
