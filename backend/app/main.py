import logging
import sys
from pathlib import Path

# Path setup — Streamlit runs this as a script, so we must inline it
_backend = str(Path(__file__).resolve().parent.parent)
if _backend not in sys.path: sys.path.extend([_backend, str(Path(_backend).parent)])

import streamlit as st

from services.export_service import get_residents

# Config
st.set_page_config(
    page_title="Beta 6 Operations Studio",
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

from app._sidebar import render_sidebar
render_sidebar()

# Main Area
st.title("🧠 Welcome to Beta 6 Operations Studio")
st.markdown("""
This is the unified control center for managing the Beta 6 ML pipeline and data quality.
Please select a tool from the sidebar to begin:

- **1_correction_studio**: Rapid manual labeling and low-confidence triage.
- **2_export**: Extracting prediction datasets and ADL features.
- **3_audit_trail**: Review past corrections and rollback changes.
- **4_ops_dashboard**: Top-level model health and data freshness metrics.

**Current Context:** `{}`
""".format(st.session_state.get("global_resident", "All")))

