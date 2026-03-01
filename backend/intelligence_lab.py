"""
Intelligence Lab - Beta 5 Add-On Visualization

A Streamlit dashboard to visualize Intelligence Phase features:
- Trajectory Timeline (Cross-Room Tracking)
- Model Training History
- Context Episodes (Coming Soon)
- Routine Anomalies (Coming Soon)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add backend to path for settings import
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from elderlycare_v1_16.config.settings import DB_PATH
from datetime import datetime, timedelta
import json
from utils.intelligence_db import query_to_dataframe


st.set_page_config(
    page_title="Intelligence Lab - Beta 5",
    page_icon="🧠",
    layout="wide"
)

# Database connection (PostgreSQL compatible)
try:
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
    _lab_adapter = LegacyDatabaseAdapter()
except ImportError:
    import sqlite3
    _lab_adapter = None

def get_db_connection():
    """Get database connection compatible with PostgreSQL or SQLite."""
    if _lab_adapter:
        return _lab_adapter.get_connection()
    else:
        import sqlite3
        return sqlite3.connect(DB_PATH)

def get_elders():
    """Get list of elders from database."""
    with get_db_connection() as conn:
        df = query_to_dataframe(conn, "SELECT DISTINCT elder_id FROM elders ORDER BY elder_id")
        return df['elder_id'].tolist() if not df.empty else []

def get_trajectory_data(elder_id: str, date_str: str):
    """Fetch trajectory events for a specific day."""
    with get_db_connection() as conn:
        query = """
            SELECT id, start_time, end_time, path, primary_activity, 
                   room_sequence, duration_minutes, confidence
            FROM trajectory_events
            WHERE elder_id = ? AND record_date = ?
            ORDER BY start_time
        """
        df = query_to_dataframe(conn, query, (elder_id, date_str))
        return df

def get_available_dates(elder_id: str):
    """Get dates with trajectory data for an elder."""
    with get_db_connection() as conn:
        query = """
            SELECT DISTINCT record_date 
            FROM trajectory_events 
            WHERE elder_id = ?
            ORDER BY record_date DESC
        """
        df = query_to_dataframe(conn, query, (elder_id,))
        return df['record_date'].tolist() if not df.empty else []

def get_training_history(elder_id: str):
    """Fetch training logs for an elder."""
    with get_db_connection() as conn:
        try:
            query = """
                SELECT timestamp, room, model_type, accuracy, samples_count, epochs, status, error_message,
                       data_start_time, data_end_time
                FROM model_training_history
                WHERE elder_id = ?
                ORDER BY timestamp DESC
            """
            df = query_to_dataframe(conn, query, (elder_id,))
            return df
        except Exception as e:
            return pd.DataFrame()

# ============================================================
# MAIN UI
# ============================================================

st.title("🧠 Intelligence Lab")
st.markdown("*Beta 5 Add-On: Advanced Activity Intelligence*")

# Sidebar
st.sidebar.header("Configuration")

elders = get_elders()
if not elders:
    st.warning("No elders found in database. Process some data first.")
    st.stop()

selected_elder = st.sidebar.selectbox("Select Elder", elders)

# Get available dates for this elder
available_dates = get_available_dates(selected_elder)
if not available_dates:
    st.info(f"No trajectory data available for {selected_elder}. Run the Intelligence pipeline first.")
    # Show history even if no trajectory data
    st.markdown("---")
else:
    selected_date = st.sidebar.selectbox("Select Date", available_dates)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Intelligence Modules")
    st.sidebar.markdown("✅ **Part 1**: Cross-Room Tracking")
    st.sidebar.markdown("⏳ **Part 2**: Context Classification")
    st.sidebar.markdown("⏳ **Part 3**: Pattern Learning")

    # ============================================================
    # TRAJECTORY VISUALIZATION
    # ============================================================

    st.header(f"📍 Cross-Room Trajectories: {selected_date}")

    trajectories = get_trajectory_data(selected_elder, selected_date)

    if trajectories.empty:
        st.info("No trajectories found for this date.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trajectories", len(trajectories))
        col2.metric("Total Duration", f"{trajectories['duration_minutes'].sum():.0f} min")
        col3.metric("Avg Confidence", f"{trajectories['confidence'].mean():.1%}")
        col4.metric("Unique Activities", trajectories['primary_activity'].nunique())
        
        st.markdown("---")
        
        # Gantt Chart
        st.subheader("Timeline View")
        
        # Prepare data for Gantt
        trajectories['start_time'] = pd.to_datetime(trajectories['start_time'])
        trajectories['end_time'] = pd.to_datetime(trajectories['end_time'])
        
        # Create labels
        trajectories['label'] = trajectories.apply(
            lambda x: f"{x['primary_activity']} ({x['duration_minutes']:.0f}m)", 
            axis=1
        )
        
        # Color by activity
        fig = px.timeline(
            trajectories,
            x_start='start_time',
            x_end='end_time',
            y='path',
            color='primary_activity',
            hover_data=['duration_minutes', 'confidence'],
            title="Movement Trajectories Throughout the Day",
            labels={'path': 'Movement Path', 'primary_activity': 'Activity'}
        )
        
        fig.update_yaxes(categoryorder='total ascending')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Room Sequence Details
        st.subheader("Trajectory Details")
        
        for idx, row in trajectories.iterrows():
            with st.expander(f"🔄 {row['start_time'].strftime('%H:%M')} - {row['end_time'].strftime('%H:%M')} | {row['primary_activity']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Path:** `{row['path']}`")
                    st.markdown(f"**Duration:** {row['duration_minutes']:.1f} minutes")
                    st.markdown(f"**Confidence:** {row['confidence']:.1%}")
                    
                with col2:
                    # Parse room sequence
                    try:
                        sequence = json.loads(row['room_sequence'])
                        rooms = [s['room'] for s in sequence]
                        st.markdown("**Room Visits:**")
                        for i, s in enumerate(sequence):
                            st.markdown(f"  {i+1}. **{s['room']}** ({s['duration_min']}m)")
                    except:
                        st.text("Room sequence unavailable")

        # Raw Data Table
        with st.expander("📊 Raw Data Table"):
            display_df = trajectories[['start_time', 'end_time', 'path', 'primary_activity', 'duration_minutes', 'confidence']].copy()
            display_df.columns = ['Start', 'End', 'Path', 'Activity', 'Duration (min)', 'Confidence']
            st.dataframe(display_df, use_container_width=True)

# ============================================================
# MODEL TRAINING HISTORY PANEL (NEW)
# ============================================================
st.markdown("---")
st.subheader("🛠️ Model Training History")

training_df = get_training_history(selected_elder)

if training_df.empty:
    st.info("No training history found. Train models to see logs here.")
else:
    with st.expander("View Training Logs", expanded=True):
        # Optional Filters
        rooms = training_df['room'].unique().tolist()
        selected_rooms = st.multiselect("Filter by Room", rooms, default=rooms)
        
        filtered_df = training_df[training_df['room'].isin(selected_rooms)]
        
        # Create a formatted data range column if timestamps exist
        if 'data_start_time' in filtered_df.columns and 'data_end_time' in filtered_df.columns:
            def format_range(row):
                if pd.isna(row['data_start_time']) or pd.isna(row['data_end_time']):
                    return 'N/A'
                try:
                    start = pd.to_datetime(row['data_start_time'])
                    end = pd.to_datetime(row['data_end_time'])
                    return f"{start.strftime('%d %b %H:%M')} - {end.strftime('%H:%M')}"
                except:
                    return 'N/A'
            filtered_df = filtered_df.copy()
            filtered_df['data_range'] = filtered_df.apply(format_range, axis=1)
            display_cols = ['timestamp', 'room', 'model_type', 'accuracy', 'samples_count', 'data_range', 'status']
        else:
            display_cols = ['timestamp', 'room', 'model_type', 'accuracy', 'samples_count', 'status']
        
        st.dataframe(
            filtered_df[display_cols] if all(c in filtered_df.columns for c in display_cols) else filtered_df,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Run Time", format="D MMM, HH:mm:ss"),
                "accuracy": st.column_config.ProgressColumn(
                    "Accuracy", 
                    format="%.2f", 
                    min_value=0, 
                    max_value=1
                ),
                "samples_count": st.column_config.NumberColumn("Samples"),
                "data_range": st.column_config.TextColumn("Training Data Range"),
                "status": st.column_config.TextColumn("Status"),
            },
            hide_index=True,
            use_container_width=True
        )

st.markdown("---")
st.caption("Intelligence Lab - Beta 5 Add-On | Part 1: Cross-Room Tracking")
