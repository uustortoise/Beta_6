"""
Sleep Analysis Dashboard with Interactive Charts and Collapsible Sections

This module provides a comprehensive dashboard for sleep pattern analysis
with multiple chart types and collapsible sections for better organization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def auto_detect_sleep_room(pred_data: Dict[str, pd.DataFrame]) -> Optional[str]:
    """
    Automatically detect the most likely sleep room from prediction data
    
    Parameters:
    - pred_data: Dictionary of room names to prediction DataFrames
    
    Returns:
    - Room name with highest sleep probability, or None if no data
    """
    if not pred_data:
        return None
    
    sleep_scores = {}
    
    for room_name, df in pred_data.items():
        if df.empty or 'predicted_activity' not in df.columns:
            continue
        
        # 1. Check for 'sleeping' activity presence
        sleep_count = (df['predicted_activity'] == 'sleeping').sum()
        total_count = len(df)
        sleep_ratio = sleep_count / total_count if total_count > 0 else 0
        
        # 2. Check for nighttime activity patterns
        night_sleep_ratio = 0
        if 'timestamp' in df.columns:
            try:
                night_hours = df['timestamp'].dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5, 6])
                night_sleep_count = (df[night_hours]['predicted_activity'] == 'sleeping').sum()
                night_total = max(night_hours.sum(), 1)
                night_sleep_ratio = night_sleep_count / night_total
            except Exception as e:
                logger.warning(f"Error calculating night sleep ratio for {room_name}: {e}")
        
        # 3. Room name hints
        room_lower = room_name.lower()
        name_hint = 1.0 if any(keyword in room_lower for keyword in ['bed', 'sleep', 'master']) else 0.5
        
        # Combined score
        sleep_scores[room_name] = (
            sleep_ratio * 0.4 + 
            night_sleep_ratio * 0.4 + 
            name_hint * 0.2
        )
    
    # Return room with highest sleep score
    if sleep_scores:
        best_room = max(sleep_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Auto-detected sleep room: {best_room} (score: {sleep_scores[best_room]:.2f})")
        return best_room
    
    return None


def create_collapsible_section(title: str, content_func, expanded: bool = True, key: str = None):
    """
    Create a collapsible section with expand/collapse toggle
    
    Parameters:
    - title: Section title
    - content_func: Function that renders the section content
    - expanded: Whether section is expanded by default
    - key: Unique key for session state
    
    Returns:
    - None
    """
    try:
        if key is None:
            # Create a more unique key using title and a hash
            import hashlib
            key_hash = hashlib.md5(title.encode()).hexdigest()[:8]
            key = f"collapsible_{key_hash}"
        
        # Initialize session state for expansion state
        if key not in st.session_state:
            st.session_state[key] = expanded
        
        # Create section header with toggle
        col1, col2 = st.columns([1, 20])
        with col1:
            # Toggle icon (▼ for expanded, ▶ for collapsed)
            icon = "▼" if st.session_state[key] else "▶"
            toggle_key = f"{key}_toggle_{icon}"  # Include icon in key for uniqueness
            if st.button(icon, key=toggle_key, help=f"{'Collapse' if st.session_state[key] else 'Expand'} section"):
                st.session_state[key] = not st.session_state[key]
                st.rerun()
        
        with col2:
            st.markdown(f"### {title}")
        
        # Render content if expanded
        if st.session_state[key]:
            st.markdown("---")
            try:
                content_func()
            except Exception as e:
                logger.error(f"Error rendering collapsible section '{title}': {e}")
                st.error(f"Error rendering section: {str(e)}")
            st.markdown("---")
    
    except Exception as e:
        logger.error(f"Error creating collapsible section '{title}': {e}")
        # Fallback: just render the title and content without collapsible
        st.markdown(f"### {title}")
        st.markdown("---")
        try:
            content_func()
        except Exception as inner_e:
            logger.error(f"Error in fallback rendering: {inner_e}")
            st.error(f"Error: {str(inner_e)}")
        st.markdown("---")


def create_quality_gauge(quality_score: Dict) -> go.Figure:
    """Create sleep quality gauge chart"""
    score = quality_score.get('overall_score', 0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Sleep Quality Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 85], 'color': "lightgreen"},
                {'range': [85, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_duration_trend(sleep_history: List[Dict]) -> go.Figure:
    """Create sleep duration trend line chart"""
    if not sleep_history or len(sleep_history) < 2:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient historical data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    # Extract dates and durations
    dates = []
    durations = []
    
    for entry in sleep_history[-7:]:  # Last 7 entries
        if 'timestamp' in entry and 'metrics' in entry:
            dates.append(entry['timestamp'].date())
            durations.append(entry['metrics'].get('duration_hours', 0))
    
    if len(dates) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need more data for trend analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=durations,
        mode='lines+markers',
        name='Sleep Duration',
        line=dict(color='blue', width=2),
        marker=dict(size=8, color='blue')
    ))
    
    # Add recommended range (7-9 hours)
    fig.add_hrect(
        y0=7, y1=9,
        fillcolor="lightgreen", opacity=0.2,
        annotation_text="Recommended Range", 
        annotation_position="top left",
        line_width=0
    )
    
    fig.update_layout(
        title="Sleep Duration Trend",
        xaxis_title="Date",
        yaxis_title="Hours",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    return fig


def create_sleep_awake_distribution(activity_data: pd.DataFrame) -> go.Figure:
    """Create pie chart showing sleep vs awake time distribution"""
    if activity_data.empty or 'predicted_activity' not in activity_data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No activity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    # Helper function to check if activity is sleep (using strict detection like SleepDetector)
    def _is_sleep_activity(activity: str) -> bool:
        """
        Strict sleep detection for dashboard display.
        Only counts definite sleep activities, excludes ambiguous terms.
        """
        if activity is None:
            return False
        
        activity_lower = str(activity).lower().strip()
        
        # Check for definite sleep keywords
        definite_sleep_keywords = ['sleeping', 'sleep', 'deep_sleep', 'rem_sleep']
        
        # EXCLUDE ambiguous activities
        exclude_keywords = ['inactive', 'resting', 'sedentary', 'sitting', 
                           'watching', 'lying', 'rest', 'relaxing']
        
        # Check for definite sleep AND not excluded
        has_sleep_keyword = any(keyword in activity_lower for keyword in definite_sleep_keywords)
        has_excluded_keyword = any(keyword in activity_lower for keyword in exclude_keywords)
        
        return has_sleep_keyword and not has_excluded_keyword
    
    # Count sleep entries using flexible matching
    sleep_time = sum(1 for activity in activity_data['predicted_activity'] 
                    if _is_sleep_activity(activity))
    awake_time = len(activity_data) - sleep_time
    
    # DIAGNOSTIC: Log detailed activity breakdown
    logger.info("🔍 SLEEP DIAGNOSTIC: Activity Breakdown for Sleep vs Awake Distribution")
    logger.info(f"   Total data points: {len(activity_data)}")
    logger.info(f"   Sleep count: {sleep_time} ({sleep_time/len(activity_data)*100:.1f}%)")
    logger.info(f"   Awake count: {awake_time} ({awake_time/len(activity_data)*100:.1f}%)")
    
    # Count specific activities
    activity_counts = activity_data['predicted_activity'].value_counts()
    logger.info("🔍 SLEEP DIAGNOSTIC: Activity counts:")
    for activity, count in activity_counts.items():
        percentage = count / len(activity_data) * 100
        logger.info(f"   '{activity}': {count} ({percentage:.1f}%)")
    
    # Check which activities are classified as sleep
    sleep_activities = []
    for activity in activity_data['predicted_activity'].unique():
        if _is_sleep_activity(activity):
            sleep_activities.append(activity)
    
    logger.info(f"🔍 SLEEP DIAGNOSTIC: Activities classified as SLEEP: {sleep_activities}")
    
    if sleep_time + awake_time == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No activity data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    fig = go.Figure(data=[go.Pie(
        labels=['Sleeping', 'Awake'],
        values=[sleep_time, awake_time],
        hole=0.3,
        marker_colors=['#1f77b4', '#ff7f0e'],
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title="Sleep vs Awake Distribution",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True
    )
    
    return fig


def create_sleep_timeline(sleep_periods: List[Dict], room_name: str = "Room") -> go.Figure:
    """Create Gantt chart showing sleep periods timeline"""
    if not sleep_periods:
        fig = go.Figure()
        fig.add_annotation(
            text="No sleep periods detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    # Create timeline data
    timeline_data = []
    for i, period in enumerate(sleep_periods):
        timeline_data.append({
            'Period': f"Sleep {i+1}",
            'Start': period['start'],
            'End': period['end'],
            'Duration (hours)': period['duration_minutes'] / 60,
            'Room': room_name
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = go.Figure()
    
    for i, row in timeline_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Start'], row['End']],
            y=[row['Room'], row['Room']],
            mode='lines+markers',
            line=dict(width=10, color='blue'),
            marker=dict(size=10, color='blue'),
            name=row['Period'],
            hoverinfo='text',
            text=f"{row['Period']}: {row['Duration (hours)']:.1f} hours",
            showlegend=True
        ))
    
    fig.update_layout(
        title="Sleep Periods Timeline",
        xaxis_title="Time",
        yaxis_title="Room",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified'
    )
    
    return fig


def create_movement_pattern_chart(sensor_data: pd.DataFrame, sleep_periods: List[Dict]) -> go.Figure:
    """Create chart showing movement patterns during sleep"""
    if sensor_data.empty or 'timestamp' not in sensor_data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No sensor data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    fig = go.Figure()
    
    # Plot motion sensor data if available
    if 'motion' in sensor_data.columns:
        fig.add_trace(go.Scatter(
            x=sensor_data['timestamp'],
            y=sensor_data['motion'],
            mode='lines',
            name='Motion Level',
            line=dict(color='gray', width=1),
            opacity=0.7
        ))
    
    # Highlight sleep periods
    for i, period in enumerate(sleep_periods):
        fig.add_vrect(
            x0=period['start'], x1=period['end'],
            fillcolor="blue", opacity=0.2,
            annotation_text=f"Sleep {i+1}",
            annotation_position="top left",
            line_width=0
        )
    
    fig.update_layout(
        title="Movement Patterns During Sleep",
        xaxis_title="Time",
        yaxis_title="Motion Level",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True
    )
    
    return fig


def create_environmental_conditions_chart(sensor_data: pd.DataFrame, sleep_periods: List[Dict]) -> go.Figure:
    """Create chart showing temperature and humidity during sleep"""
    if sensor_data.empty or 'timestamp' not in sensor_data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No sensor data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    fig = go.Figure()
    
    has_temp = 'temperature' in sensor_data.columns
    has_humidity = 'humidity' in sensor_data.columns
    
    if not has_temp and not has_humidity:
        fig.add_annotation(
            text="No environmental sensor data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    # Plot temperature if available
    if has_temp:
        fig.add_trace(go.Scatter(
            x=sensor_data['timestamp'],
            y=sensor_data['temperature'],
            mode='lines',
            name='Temperature (°C)',
            line=dict(color='red', width=2),
            yaxis='y'
        ))
    
    # Plot humidity if available
    if has_humidity:
        fig.add_trace(go.Scatter(
            x=sensor_data['timestamp'],
            y=sensor_data['humidity'],
            mode='lines',
            name='Humidity (%)',
            line=dict(color='blue', width=2),
            yaxis='y2'
        ))
    
    # Highlight sleep periods
    for i, period in enumerate(sleep_periods):
        fig.add_vrect(
            x0=period['start'], x1=period['end'],
            fillcolor="lightblue", opacity=0.1,
            annotation_text=f"Sleep {i+1}",
            annotation_position="top left",
            line_width=0
        )
    
    # Configure dual y-axes
    layout_updates = {
        'title': "Environmental Conditions During Sleep",
        'xaxis_title': "Time",
        'height': 300,
        'margin': dict(l=20, r=20, t=50, b=20),
        'showlegend': True
    }
    
    if has_temp and has_humidity:
        layout_updates['yaxis'] = dict(title="Temperature (°C)", side="left")
        layout_updates['yaxis2'] = dict(
            title="Humidity (%)", 
            side="right", 
            overlaying="y",
            range=[0, 100]  # Humidity percentage range
        )
    elif has_temp:
        layout_updates['yaxis_title'] = "Temperature (°C)"
    elif has_humidity:
        layout_updates['yaxis_title'] = "Humidity (%)"
    
    fig.update_layout(**layout_updates)
    
    return fig


def create_sleep_quality_radar(quality_score: Dict) -> go.Figure:
    """Create radar chart showing sleep quality factors"""
    factor_scores = quality_score.get('factor_scores', {})
    
    if not factor_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No factor scores available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    factors = list(factor_scores.keys())
    scores = list(factor_scores.values())
    
    # Close the polygon
    factors_display = [f.replace('_', ' ').title() for f in factors]
    factors_display.append(factors_display[0])
    scores.append(scores[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=factors_display,
        fill='toself',
        name='Sleep Quality Factors',
        line=dict(color='blue', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=10)
            )
        ),
        title="Sleep Quality Factors Analysis",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    return fig


def create_historical_comparison(sleep_history: List[Dict]) -> go.Figure:
    """Create bar chart comparing sleep metrics over multiple days"""
    if not sleep_history or len(sleep_history) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need more data for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    # Get last 7 days of data
    recent_history = sleep_history[-7:]
    
    dates = []
    durations = []
    qualities = []
    
    for entry in recent_history:
        if 'timestamp' in entry and 'metrics' in entry and 'quality_score' in entry:
            dates.append(entry['timestamp'].strftime('%m/%d'))
            durations.append(entry['metrics'].get('duration_hours', 0))
            qualities.append(entry['quality_score'].get('overall_score', 0))
    
    if len(dates) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    # Create grouped bar chart
    fig = go.Figure()
    
    # Add duration bars
    fig.add_trace(go.Bar(
        x=dates,
        y=durations,
        name='Duration (hours)',
        marker_color='#1f77b4',
        text=[f"{d:.1f}h" for d in durations],
        textposition='auto'
    ))
    
    # Add quality bars (scaled to match duration range)
    max_duration = max(durations) if durations else 10
    scaled_qualities = [q * max_duration / 100 for q in qualities]
    
    fig.add_trace(go.Bar(
        x=dates,
        y=scaled_qualities,
        name='Quality Score (scaled)',
        marker_color='#2ca02c',
        text=[f"{q:.0f}" for q in qualities],
        textposition='auto'
    ))
    
    # Add secondary y-axis for quality scores
    fig.update_layout(
        title="Sleep Metrics Comparison (Last 7 Days)",
        xaxis_title="Date",
        yaxis=dict(
            title="Duration (hours)",
            side="left",
            range=[0, max_duration * 1.2]
        ),
        yaxis2=dict(
            title="Quality Score",
            side="right",
            overlaying="y",
            range=[0, 100],
            tickmode="sync"
        ),
        barmode='group',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig
