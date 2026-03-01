"""
Long-Term ADL (Activities of Daily Living) Analysis Module
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ADLAnalyzer:
    """
    Analyzer for long-term trends in Activities of Daily Living.
    Focuses on daily aggregations, trend detection, and anomaly spotting over weeks/months.
    """
    
    # Standard activities for consistent tracking
    STANDARD_ACTIVITIES = ['sleeping', 'lying', 'sitting', 'standing', 'walking', 'fall']
    
    def __init__(self, platform: Any):
        self.platform = platform
    
    def analyze_and_save(self, elder_id: str, prediction_data: Dict[str, pd.DataFrame], data_manager: Any, custom_timestamp: datetime = None) -> Dict[str, Any]:
        """
        Main entry point: Analyzes prediction data, updates history, and returns current stats.
        """
        try:
            # 1. Calculate Daily Stats from current prediction
            daily_stats = self.aggregate_daily_stats(prediction_data)
            
            if daily_stats.empty:
                logger.warning("No daily stats generated from prediction data")
                return {}
                
            # 2. Load existing history
            history = data_manager.load_adl_history(elder_id)
            history_df = pd.DataFrame(history)
            if not history_df.empty:
                if 'date' in history_df.columns:
                    history_df['date'] = pd.to_datetime(history_df['date']).dt.date
                    history_df = history_df.set_index('date')
            
            # 3. Update history with new data (merge/overwrite for same dates)
            # Convert daily_stats index to date objects for matching
            daily_stats.index = pd.to_datetime(daily_stats.index).date
            daily_stats.index.name = 'date'
            
            if not history_df.empty:
                # Combine, preferring new data for same dates
                combined_df = daily_stats.combine_first(history_df)
                combined_df.sort_index(inplace=True)
            else:
                combined_df = daily_stats
            
            # 4. Detect Anomalies on updated history
            anomalies = self.detect_trends_and_anomalies(combined_df)
            
            # 5. Save updated history
            # Reset index to make date a column again for JSON serialization
            save_df = combined_df.reset_index()
            # Handle case where reset_index uses 'index' as name because original index name was lost
            if 'date' not in save_df.columns and 'index' in save_df.columns:
                save_df = save_df.rename(columns={'index': 'date'})
            
            # Convert date objects to strings
            save_df['date'] = save_df['date'].astype(str)
            
            new_history_records = save_df.to_dict('records')
            data_manager.save_adl_history(elder_id, new_history_records)
            
            return {
                "daily_stats": daily_stats.reset_index().to_dict('records'),
                "anomalies": anomalies,
                "history_length": len(new_history_records)
            }
            
        except Exception as e:
            logger.error(f"ADL Analysis failed: {e}", exc_info=True)
            return {"error": str(e)}

    def aggregate_daily_stats(self, prediction_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregates room-level prediction data into a daily statistics DataFrame.
        """
        all_activities = []
        
        # Merge all rooms into a single stream for overall stats
        for room, df in prediction_data.items():
            if df.empty or 'timestamp' not in df.columns:
                continue
            
            # Look for activity column
            act_col = 'predicted_activity' if 'predicted_activity' in df.columns else None
            # Fallback to 'activity' if predicted not available (e.g. training data used as mock)
            if not act_col and 'activity' in df.columns:
                act_col = 'activity'
                
            if not act_col:
                continue
                
            temp_df = df.copy()
            temp_df['activity_label'] = temp_df[act_col]
            temp_df['room'] = room
            all_activities.append(temp_df)
            
        if not all_activities:
            return pd.DataFrame()
            
        full_df = pd.concat(all_activities)
        # Ensure timestamp is datetime
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        full_df['date'] = full_df['timestamp'].dt.date
        
        # --- Calculate Metrics ---
        
        daily_stats = []
        grouped = full_df.groupby('date')
        
        for date, group in grouped:
            stats = {'date': date}
            total_samples = len(group)
            
            # 1. Total Activity Counts & Duration (assuming fixed interval, e.g., 10s or 1min)
            # We'll normalize to percentage or count. 
            # If we know interval, we can estimate minutes. Let's assume count for now.
            counts = group['activity_label'].value_counts()
            
            # Initialize standard activities with 0
            for act in self.STANDARD_ACTIVITIES:
                stats[f'count_{act}'] = 0
                
            for act, count in counts.items():
                # Normalize label
                clean_act = str(act).lower().strip()
                stats[f'count_{clean_act}'] = count
                
            # 2. Mobility Index (Proxy: Percentage of non-sedentary activities)
            sedentary_labels = ['sitting', 'lying', 'sleeping', 'no_activity', 'none', 'rest', 'bed']
            active_samples = group[~group['activity_label'].str.lower().isin(sedentary_labels)].shape[0]
            stats['mobility_index'] = (active_samples / total_samples) if total_samples > 0 else 0
            
            # 3. Activity Diversity (Entropy)
            probs = counts / total_samples
            entropy = -np.sum(probs * np.log2(probs + 1e-9))
            stats['diversity_score'] = entropy

            # 4. Night Activity (e.g., movement/bathroom between 12AM - 5AM)
            night_mask = (group['timestamp'].dt.hour >= 0) & (group['timestamp'].dt.hour < 5)
            night_activity = group[night_mask]
            stats['night_activity_count'] = len(night_activity)
            
            # Specific check for 'bathroom' if room name contains it
            bathroom_night = night_activity[night_activity['room'].str.contains('bath', case=False, na=False)]
            stats['bathroom_night_visits'] = len(bathroom_night)
            
            daily_stats.append(stats)
            
        daily_df = pd.DataFrame(daily_stats)
        if not daily_df.empty:
            daily_df = daily_df.set_index('date').sort_index()
            
        return daily_df

    def detect_trends_and_anomalies(self, daily_df: pd.DataFrame, window: int = 7) -> Dict[str, Any]:
        """
        Detects significant trends and anomalies in the daily statistics.
        """
        if daily_df.empty or len(daily_df) < 3:
            return {"status": "insufficient_data"}
            
        anomalies = []
        trends = {}
        
        # 1. Mobility Trend (Slope)
        if 'mobility_index' in daily_df.columns:
            recent = daily_df['mobility_index'].tail(30) # Last 30 days
            if len(recent) > 1:
                # Calculate slope
                y = recent.values
                x = np.arange(len(y))
                slope, _ = np.polyfit(x, y, 1)
                trends['mobility_slope'] = slope
                
                if slope < -0.01:
                    anomalies.append(f"Declining mobility detected (Slope: {slope:.4f})")
        
        # 2. Night Activity Spikes
        if 'night_activity_count' in daily_df.columns:
            recent_night = daily_df['night_activity_count'].iloc[-1]
            if len(daily_df) >= window:
                avg_night = daily_df['night_activity_count'].rolling(window=window).mean().iloc[-2] # Exclude current
                std_night = daily_df['night_activity_count'].rolling(window=window).std().iloc[-2]
                
                if not np.isnan(avg_night) and not np.isnan(std_night):
                    if recent_night > avg_night + 2 * std_night and recent_night > 10:
                        anomalies.append(f"Unusual high night activity ({recent_night} events vs avg {avg_night:.1f})")

        return {
            "detected_anomalies": anomalies,
            "trends": trends
        }

    def create_dashboard(self, history_df: pd.DataFrame, visible_charts: List[str] = None):
        """
        Creates a Plotly dashboard for the aggregated ADL stats.
        visible_charts: list of chart keys to display ['mobility', 'diversity', 'night', 'activities']
        """
        if history_df.empty:
            return go.Figure().add_annotation(text="No historical ADL data available")
            
        # Default visibility
        if visible_charts is None:
            visible_charts = ['mobility', 'diversity', 'night']
            
        # Determine rows needed
        rows = 0
        chart_map = {}
        
        if 'mobility' in visible_charts:
            rows += 1
            chart_map['mobility'] = rows
        if 'diversity' in visible_charts:
            rows += 1
            chart_map['diversity'] = rows
        if 'night' in visible_charts:
            rows += 1
            chart_map['night'] = rows
        if 'activities' in visible_charts:
            rows += 1
            chart_map['activities'] = rows
            
        if rows == 0:
             return go.Figure().add_annotation(text="No charts selected")

        # Define titles for subplots
        title_map = {
            'mobility': '24-Hour Activity Ratio',
            'diversity': 'Activity Diversity',
            'night': 'Overnight Restlessness Trend',
            'activities': 'Activity Stack'
        }
        
        subplot_titles = [title_map.get(k, k.title()) for k in sorted(chart_map, key=chart_map.get)]

        fig = make_subplots(
            rows=rows, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=subplot_titles
        )
        
        # Ensure index is datetime
        if not isinstance(history_df.index, pd.DatetimeIndex):
            history_df.index = pd.to_datetime(history_df.index)
        
        # 1. Mobility Index
        if 'mobility' in visible_charts:
            row = chart_map['mobility']
            fig.add_trace(
                go.Scatter(x=history_df.index, y=history_df.get('mobility_index', []), 
                          mode='lines+markers', name='24-Hour Activity Ratio',
                          line=dict(color='#3b82f6', width=2)),
                row=row, col=1
            )
            # Trendline
            if len(history_df) > 1 and 'mobility_index' in history_df.columns:
                z = np.polyfit(range(len(history_df)), history_df['mobility_index'], 1)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(x=history_df.index, y=p(range(len(history_df))), 
                              mode='lines', name='Activity Trend',
                              line=dict(color='orange', dash='dash'), hoverinfo='skip'),
                    row=row, col=1
                )

        # 2. Diversity Score
        if 'diversity' in visible_charts:
            row = chart_map['diversity']
            fig.add_trace(
                go.Bar(x=history_df.index, y=history_df.get('diversity_score', []), 
                      name='Activity Diversity', marker_color='#10b981'),
                row=row, col=1
            )

        # 3. Night Stats
        if 'night' in visible_charts:
            row = chart_map['night']
            if 'night_activity_count' in history_df.columns:
                fig.add_trace(
                    go.Bar(x=history_df.index, y=history_df['night_activity_count'], 
                          name='Overnight Restlessness', marker_color='#6b7280', opacity=0.6),
                    row=row, col=1
                )
            if 'bathroom_night_visits' in history_df.columns:
                fig.add_trace(
                    go.Scatter(x=history_df.index, y=history_df['bathroom_night_visits'], 
                              mode='lines+markers', name='Bathroom Visits',
                              line=dict(color='#ef4444', width=2)),
                    row=row, col=1
                )
                
        # 4. Specific Activities Stacked
        if 'activities' in visible_charts:
            row = chart_map['activities']
            # Find all count columns
            count_cols = [c for c in history_df.columns if c.startswith('count_')]
            for col in count_cols:
                act_name = col.replace('count_', '').title()
                fig.add_trace(
                    go.Bar(x=history_df.index, y=history_df[col], name=act_name),
                    row=row, col=1
                )
            fig.update_layout(barmode='stack')

        fig.update_layout(height=300*rows, title_text="Health Trends Analysis", showlegend=True)
        return fig
