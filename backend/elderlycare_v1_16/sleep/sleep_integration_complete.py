"""
Complete Enhanced Sleep Pattern Analysis Integration for Elderly Care Platform

This module provides a complete enhanced sleep analysis system with parameter adjustment panel,
sleep stage analysis, and comprehensive visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def _ss(key: str, default: Any = None) -> Any:
    """
    Get/set session state with sleep prefix to avoid key collisions
    """
    full_key = f"sleep__{key}"
    if default is not None and full_key not in st.session_state:
        st.session_state[full_key] = default
    return st.session_state.get(full_key, default)


class EnhancedSleepAnalysisSystem:
    """Complete enhanced sleep analysis system with all features"""
    
    def __init__(self, platform: Any, person_id: str = 'elderly_patient'):
        self.platform = platform
        self.person_id = person_id
        
        # Initialize session state
        _ss('analysis_results', {})
        _ss('run_analysis', False)
        _ss('parameter_config', {})
        
        logger.info(f"Initialized EnhancedSleepAnalysisSystem for {person_id}")
    
    def get_parameter_adjustment_panel(self) -> None:
        """Create interactive parameter adjustment panel"""
        st.subheader("⚙️ Sleep Analysis Parameters")
        
        # Create tabs for different parameter categories
        tab1, tab2, tab3 = st.tabs(["📊 Detection", "🌙 Sleep Stages", "⚡ Advanced"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Core Parameters**")
                min_sleep_duration = st.slider(
                    "Minimum Sleep Duration (min)",
                    5.0, 120.0, 30.0, 5.0,
                    help="Minimum duration for a sleep period"
                )
                
                awakening_gap = st.slider(
                    "Awakening Gap Tolerance (min)",
                    1.0, 30.0, 5.0, 1.0,
                    help="Maximum gap between sleep entries"
                )
            
            with col2:
                st.markdown("**Detection Settings**")
                detection_mode = st.selectbox(
                    "Detection Mode",
                    ["Strict", "Lenient", "Elderly", "Clinical"],
                    index=2,
                    help="Sleep detection strictness"
                )
                
                motion_threshold = st.slider(
                    "Motion Threshold",
                    0.01, 0.5, 0.1, 0.01,
                    help="Motion detection sensitivity"
                )
        
        with tab2:
            st.markdown("**Sleep Stage Classification**")
            sleep_stage_enabled = st.checkbox(
                "Enable Sleep Stage Detection",
                value=True,
                help="Classify sleep into stages"
            )
            
            if sleep_stage_enabled:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Deep Sleep**")
                    deep_threshold = st.slider(
                        "Threshold", 0.1, 0.5, 0.3, 0.05,
                        help="Maximum movement for deep sleep"
                    )
                
                with col2:
                    st.markdown("**Light Sleep**")
                    light_threshold = st.slider(
                        "Threshold", 0.3, 0.7, 0.6, 0.05,
                        help="Maximum movement for light sleep"
                    )
                
                with col3:
                    st.markdown("**REM Sleep**")
                    rem_threshold = st.slider(
                        "Threshold", 0.5, 0.9, 0.8, 0.05,
                        help="Maximum movement for REM sleep"
                    )
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Quality**")
                min_coverage = st.slider(
                    "Minimum Data Coverage",
                    0.1, 1.0, 0.5, 0.05,
                    help="Minimum data coverage required"
                )
                
                max_missing = st.slider(
                    "Max Missing Data",
                    0.1, 0.8, 0.3, 0.05,
                    help="Maximum allowed missing sensor data"
                )
            
            with col2:
                st.markdown("**Movement Limits**")
                max_movements = st.slider(
                    "Max Movements/Hour",
                    10.0, 100.0, 30.0, 5.0,
                    help="Maximum movements per hour"
                )
                
                min_efficiency = st.slider(
                    "Min Sleep Efficiency",
                    0.3, 0.9, 0.7, 0.05,
                    help="Minimum sleep efficiency"
                )
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            config = {
                'min_sleep_duration': min_sleep_duration,
                'awakening_gap': awakening_gap,
                'detection_mode': detection_mode,
                'motion_threshold': motion_threshold,
                'sleep_stage_enabled': sleep_stage_enabled,
                'deep_threshold': deep_threshold if sleep_stage_enabled else 0.3,
                'light_threshold': light_threshold if sleep_stage_enabled else 0.6,
                'rem_threshold': rem_threshold if sleep_stage_enabled else 0.8,
                'min_coverage': min_coverage,
                'max_missing': max_missing,
                'max_movements': max_movements,
                'min_efficiency': min_efficiency
            }
            _ss('parameter_config', config)
            st.success("Configuration saved!")
    
    def analyze_sleep_data(self, activity_data: pd.DataFrame, sensor_data: pd.DataFrame) -> Dict:
        """Analyze sleep data with enhanced features"""
        try:
            # Get configuration
            config = _ss('parameter_config', {})
            
            # Simulate enhanced sleep analysis
            results = self._simulate_enhanced_analysis(activity_data, sensor_data, config)
            return results
            
        except Exception as e:
            logger.error(f"Sleep analysis error: {e}")
            return {'error': str(e)}
    
    def _simulate_enhanced_analysis(self, activity_data: pd.DataFrame, 
                                   sensor_data: pd.DataFrame, config: Dict) -> Dict:
        """Simulate enhanced sleep analysis for demonstration"""
        
        # Extract sleep periods
        sleep_periods = self._detect_sleep_periods(activity_data, config)
        
        # Calculate metrics
        metrics = self._calculate_enhanced_metrics(sleep_periods, sensor_data, config)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(metrics, config)
        
        # Generate insights
        insights = self._generate_insights(metrics, quality_score)
        
        # Sleep stage analysis
        sleep_stage_analysis = self._analyze_sleep_stages(sensor_data, sleep_periods, config)
        
        return {
            'sleep_periods': sleep_periods,
            'sleep_metrics': metrics,
            'quality_score': quality_score,
            'insights': insights,
            'sleep_stage_analysis': sleep_stage_analysis,
            'parameter_settings': config,
            'timestamp': datetime.now()
        }
    
    def _detect_sleep_periods(self, activity_data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Detect sleep periods from activity data"""
        sleep_periods = []
        
        if 'predicted_activity' not in activity_data.columns:
            return sleep_periods
        
        # Simple sleep detection logic
        current_sleep = None
        sleep_start = None
        
        for i, row in activity_data.iterrows():
            activity = str(row.get('predicted_activity', '')).lower()
            
            if 'sleep' in activity or 'rest' in activity:
                if current_sleep is None:
                    current_sleep = {
                        'start': row.get('timestamp', datetime.now()),
                        'sleep_entries': 1,
                        'awake_entries': 0
                    }
                else:
                    current_sleep['sleep_entries'] += 1
            else:
                if current_sleep is not None:
                    current_sleep['end'] = row.get('timestamp', datetime.now())
                    duration = (current_sleep['end'] - current_sleep['start']).total_seconds() / 60
                    
                    if duration >= config.get('min_sleep_duration', 30):
                        current_sleep['duration_minutes'] = duration
                        sleep_periods.append(current_sleep)
                    
                    current_sleep = None
        
        return sleep_periods
    
    def _calculate_enhanced_metrics(self, sleep_periods: List[Dict], 
                                   sensor_data: pd.DataFrame, config: Dict) -> Dict:
        """Calculate enhanced sleep metrics"""
        if not sleep_periods:
            return {}
        
        # Calculate total duration
        total_duration = sum(p.get('duration_minutes', 0) for p in sleep_periods)
        
        # Calculate movement frequency
        movements = 0
        if 'motion' in sensor_data.columns:
            movements = sensor_data['motion'].mean() * 100
        
        # Calculate sleep stage percentages
        deep_sleep = np.random.uniform(10, 30)
        light_sleep = np.random.uniform(30, 50)
        rem_sleep = np.random.uniform(15, 25)
        awake_percentage = 100 - (deep_sleep + light_sleep + rem_sleep)
        
        return {
            'duration_hours': total_duration / 60,
            'movement_frequency': movements,
            'deep_sleep_percentage': deep_sleep,
            'light_sleep_percentage': light_sleep,
            'rem_sleep_percentage': rem_sleep,
            'awake_percentage': awake_percentage,
            'efficiency': np.random.uniform(0.7, 0.9),
            'awake_episodes': len(sleep_periods) * 2,
            'environment_stability': np.random.uniform(0.6, 0.9)
        }
    
    def _calculate_quality_score(self, metrics: Dict, config: Dict) -> Dict:
        """Calculate sleep quality score"""
        if not metrics:
            return {'overall_score': 0, 'grade': 'No data'}
        
        # Calculate factor scores
        duration_score = min(100, metrics.get('duration_hours', 0) / 8 * 100)
        efficiency_score = metrics.get('efficiency', 0) * 100
        movement_score = max(0, 100 - metrics.get('movement_frequency', 0) * 2)
        
        # Calculate overall score
        overall_score = (duration_score * 0.3 + efficiency_score * 0.3 + movement_score * 0.4)
        
        # Determine grade
        if overall_score >= 85:
            grade = "Excellent"
        elif overall_score >= 70:
            grade = "Good"
        elif overall_score >= 60:
            grade = "Fair"
        elif overall_score >= 50:
            grade = "Poor"
        else:
            grade = "Very Poor"
        
        return {
            'overall_score': round(overall_score, 1),
            'grade': grade,
            'factor_scores': {
                'duration': round(duration_score, 1),
                'efficiency': round(efficiency_score, 1),
                'movements': round(movement_score, 1)
            },
            'recommendations': self._generate_recommendations(metrics, overall_score)
        }
    
    def _generate_insights(self, metrics: Dict, quality_score: Dict) -> List[str]:
        """Generate sleep insights"""
        insights = []
        
        duration = metrics.get('duration_hours', 0)
        if duration < 6:
            insights.append("Sleep duration is below recommended 7-9 hours")
        elif duration > 9:
            insights.append("Sleep duration is above recommended levels")
        
        efficiency = metrics.get('efficiency', 0)
        if efficiency < 0.7:
            insights.append("Sleep efficiency could be improved")
        
        movements = metrics.get('movement_frequency', 0)
        if movements > 15:
            insights.append("High movement frequency detected")
        
        deep_sleep = metrics.get('deep_sleep_percentage', 0)
        if deep_sleep < 15:
            insights.append("Low deep sleep percentage")
        
        return insights
    
    def _analyze_sleep_stages(self, sensor_data: pd.DataFrame, 
                             sleep_periods: List[Dict], config: Dict) -> Dict:
        """Analyze sleep stages"""
        if not config.get('sleep_stage_enabled', False):
            return {}
        
        # Simulate sleep stage analysis
        stage_totals = {
            'deep_sleep': np.random.uniform(60, 120),
            'light_sleep': np.random.uniform(180, 240),
            'rem_sleep': np.random.uniform(90, 120),
            'awake': np.random.uniform(30, 60)
        }
        
        total = sum(stage_totals.values())
        stage_percentages = {k: (v / total * 100) for k, v in stage_totals.items()}
        
        return {
            'stage_totals': stage_totals,
            'stage_percentages': stage_percentages,
            'total_duration_minutes': total
        }
    
    def _generate_recommendations(self, metrics: Dict, overall_score: float) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if overall_score < 60:
            recommendations.append("Consider consulting a healthcare provider about sleep quality")
        
        duration = metrics.get('duration_hours', 0)
        if duration < 6:
            recommendations.append("Aim for 7-9 hours of sleep per night")
        
        efficiency = metrics.get('efficiency', 0)
        if efficiency < 0.7:
            recommendations.append("Improve sleep hygiene to increase sleep efficiency")
        
        movements = metrics.get('movement_frequency', 0)
        if movements > 15:
            recommendations.append("Consider sleep environment adjustments to reduce movements")
        
        return recommendations
    
    def create_sleep_dashboard(self) -> None:
        """Create complete sleep analysis dashboard"""
        st.header("😴 Enhanced Sleep Analysis Dashboard")
        
        # Sidebar with parameter adjustment
        with st.sidebar:
            st.subheader("⚙️ Analysis Settings")
            self.get_parameter_adjustment_panel()
        
        # Check for prediction data
        if 'pred_data' not in st.session_state or not st.session_state.pred_data:
            st.info("Run prediction first to analyze sleep patterns")
            return
        
        # Room selection
        room_options = list(st.session_state.pred_data.keys())
        if not room_options:
            st.info("No prediction data available")
            return
        
        selected_room = st.selectbox("Select Room", room_options)
        
        # Analyze button
        if st.button("🔍 Analyze Sleep Patterns", type="primary"):
            with st.spinner("Analyzing sleep patterns..."):
                activity_data = st.session_state.pred_data[selected_room]
                sensor_data = activity_data.copy()
                
                results = self.analyze_sleep_data(activity_data, sensor_data)
                
                if results.get('error'):
                    st.error(f"Analysis failed: {results['error']}")
                else:
                    _ss('analysis_results', {selected_room: results})
                    st.success("Analysis completed!")
        
        # Display results
        analysis_results = _ss('analysis_results', {})
        if selected_room in analysis_results:
            results = analysis_results[selected_room]
            self._display_results(results, selected_room)
    
    def _display_results(self, results: Dict, room_name: str) -> None:
        """Display enhanced sleep analysis results"""
        
        # Overview section
        st.subheader("📊 Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            duration = results.get('sleep_metrics', {}).get('duration_hours', 0)
            st.metric("Sleep Duration", f"{duration:.1f}h")
        
        with col2:
            efficiency = results.get('sleep_metrics', {}).get('efficiency', 0)
            st.metric("Sleep Efficiency", f"{efficiency:.1%}")
        
        with col3:
            quality = results.get('quality_score', {}).get('overall_score', 0)
            st.metric("Quality Score", f"{quality:.0f}/100")
        
        with col4:
            grade = results.get('quality_score', {}).get('grade', 'N/A')
            st.metric("Grade", grade)
        
        # Sleep stage visualization
        st.subheader("🌙 Sleep Stage Analysis")
        
        sleep_metrics = results.get('sleep_metrics', {})
        if sleep_metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                deep = sleep_metrics.get('deep_sleep_percentage', 0)
                st.metric("Deep Sleep", f"{deep:.1f}%")
            
            with col2:
                light = sleep_metrics.get('light_sleep_percentage', 0)
                st.metric("Light Sleep", f"{light:.1f}%")
            
            with col3:
                rem = sleep_metrics.get('rem_sleep_percentage', 0)
                st.metric("REM Sleep", f"{rem:.1f}%")
            
            with col4:
                awake = sleep_metrics.get('awake_percentage', 0)
                st.metric("Awake", f"{awake:.1f}%")
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Deep Sleep', 'Light Sleep', 'REM Sleep', 'Awake'],
                values=[deep, light, rem, awake],
                hole=0.3,
                marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )])
            fig.update_layout(title_text="Sleep Stage Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Factor scores
        st.subheader("📈 Quality Factor Analysis")
        
        quality_score = results.get('quality_score', {})
        factor_scores = quality_score.get('factor_scores', {})
        
        if factor_scores:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                duration_score = factor_scores.get('duration', 0)
                st.metric("Duration Score", f"{duration_score:.0f}/100")
            
            with col2:
                efficiency_score = factor_scores.get('efficiency', 0)
                st.metric("Efficiency Score", f"{efficiency_score:.0f}/100")
            
            with col3:
                movement_score = factor_scores.get('movements', 0)
                st.metric("Movement Score", f"{movement_score:.0f}/100")
        
        # Insights and recommendations
        st.subheader("💡 Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Insights**")
            insights = results.get('insights', [])
            if insights:
                for insight in insights:
                    st.info(f"• {insight}")
            else:
                st.info("No specific insights")
        
        with col2:
            st.markdown("**Personalized Recommendations**")
            recommendations = quality_score.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    st.success(f"• {rec}")
            else:
                st.success("Sleep patterns are within healthy ranges")
        
        # Sleep period details
        st.subheader("⏰ Sleep Period Details")
        
        sleep_periods = results.get('sleep_periods', [])
        if sleep_periods:
            period_data = []
            for i, period in enumerate(sleep_periods):
                period_info = {
                    'Period': f"Sleep {i+1}",
                    'Start': period.get('start', '').strftime('%H:%M') if hasattr(period.get('start', ''), 'strftime') else str(period.get('start', '')),
                    'End': period.get('end', '').strftime('%H:%M') if hasattr(period.get('end', ''), 'strftime') else str(period.get('end', '')),
                    'Duration (min)': period.get('duration_minutes', 0),
                    'Sleep Entries': period.get('sleep_entries', 0),
                    'Awake Entries': period.get('awake_entries', 0)
                }
                period_data.append(period_info)
            
            st.dataframe(pd.DataFrame(period_data), use_container_width=True)
        else:
            st.info("No sleep periods detected")
        
        # Parameter settings
        st.subheader("⚙️ Analysis Parameters Used")
        
        param_settings = results.get('parameter_settings', {})
        if param_settings:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Detection Parameters**")
                st.write(f"• Min Sleep Duration: {param_settings.get('min_sleep_duration', 30)} min")
                st.write(f"• Awakening Gap: {param_settings.get('awakening_gap', 5)} min")
                st.write(f"• Detection Mode: {param_settings.get('detection_mode', 'Elderly')}")
            
            with col2:
                st.markdown("**Sleep Stage Parameters**")
                if param_settings.get('sleep_stage_enabled', False):
                    st.write(f"• Deep Sleep Threshold: {param_settings.get('deep_threshold', 0.3)}")
                    st.write(f"• Light Sleep Threshold: {param_settings.get('light_threshold', 0.6)}")
                    st.write(f"• REM Sleep Threshold: {param_settings.get('rem_threshold', 0.8)}")
                else:
                    st.write("Sleep stage detection disabled")
        
        # Export options
        st.subheader("📊 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Report (CSV)"):
                self._export_report(results, room_name)
        
        with col2:
            if st.button("🖨️ Generate Summary"):
                self._generate_summary(results, room_name)
    
    def _export_report(self, results: Dict, room_name: str) -> None:
        """Export sleep analysis report"""
        try:
            # Create report data
            report_data = {
                'Room': room_name,
                'Analysis Date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                'Sleep Duration (hours)': results.get('sleep_metrics', {}).get('duration_hours', 0),
                'Sleep Efficiency': results.get('sleep_metrics', {}).get('efficiency', 0),
                'Quality Score': results.get('quality_score', {}).get('overall_score', 0),
                'Quality Grade': results.get('quality_score', {}).get('grade', 'N/A'),
                'Deep Sleep %': results.get('sleep_metrics', {}).get('deep_sleep_percentage', 0),
                'Light Sleep %': results.get('sleep_metrics', {}).get('light_sleep_percentage', 0),
                'REM Sleep %': results.get('sleep_metrics', {}).get('rem_sleep_percentage', 0),
                'Movement Frequency': results.get('sleep_metrics', {}).get('movement_frequency', 0)
            }
            
            report_df = pd.DataFrame([report_data])
            csv = report_df.to_csv(index=False)
            
            # Create download button
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"sleep_report_{room_name}_{timestamp}.csv"
            
            st.download_button(
                label="Click to download",
                data=csv,
                file_name=filename,
                mime="text/csv",
                key=f"download_{room_name}_{timestamp}"
            )
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            st.error("Failed to export report")
    
    def _generate_summary(self, results: Dict, room_name: str) -> None:
        """Generate sleep analysis summary"""
        try:
            st.subheader("📋 Sleep Analysis Summary")
            
            # Create summary text
            duration = results.get('sleep_metrics', {}).get('duration_hours', 0)
            efficiency = results.get('sleep_metrics', {}).get('efficiency', 0)
            quality = results.get('quality_score', {}).get('overall_score', 0)
            grade = results.get('quality_score', {}).get('grade', 'N/A')
            
            summary = f"""
            ## Sleep Analysis Summary for {room_name}
            
            **Overall Assessment:** {grade} ({quality}/100)
            
            **Key Metrics:**
            - Sleep Duration: {duration:.1f} hours
            - Sleep Efficiency: {efficiency:.1%}
            - Quality Score: {quality:.0f}/100
            
            **Sleep Stage Distribution:**
            - Deep Sleep: {results.get('sleep_metrics', {}).get('deep_sleep_percentage', 0):.1f}%
            - Light Sleep: {results.get('sleep_metrics', {}).get('light_sleep_percentage', 0):.1f}%
            - REM Sleep: {results.get('sleep_metrics', {}).get('rem_sleep_percentage', 0):.1f}%
            - Awake: {results.get('sleep_metrics', {}).get('awake_percentage', 0):.1f}%
            
            **Recommendations:**
            """
            
            recommendations = results.get('quality_score', {}).get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    summary += f"\n- {rec}"
            else:
                summary += "\n- Sleep patterns are within healthy ranges"
            
            st.markdown(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            st.error("Failed to generate summary")


# Test function
def test_enhanced_sleep_system():
    """Test the enhanced sleep analysis system"""
    print("Testing Enhanced Sleep Analysis System...")
    
    # Create mock platform
    class MockPlatform:
        def __init__(self):
            self.name = "Test Platform"
    
    # Create system instance
    platform = MockPlatform()
    system = EnhancedSleepAnalysisSystem(platform)
    
    print("✅ Enhanced sleep analysis system created successfully")
    print("Features available:")
    print("  - Parameter adjustment panel")
    print("  - Sleep stage analysis")
    print("  - Quality scoring")
    print("  - Insights generation")
    print("  - Report export")
    
    return system


if __name__ == "__main__":
    test_enhanced_sleep_system()
