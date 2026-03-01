"""
ICOPE Processor Module
Main integration module for ICOPE aging assessment
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .icope_scoring import ICOPE_Scoring
from .icope_visualizer import ICOPE_Visualizer, display_icope_dashboard
from .icope_interpreter import ICOPE_Interpreter


class ICOPE_Processor:
    """Main processor for ICOPE aging assessment"""
    
    def __init__(self, platform: Any = None):
        self.platform = platform
        self.scoring = ICOPE_Scoring()
        self.visualizer = ICOPE_Visualizer()
        self.interpreter = ICOPE_Interpreter()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for ICOPE"""
        if 'icope_data' not in st.session_state:
            st.session_state.icope_data = {}
        
        if 'icope_history' not in st.session_state:
            st.session_state.icope_history = {}
        
        if 'icope_last_assessment' not in st.session_state:
            st.session_state.icope_last_assessment = None
    
    def collect_data_from_app(self) -> Dict[str, Any]:
        """
        Collect data from existing app modules
        
        Returns:
            Dictionary with sensor, sleep, and anomaly data
        """
        data = {
            'sensor_data': {},
            'sleep_data': {},
            'anomaly_data': {}
        }
        
        try:
            # Collect sensor data from main app
            if hasattr(self.platform, 'get_sensor_data'):
                data['sensor_data'] = self.platform.get_sensor_data()
            elif 'pred_data' in st.session_state and st.session_state.pred_data:
                # Extract from prediction data
                pred_data = st.session_state.pred_data
                if pred_data and isinstance(pred_data, dict):
                    # Get first room's data
                    first_room = next(iter(pred_data.keys()))
                    room_data = pred_data[first_room]
                    
                    # Extract sensor readings
                    sensor_columns = ['motion', 'temperature', 'humidity', 'co2', 'sound', 'light']
                    for col in sensor_columns:
                        if col in room_data.columns:
                            data['sensor_data'][col] = room_data[col].tolist()
            
            # Collect sleep data
            if 'sleep__analysis_results' in st.session_state:
                sleep_results = st.session_state.sleep__analysis_results
                if sleep_results and isinstance(sleep_results, dict):
                    first_room = next(iter(sleep_results.keys()))
                    data['sleep_data'] = sleep_results[first_room]
            
            # Collect anomaly data (simulated for now)
            data['anomaly_data'] = {
                'anomalies': [],
                'risk_level': 'low'
            }
            
        except Exception as e:
            st.warning(f"Data collection warning: {str(e)}")
        
        return data
    
    def calculate_icope_assessment(self) -> Dict[str, Any]:
        """
        Calculate complete ICOPE assessment
        
        Returns:
            Dictionary with assessment results
        """
        # Collect data
        app_data = self.collect_data_from_app()
        
        # Calculate domain scores
        domain_scores = self.scoring.calculate_domain_scores(
            app_data['sensor_data'],
            app_data['sleep_data'],
            app_data['anomaly_data']
        )
        
        # Determine assessment timestamp (use prediction data time if available)
        assessment_time_dt = datetime.now()
        
        # Try to find timestamp in pred_data (bridged via session state)
        if 'pred_data' in st.session_state and st.session_state.pred_data:
             try:
                 # Check first room's dataframe
                 first_room = next(iter(st.session_state.pred_data.values()))
                 if 'timestamp' in first_room.columns and not first_room.empty:
                     assessment_time_dt = first_room['timestamp'].max()
             except:
                 pass
        
        assessment_time = assessment_time_dt.strftime('%Y-%m-%d %H:%M')
        date_key = assessment_time_dt.strftime('%Y-%m-%d')
        
        # Store in session state
        st.session_state.icope_data = domain_scores
        st.session_state.icope_last_assessment = assessment_time
        
        # Store in history
        if 'icope_history' not in st.session_state:
            st.session_state.icope_history = {}
            
        st.session_state.icope_history[date_key] = {
            domain: data['score'] for domain, data in domain_scores.items()
        }
        
        return {
            'domain_scores': domain_scores,
            'overall_score': self.scoring.get_overall_score(),
            'assessment_time': assessment_time,
            'data_sources': list(app_data.keys())
        }
    
    def create_icope_dashboard(self):
        """Create complete ICOPE dashboard"""
        st.header("🌀 ICOPE Aging Assessment Dashboard")
        
        # Explanation panel
        self.interpreter.display_icope_explanation_panel()
        
        # Check if assessment exists
        if not st.session_state.get('icope_data'):
            st.info("No ICOPE assessment available. Run assessment to see results.")
            
            if st.button("🔍 Run ICOPE Assessment", type="primary"):
                with st.spinner("Calculating ICOPE assessment..."):
                    assessment = self.calculate_icope_assessment()
                    st.success(f"Assessment completed at {assessment['assessment_time']}")
                    st.rerun()
            
            return
        
        # Display assessment info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Last Assessment",
                st.session_state.icope_last_assessment.split()[0]
            )
        
        with col2:
            overall_score = self.scoring.get_overall_score()
            st.metric("Overall Score", f"{overall_score:.1f}/100")
        
        with col3:
            if st.button("🔄 Update Assessment", type="secondary"):
                with st.spinner("Updating assessment..."):
                    self.calculate_icope_assessment()
                    st.rerun()
        
        # Visualizations
        display_icope_dashboard(
            st.session_state.icope_data,
            st.session_state.icope_history
        )
        
        # Interpretation
        self.interpreter.display_interpretation_dashboard(st.session_state.icope_data)
        
        # Export options
        st.subheader("📤 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Assessment Report"):
                self._export_assessment_report()
        
        with col2:
            if st.button("🖨️ Generate Summary", key="icope_generate_summary"):
                self._generate_assessment_summary()
    
    def _export_assessment_report(self):
        """Export assessment report"""
        try:
            # Create report data
            report_data = {
                'Assessment Date': st.session_state.icope_last_assessment,
                'Overall Score': f"{self.scoring.get_overall_score():.1f}/100",
                'Interpretation': self.scoring.get_aging_interpretation()
            }
            
            # Add domain scores
            for domain, data in st.session_state.icope_data.items():
                report_data[f"{domain.capitalize()} Score"] = f"{data['score']:.1f}/100"
                report_data[f"{domain.capitalize()} Trend"] = f"{data['trend']:+.1f}"
            
            # Create DataFrame
            report_df = pd.DataFrame([report_data])
            
            # Convert to CSV
            csv = report_df.to_csv(index=False)
            
            # Create download button
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"icope_assessment_{timestamp}.csv"
            
            st.download_button(
                label="Click to download CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Failed to export report: {str(e)}")
    
    def _generate_assessment_summary(self):
        """Generate assessment summary"""
        try:
            overall_score = self.scoring.get_overall_score()
            interpretation = self.scoring.get_aging_interpretation()
            
            summary = f"""
            # ICOPE Aging Assessment Summary
            
            **Assessment Date:** {st.session_state.icope_last_assessment}
            **Overall Score:** {overall_score:.1f}/100
            **Interpretation:** {interpretation}
            
            ## Domain Scores:
            """
            
            for domain, data in st.session_state.icope_data.items():
                trend_symbol = "↑" if data['trend'] > 0 else "↓" if data['trend'] < 0 else "→"
                summary += f"\n- **{domain.capitalize()}:** {data['score']:.1f}/100 ({trend_symbol}{abs(data['trend']):.1f})"
            
            summary += "\n\n## Recommendations:"
            
            # Get recommendations from interpreter
            overall_interpretation = self.interpreter.interpret_overall_assessment(
                st.session_state.icope_data
            )
            
            for action in overall_interpretation['priority_actions']:
                summary += f"\n- {action}"
            
            st.markdown(summary)
            
        except Exception as e:
            st.error(f"Failed to generate summary: {str(e)}")
    
    def get_icope_insights(self) -> List[str]:
        """Get key insights from ICOPE assessment"""
        insights = []
        
        if not st.session_state.icope_data:
            return ["No assessment data available"]
        
        # Overall insight
        overall_score = self.scoring.get_overall_score()
        if overall_score >= 80:
            insights.append("Excellent overall aging trajectory")
        elif overall_score >= 70:
            insights.append("Good aging trajectory with room for improvement")
        elif overall_score >= 60:
            insights.append("Moderate aging trajectory - monitor closely")
        elif overall_score >= 50:
            insights.append("Concerning aging trajectory - consider interventions")
        else:
            insights.append("Critical aging trajectory - immediate attention needed")
        
        # Domain-specific insights
        for domain, data in st.session_state.icope_data.items():
            if data['score'] < 60:
                insights.append(f"Low {domain} score ({data['score']:.1f}/100) needs attention")
            elif data['trend'] < -2:
                insights.append(f"Declining trend in {domain} ({data['trend']:.1f})")
            elif data['trend'] > 2:
                insights.append(f"Improving trend in {domain} ({data['trend']:+.1f})")
        
        return insights
    
    def get_recommendations(self) -> List[str]:
        """Get personalized recommendations"""
        recommendations = []
        
        if not st.session_state.icope_data:
            return ["Run assessment to get personalized recommendations"]
        
        # Domain-specific recommendations
        for domain, data in st.session_state.icope_data.items():
            if data['score'] < 60:
                recommendations.extend(self.scoring.get_domain_recommendations(domain))
        
        # General recommendations
        overall_score = self.scoring.get_overall_score()
        if overall_score < 70:
            recommendations.append("Consider comprehensive geriatric assessment")
            recommendations.append("Discuss findings with healthcare provider")
        
        if len(recommendations) == 0:
            recommendations.append("Maintain current healthy lifestyle")
            recommendations.append("Continue regular monitoring")
        
        return recommendations[:5]  # Return top 5 recommendations


# Test function
def test_icope_processor():
    """Test the ICOPE processor"""
    print("Testing ICOPE Processor...")
    
    processor = ICOPE_Processor()
    
    # Test data collection
    print("1. Testing data collection...")
    app_data = processor.collect_data_from_app()
    print(f"   Collected data from: {list(app_data.keys())}")
    
    # Test assessment calculation
    print("2. Testing assessment calculation...")
    assessment = processor.calculate_icope_assessment()
    print(f"   Assessment completed: {assessment['assessment_time']}")
    print(f"   Overall score: {assessment['overall_score']:.1f}/100")
    
    # Test insights
    print("3. Testing insights generation...")
    insights = processor.get_icope_insights()
    print(f"   Insights: {len(insights)} generated")
    for insight in insights[:3]:
        print(f"   • {insight}")
    
    # Test recommendations
    print("4. Testing recommendations...")
    recommendations = processor.get_recommendations()
    print(f"   Recommendations: {len(recommendations)} generated")
    for rec in recommendations[:3]:
        print(f"   • {rec}")
    
    print("\n✅ ICOPE Processor test completed successfully!")
    
    return processor


if __name__ == "__main__":
    test_icope_processor()
