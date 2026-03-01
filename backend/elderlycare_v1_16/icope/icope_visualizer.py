"""
ICOPE Visualization Module
Creates radar charts and visualizations for aging assessment
"""

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict, List, Any
import numpy as np


class ICOPE_Visualizer:
    """Creates visualizations for ICOPE aging assessment"""
    
    def __init__(self):
        # Color scheme matching existing app
        self.colors = {
            'primary': '#4A90E2',      # Main blue
            'trend_positive': '#7ED321', # Green for positive trends
            'trend_negative': '#d62728', # Red for negative trends
            'neutral': '#ff7f0e',       # Orange for neutral
            'background': '#f8f9fa',    # Light background
            'text': '#333333'           # Dark text
        }
        
    def create_radar_chart(self, icope_data: Dict, show_trends: bool = True) -> go.Figure:
        """
        Create Plotly radar chart for ICOPE domains
        
        Args:
            icope_data: Dictionary with domain scores and trends
            show_trends: Whether to show trend lines
            
        Returns:
            Plotly Figure object
        """
        # Extract domain names and scores
        domains = list(icope_data.keys())
        scores = [icope_data[d]['score'] for d in domains]
        trends = [icope_data[d]['trend'] for d in domains]
        
        # Capitalize domain names for display
        display_domains = [d.capitalize() for d in domains]
        
        # Create radar chart
        fig = go.Figure()
        
        # Add main domain scores (filled area)
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=display_domains,
            fill='toself',
            name='Current Scores',
            line=dict(color=self.colors['primary'], width=3),
            fillcolor=self._adjust_alpha(self.colors['primary'], 0.3),
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}/100<extra></extra>'
        ))
        
        # Add trend lines if requested
        if show_trends and any(t != 0 for t in trends):
            # Calculate projected scores (current + trend)
            projected_scores = [min(100, max(0, s + t)) for s, t in zip(scores, trends)]
            
            # Add trend line
            fig.add_trace(go.Scatterpolar(
                r=projected_scores,
                theta=display_domains,
                name='Trend Projection',
                line=dict(
                    color=self.colors['trend_positive'] if np.mean(trends) > 0 else self.colors['trend_negative'],
                    width=2,
                    dash='dash'
                ),
                hovertemplate='<b>%{theta}</b><br>Projected: %{r:.1f}/100<extra></extra>'
            ))
            
            # Add trend markers
            for i, (domain, trend) in enumerate(zip(display_domains, trends)):
                if trend != 0:
                    # Add trend arrow annotation
                    arrow_symbol = '↑' if trend > 0 else '↓'
                    arrow_color = self.colors['trend_positive'] if trend > 0 else self.colors['trend_negative']
                    
                    fig.add_annotation(
                        x=domain,
                        y=scores[i],
                        text=f"{arrow_symbol}{abs(trend):.1f}",
                        showarrow=False,
                        font=dict(size=10, color=arrow_color),
                        xanchor='center',
                        yanchor='bottom'
                    )
        
        # Add score markers
        for i, (domain, score) in enumerate(zip(display_domains, scores)):
            fig.add_annotation(
                x=domain,
                y=score,
                text=f"{score:.0f}",
                showarrow=False,
                font=dict(size=12, color=self.colors['text']),
                xanchor='center',
                yanchor='middle',
                bgcolor='white',
                bordercolor=self.colors['primary'],
                borderwidth=1,
                borderpad=2
            )
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['0', '25', '50', '75', '100'],
                    tickfont=dict(size=10),
                    gridcolor='lightgray',
                    linecolor='gray'
                ),
                angularaxis=dict(
                    gridcolor='lightgray',
                    linecolor='gray',
                    rotation=90  # Start at top
                ),
                bgcolor=self.colors['background']
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            title=dict(
                text="ICOPE Aging Assessment Radar Chart",
                font=dict(size=18, color=self.colors['text']),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_trend_chart(self, historical_data: Dict) -> go.Figure:
        """
        Create line chart showing domain trends over time
        
        Args:
            historical_data: Dictionary with historical scores by date
            
        Returns:
            Plotly Figure object
        """
        if not historical_data:
            return self._create_empty_chart("No historical data available")
        
        # Prepare data for plotting
        dates = sorted(historical_data.keys())
        domains = list(next(iter(historical_data.values())).keys())
        
        # Create figure
        fig = go.Figure()
        
        # Add line for each domain
        colors = px.colors.qualitative.Set3
        for i, domain in enumerate(domains):
            domain_scores = [historical_data[date].get(domain, 0) for date in dates]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=domain_scores,
                mode='lines+markers',
                name=domain.capitalize(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{domain.capitalize()}</b><br>Date: %{{x}}<br>Score: %{{y:.1f}}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="ICOPE Domain Trends Over Time",
                font=dict(size=16, color=self.colors['text'])
            ),
            xaxis=dict(
                title="Date",
                gridcolor='lightgray',
                tickangle=45
            ),
            yaxis=dict(
                title="Score (0-100)",
                range=[0, 100],
                gridcolor='lightgray'
            ),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='white',
            plot_bgcolor='white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_domain_comparison_chart(self, icope_data: Dict) -> go.Figure:
        """
        Create bar chart comparing domain scores
        
        Args:
            icope_data: Dictionary with domain scores
            
        Returns:
            Plotly Figure object
        """
        domains = list(icope_data.keys())
        scores = [icope_data[d]['score'] for d in domains]
        trends = [icope_data[d]['trend'] for d in domains]
        
        # Capitalize domain names
        display_domains = [d.capitalize() for d in domains]
        
        # Determine bar colors based on scores
        bar_colors = []
        for score in scores:
            if score >= 80:
                bar_colors.append(self.colors['trend_positive'])
            elif score >= 60:
                bar_colors.append(self.colors['primary'])
            elif score >= 40:
                bar_colors.append(self.colors['neutral'])
            else:
                bar_colors.append(self.colors['trend_negative'])
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=display_domains,
                y=scores,
                marker_color=bar_colors,
                text=[f"{s:.1f}" for s in scores],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}/100<extra></extra>'
            )
        ])
        
        # Add trend annotations
        for i, (domain, trend) in enumerate(zip(display_domains, trends)):
            if trend != 0:
                arrow_symbol = '↑' if trend > 0 else '↓'
                arrow_color = self.colors['trend_positive'] if trend > 0 else self.colors['trend_negative']
                
                fig.add_annotation(
                    x=domain,
                    y=scores[i],
                    text=f"{arrow_symbol}{abs(trend):.1f}",
                    showarrow=False,
                    font=dict(size=10, color=arrow_color),
                    xanchor='center',
                    yanchor='bottom',
                    yshift=10
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="ICOPE Domain Score Comparison",
                font=dict(size=16, color=self.colors['text'])
            ),
            xaxis=dict(
                title="Domain",
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Score (0-100)",
                range=[0, 100],
                gridcolor='lightgray'
            ),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_aging_score_gauge(self, overall_score: float) -> go.Figure:
        """
        Create gauge chart for overall aging score
        
        Args:
            overall_score: Overall ICOPE aging score (0-100)
            
        Returns:
            Plotly Figure object
        """
        # Determine gauge color
        if overall_score >= 80:
            gauge_color = self.colors['trend_positive']
            level = "Excellent"
        elif overall_score >= 70:
            gauge_color = self.colors['primary']
            level = "Good"
        elif overall_score >= 60:
            gauge_color = self.colors['neutral']
            level = "Moderate"
        elif overall_score >= 50:
            gauge_color = '#F5A623'  # Orange
            level = "Concerning"
        else:
            gauge_color = self.colors['trend_negative']
            level = "Critical"
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Aging Score", 'font': {'size': 16}},
            number={'suffix': "/100", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#f8f9fa'},
                    {'range': [50, 60], 'color': '#f0f0f0'},
                    {'range': [60, 70], 'color': '#e8e8e8'},
                    {'range': [70, 80], 'color': '#e0e0e0'},
                    {'range': [80, 100], 'color': '#d8d8d8'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        # Add level annotation
        fig.add_annotation(
            x=0.5,
            y=0.3,
            text=f"Level: {level}",
            showarrow=False,
            font=dict(size=14, color=gauge_color),
            xanchor='center'
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='white'
        )
        
        return fig
    
    def _adjust_alpha(self, color: str, alpha: float) -> str:
        """
        Adjust color alpha (transparency)
        
        Args:
            color: Hex color string
            alpha: Alpha value (0-1)
            
        Returns:
            RGBA color string
        """
        # Convert hex to RGB
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=14, color=self.colors['text']),
            xanchor='center',
            yanchor='middle'
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig


# Streamlit helper functions
def display_icope_dashboard(icope_data: Dict, historical_data: Dict = None):
    """
    Display complete ICOPE dashboard in Streamlit
    
    Args:
        icope_data: Current ICOPE domain scores
        historical_data: Historical scores for trend analysis
    """
    visualizer = ICOPE_Visualizer()
    
    # Overall score gauge
    st.subheader("🌀 Overall Aging Assessment")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        overall_score = sum(d['score'] * d['weight'] for d in icope_data.values()) / sum(d['weight'] for d in icope_data.values())
        gauge_fig = visualizer.create_aging_score_gauge(overall_score)
        st.plotly_chart(gauge_fig, width='stretch')
    
    # Radar chart
    st.subheader("📊 ICOPE Domain Radar Chart")
    radar_fig = visualizer.create_radar_chart(icope_data, show_trends=True)
    st.plotly_chart(radar_fig, width='stretch')
    
    # Domain comparison
    st.subheader("📈 Domain Score Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        bar_fig = visualizer.create_domain_comparison_chart(icope_data)
        st.plotly_chart(bar_fig, width='stretch')
    
    # Trend chart if historical data available
    if historical_data:
        with col2:
            trend_fig = visualizer.create_trend_chart(historical_data)
            st.plotly_chart(trend_fig, width='stretch')
    else:
        with col2:
            st.info("Historical trend data will appear here as more assessments are completed.")
    
    # Domain details
    st.subheader("🔍 Domain Details")
    for domain, data in icope_data.items():
        with st.expander(f"{domain.capitalize()} - {data['score']:.1f}/100", expanded=False):
            st.markdown(f"**Description:** {data.get('description', 'No description available')}")
            st.markdown(f"**Score:** {data['score']:.1f}/100")
            
            if data['trend'] != 0:
                trend_text = f"↑ {data['trend']:.1f}" if data['trend'] > 0 else f"↓ {abs(data['trend']):.1f}"
                trend_color = "green" if data['trend'] > 0 else "red"
                st.markdown(f"**Trend:** :{trend_color}[{trend_text}]")
            else:
                st.markdown("**Trend:** Stable")
            
            if 'indicators' in data:
                st.markdown("**Key Indicators:**")
                for indicator in data['indicators']:
                    st.markdown(f"- {indicator.replace('_', ' ').title()}")


if __name__ == "__main__":
    # Test the visualizer
    visualizer = ICOPE_Visualizer()
    
    # Sample data
    sample_icope_data = {
        'locomotion': {'score': 75.5, 'trend': 2.3, 'weight': 0.25},
        'cognition': {'score': 82.1, 'trend': -1.2, 'weight': 0.20},
        'psychological': {'score': 68.7, 'trend': 0.5, 'weight': 0.15},
        'sensory': {'score': 90.2, 'trend': 0, 'weight': 0.15},
        'vitality': {'score': 72.8, 'trend': 3.1, 'weight': 0.25}
    }
    
    # Test radar chart
    radar_fig = visualizer.create_radar_chart(sample_icope_data)
    radar
