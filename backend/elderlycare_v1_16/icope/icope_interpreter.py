"""
ICOPE Interpretation Module
Provides explanations and implications for aging assessment
"""

import streamlit as st
from typing import Dict, List, Any
from datetime import datetime


class ICOPE_Interpreter:
    """Provides explanations and implications for ICOPE aging assessment"""
    
    def __init__(self):
        # ICOPE framework explanation
        self.icope_explanation = {
            'overview': """
            **ICOPE (Integrated Care for Older People)** is a WHO framework that assesses aging through 
            five key domains of intrinsic capacity. It provides a comprehensive view of an individual's 
            aging trajectory and identifies areas for intervention.
            """,
            'purpose': """
            The ICOPE assessment helps:
            - Monitor aging trajectory across multiple domains
            - Identify early signs of functional decline
            - Guide personalized interventions
            - Track effectiveness of care strategies
            - Support healthy aging goals
            """,
            'domains': {
                'locomotion': """
                **Locomotion** assesses mobility, balance, and physical function. 
                This domain is critical for maintaining independence and preventing falls.
                
                **Clinical Significance:** 
                - Scores below 60 may indicate increased fall risk
                - Declining trends suggest progressive mobility issues
                - Early intervention can prevent functional decline
                """,
                'cognition': """
                **Cognition** evaluates memory, attention, and executive function.
                This domain is essential for daily decision-making and independence.
                
                **Clinical Significance:**
                - Scores below 65 may indicate cognitive impairment risk
                - Rapid declines suggest need for cognitive assessment
                - Cognitive stimulation can slow decline
                """,
                'psychological': """
                **Psychological** assesses mood, anxiety, and psychological well-being.
                Mental health significantly impacts overall quality of life.
                
                **Clinical Significance:**
                - Scores below 60 may indicate depression or anxiety
                - Psychological distress affects physical health
                - Early intervention improves outcomes
                """,
                'sensory': """
                **Sensory** evaluates vision, hearing, and sensory function.
                Sensory health affects communication, safety, and social engagement.
                
                **Clinical Significance:**
                - Scores below 70 may indicate sensory impairment
                - Sensory loss increases isolation and accident risk
                - Corrective measures can significantly improve function
                """,
                'vitality': """
                **Vitality** assesses nutrition, energy, and overall vitality.
                This domain reflects overall physiological resilience.
                
                **Clinical Significance:**
                - Scores below 65 may indicate nutritional or energy deficits
                - Low vitality affects all other domains
                - Holistic interventions can improve vitality
                """
            }
        }
        
        # Interpretation guidelines
        self.interpretation_guidelines = {
            'score_ranges': {
                'excellent': (80, 100, "🟢 Excellent aging trajectory"),
                'good': (70, 80, "🟡 Good aging trajectory"),
                'moderate': (60, 70, "🟠 Moderate aging trajectory"),
                'concerning': (50, 60, "🔴 Concerning aging trajectory"),
                'critical': (0, 50, "🔴 Critical aging trajectory - intervention needed")
            },
            'trend_interpretation': {
                'significant_improvement': (5, float('inf'), "Significant improvement"),
                'moderate_improvement': (2, 5, "Moderate improvement"),
                'stable': (-2, 2, "Stable"),
                'moderate_decline': (-5, -2, "Moderate decline"),
                'significant_decline': (-float('inf'), -5, "Significant decline")
            }
        }
    
    def display_icope_explanation_panel(self):
        """Display comprehensive ICOPE explanation panel"""
        st.subheader("📚 ICOPE Framework Explanation")
        
        with st.expander("🌐 What is ICOPE?", expanded=True):
            st.markdown(self.icope_explanation['overview'])
            st.markdown(self.icope_explanation['purpose'])
        
        with st.expander("🧩 ICOPE Domains Explained"):
            for domain, explanation in self.icope_explanation['domains'].items():
                st.markdown(f"### {domain.capitalize()}")
                st.markdown(explanation)
        
        with st.expander("📊 How to Interpret Scores"):
            st.markdown("### Score Interpretation Guidelines")
            
            for level, (min_score, max_score, description) in self.interpretation_guidelines['score_ranges'].items():
                st.markdown(f"**{description}** ({min_score}-{max_score}):")
                st.markdown(f"- {self._get_score_implications(level)}")
            
            st.markdown("### Trend Interpretation")
            for trend, (min_val, max_val, description) in self.interpretation_guidelines['trend_interpretation'].items():
                st.markdown(f"**{description}** ({min_val:+} to {max_val:+} points):")
                st.markdown(f"- {self._get_trend_implications(trend)}")
    
    def interpret_domain_score(self, domain: str, score: float, trend: float) -> Dict[str, Any]:
        """
        Interpret individual domain score
        
        Args:
            domain: Domain name
            score: Domain score (0-100)
            trend: Trend value
            
        Returns:
            Dictionary with interpretation
        """
        interpretation = {
            'domain': domain,
            'score': score,
            'trend': trend,
            'level': self._get_score_level(score),
            'trend_level': self._get_trend_level(trend),
            'implications': self._get_domain_implications(domain, score),
            'recommendations': self._get_domain_recommendations(domain, score, trend),
            'clinical_notes': self._get_clinical_notes(domain, score)
        }
        
        return interpretation
    
    def interpret_overall_assessment(self, icope_data: Dict) -> Dict[str, Any]:
        """
        Interpret overall ICOPE assessment
        
        Args:
            icope_data: Dictionary with all domain scores
            
        Returns:
            Dictionary with overall interpretation
        """
        # Calculate overall score
        total_score = 0
        total_weight = 0
        for domain, data in icope_data.items():
            total_score += data['score'] * data['weight']
            total_weight += data['weight']
        
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        # Identify strongest and weakest domains
        domains_by_score = sorted(icope_data.items(), key=lambda x: x[1]['score'])
        weakest_domain = domains_by_score[0] if domains_by_score else None
        strongest_domain = domains_by_score[-1] if domains_by_score else None
        
        # Identify improving and declining domains
        improving_domains = [(d, data) for d, data in icope_data.items() if data['trend'] > 2]
        declining_domains = [(d, data) for d, data in icope_data.items() if data['trend'] < -2]
        
        interpretation = {
            'overall_score': overall_score,
            'overall_level': self._get_score_level(overall_score),
            'weakest_domain': weakest_domain,
            'strongest_domain': strongest_domain,
            'improving_domains': improving_domains,
            'declining_domains': declining_domains,
            'summary': self._generate_overall_summary(icope_data, overall_score),
            'priority_actions': self._generate_priority_actions(icope_data),
            'monitoring_recommendations': self._generate_monitoring_recommendations(icope_data)
        }
        
        return interpretation
    
    def display_interpretation_dashboard(self, icope_data: Dict):
        """Display complete interpretation dashboard"""
        # Get overall interpretation
        overall_interpretation = self.interpret_overall_assessment(icope_data)
        
        # Overall assessment
        st.subheader("🧭 Overall Aging Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Overall Aging Score",
                f"{overall_interpretation['overall_score']:.1f}/100",
                delta=f"{overall_interpretation['overall_level']}"
            )
        
        with col2:
            if overall_interpretation['weakest_domain']:
                domain, data = overall_interpretation['weakest_domain']
                st.metric(
                    "Area Needing Attention",
                    domain.capitalize(),
                    delta=f"{data['score']:.1f}/100"
                )
        
        # Display summary
        st.markdown("### 📋 Assessment Summary")
        st.markdown(overall_interpretation['summary'])
        
        # Priority actions
        st.markdown("### 🎯 Priority Actions")
        for action in overall_interpretation['priority_actions']:
            st.markdown(f"• {action}")
        
        # Domain-specific interpretations
        st.markdown("### 🔍 Domain-Specific Insights")
        
        for domain, data in icope_data.items():
            domain_interpretation = self.interpret_domain_score(domain, data['score'], data['trend'])
            
            with st.expander(f"{domain.capitalize()} - {data['score']:.1f}/100 ({domain_interpretation['level']})", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Status:**")
                    st.markdown(f"- Score: {data['score']:.1f}/100")
                    st.markdown(f"- Level: {domain_interpretation['level']}")
                    
                    if data['trend'] != 0:
                        trend_text = f"↑ {data['trend']:.1f}" if data['trend'] > 0 else f"↓ {abs(data['trend']):.1f}"
                        trend_color = "green" if data['trend'] > 0 else "red"
                        st.markdown(f"- Trend: :{trend_color}[{trend_text}]")
                    else:
                        st.markdown("- Trend: Stable")
                
                with col2:
                    st.markdown("**Clinical Implications:**")
                    for implication in domain_interpretation['implications']:
                        st.markdown(f"- {implication}")
                
                st.markdown("**Recommendations:**")
                for recommendation in domain_interpretation['recommendations']:
                    st.markdown(f"• {recommendation}")
                
                if domain_interpretation['clinical_notes']:
                    st.markdown("**Clinical Notes:**")
                    st.markdown(domain_interpretation['clinical_notes'])
        
        # Monitoring recommendations
        st.markdown("### 📅 Monitoring Recommendations")
        for recommendation in overall_interpretation['monitoring_recommendations']:
            st.markdown(f"• {recommendation}")
    
    def _get_score_level(self, score: float) -> str:
        """Get score level based on value"""
        for level, (min_score, max_score, description) in self.interpretation_guidelines['score_ranges'].items():
            if min_score <= score < max_score:
                return description
        return "Unknown"
    
    def _get_trend_level(self, trend: float) -> str:
        """Get trend level based on value"""
        for trend_type, (min_val, max_val, description) in self.interpretation_guidelines['trend_interpretation'].items():
            if min_val <= trend < max_val:
                return description
        return "Stable"
    
    def _get_score_implications(self, level: str) -> str:
        """Get implications for score level"""
        implications = {
            'excellent': "Maintain current healthy behaviors and regular monitoring",
            'good': "Continue positive habits with minor adjustments as needed",
            'moderate': "Consider lifestyle modifications and closer monitoring",
            'concerning': "Implement targeted interventions and frequent monitoring",
            'critical': "Immediate intervention and comprehensive assessment needed"
        }
        return implications.get(level, "No specific implications available")
    
    def _get_trend_implications(self, trend: str) -> str:
        """Get implications for trend level"""
        implications = {
            'significant_improvement': "Interventions are highly effective, consider maintaining current approach",
            'moderate_improvement': "Positive progress, continue current strategies",
            'stable': "Maintain current approach, monitor for changes",
            'moderate_decline': "Review current strategies, consider adjustments",
            'significant_decline': "Urgent review needed, consider changing approach"
        }
        return implications.get(trend, "No specific trend implications")
    
    def _get_domain_implications(self, domain: str, score: float) -> List[str]:
        """Get domain-specific implications"""
        implications = {
            'locomotion': [
                "Mobility affects independence and fall risk",
                "Physical activity levels impact overall health",
                "Balance and strength are key for daily activities"
            ],
            'cognition': [
                "Cognitive function affects decision-making and safety",
                "Memory and attention impact daily functioning",
                "Cognitive stimulation supports brain health"
            ],
            'psychological': [
                "Mental health affects quality of life and physical health",
                "Social engagement supports psychological well-being",
                "Stress management impacts overall health"
            ],
            'sensory': [
                "Sensory function affects communication and safety",
                "Vision and hearing impact social engagement",
                "Sensory aids can significantly improve function"
            ],
            'vitality': [
                "Energy levels affect all aspects of functioning",
                "Nutrition impacts physical and cognitive health",
                "Sleep quality affects overall vitality"
            ]
        }
        
        domain_implications = implications.get(domain, [])
        
        # Add score-specific implications
        if score < 60:
            domain_implications.append("Consider professional assessment and intervention")
        elif score < 70:
            domain_implications.append("Monitor closely and consider preventive measures")
        
        return domain_implications
    
    def _get_domain_recommendations(self, domain: str, score: float, trend: float) -> List[str]:
        """Get domain-specific recommendations"""
        recommendations = {
            'locomotion': [
                "Engage in daily walking or light exercise (30 minutes)",
                "Practice balance exercises (e.g., standing on one leg)",
                "Consider strength training 2-3 times per week"
            ],
            'cognition': [
                "Engage in mentally stimulating activities (puzzles, reading)",
                "Practice memory exercises (recall lists, names)",
                "Maintain social connections and conversations"
            ],
            'psychological': [
                "Practice stress management techniques (meditation, deep breathing)",
                "Maintain regular sleep schedule (7-9 hours)",
                "Engage in enjoyable activities and hobbies"
            ],
            'sensory': [
                "Schedule regular vision and hearing check-ups",
                "Use appropriate sensory aids if recommended",
                "Maintain good lighting and reduce background noise"
            ],
            'vitality': [
                "Maintain balanced nutrition with adequate protein",
                "Ensure adequate sleep (7-9 hours per night)",
                "Stay hydrated and monitor energy levels"
            ]
        }
        
        domain_recommendations = recommendations.get(domain, [])
        
        # Add trend-based recommendations
        if trend < -2:
            domain_recommendations.append("Consider more frequent monitoring (weekly)")
            domain_recommendations.append("Discuss with healthcare provider")
        elif trend > 2:
            domain_recommendations.append("Continue current successful strategies")
        
        return domain_recommendations
    
    def _get_clinical_notes(self, domain: str, score: float) -> str:
        """Get clinical notes for domain"""
        if score < 50:
            return f"**Clinical Alert:** {domain.capitalize()} score indicates significant impairment. Professional assessment recommended."
        elif score < 60:
            return f"**Clinical Note:** {domain.capitalize()} score suggests moderate impairment. Monitor closely and consider interventions."
        elif score < 70:
            return f"**Clinical Note:** {domain.capitalize()} score indicates mild concerns. Preventive measures recommended."
        else:
            return f"**Clinical Note:** {domain.capitalize()} score within healthy range. Continue current practices."
    
    def _generate_overall_summary(self, icope_data: Dict, overall_score: float) -> str:
        """Generate overall assessment summary"""
        level = self._get_score_level(overall_score)
        
        summary = f"**Overall Assessment:** {level}\n\n"
        summary += f"**Overall Score:** {overall_score:.1f}/100\n\n"
        
        # Add domain highlights
        domains_by_score = sorted(icope_data.items(), key=lambda x: x[1]['score'])
        
        if domains_by_score:
            weakest = domains_by_score[0]
            strongest = domains_by_score[-1]
            
            summary += f"**Strongest Domain:** {strongest[0].capitalize()} ({strongest[1]['score']:.1f}/100)\n"
            summary += f"**Weakest Domain:** {weakest[0].capitalize()} ({weakest[1]['score']:.1f}/100)\n\n"
        
        # Add trend summary
        improving = [(d, data) for d, data in icope_data.items() if data['trend'] > 2]
        declining = [(d, data) for d, data in icope_data.items() if data['trend'] < -2]
        
        if improving:
            summary += "**Improving Domains:** " + ", ".join([d.capitalize() for d, _ in improving]) + "\n"
        
        if declining:
            summary += "**Declining Domains:** " + ", ".join([d.capitalize() for d, _ in declining]) + "\n"
        
        summary += "\n**Next Assessment Recommended:** " + self._get_next_assessment_date(overall_score)
        
        return summary
    
    def _generate_priority_actions(self, icope_data: Dict) -> List[str]:
        """Generate priority actions based on assessment"""
        actions = []
        
        # Identify domains needing attention
        for domain, data in icope_data.items():
            if data['score'] < 60:
                actions.append(f"Address {domain} concerns (score: {data['score']:.1f}/100)")
            elif data['trend'] < -2:
                actions.append(f"Monitor declining trend in {domain} (trend: {data['trend']:.1f})")
        
        # Add general actions
        if not actions:
            actions.append("Maintain current healthy behaviors")
            actions.append("Continue regular monitoring")
        else:
            actions.append("Schedule follow-up assessment in 1 month")
            actions.append("Consider multidisciplinary consultation")
        
        return actions
    
    def _generate_monitoring_recommendations(self, icope_data: Dict) -> List[str]:
        """Generate monitoring recommendations"""
        recommendations = []
        
        # Determine monitoring frequency based on scores
        low_scores = [d for d, data in icope_data.items() if data['score'] < 60]
        declining_domains = [d for d, data in icope_data.items() if data['trend'] < -2]
        
        if low_scores or declining_domains:
            recommendations.append("Weekly monitoring of key domains")
            recommendations.append("Monthly comprehensive assessment")
        else:
            recommendations.append("Monthly monitoring of all domains")
            recommendations.append("Quarterly comprehensive assessment")
        
        # Domain-specific monitoring
        for domain, data in icope_data.items():
            if data['score'] < 60:
                recommendations.append(f"Daily monitoring of {domain} indicators")
            elif data['trend'] < -2:
                recommendations.append(f"Bi-weekly {domain} trend assessment")
        
        return recommendations
    
    def _get_next_assessment_date(self, overall_score: float) -> str:
        """Get recommended next assessment date"""
        from datetime import datetime, timedelta
        
        if overall_score < 60:
            next_date = datetime.now() + timedelta(days=14)  # 2 weeks
        elif overall_score < 70:
            next_date = datetime.now() + timedelta(days=30)  # 1 month
        elif overall_score < 80:
            next_date = datetime.now() + timedelta(days=60)  # 2 months
        else:
            next_date = datetime.now() + timedelta(days=90)  # 3 months
        
        return next_date.strftime("%B %d, %Y")


if __name__ == "__main__":
    # Test the interpreter
    interpreter = ICOPE_Interpreter()
    
    # Sample data
    sample_icope_data = {
        'locomotion': {'score': 75.5, 'trend': 2.3, 'weight': 0.25},
        'cognition': {'score': 82.1, 'trend': -1.2, 'weight': 0.20},
        'psychological': {'score': 68.7, 'trend': 0.5, 'weight': 0.15},
        'sensory': {'score': 90.2, 'trend': 0, 'weight': 0.15},
        'vitality': {'score': 72.8, 'trend': 3.1, 'weight': 0.25}
    }
    
    print("ICOPE Interpreter Test:")
    print("=" * 50)
    
    # Test overall interpretation
    overall = interpreter.interpret_overall_assessment(sample_icope_data)
    print(f"Overall Score: {overall['overall_score']:.1f}/100")
    print(f"Overall Level: {overall['overall_level']}")
    print(f"\nPriority Actions:")
    for action in overall['priority_actions']:
        print(f"  • {action}")
    
    # Test domain interpretation
    print(f"\nDomain Interpretations:")
    for domain, data in sample_icope_data.items():
        interpretation = interpreter.interpret_domain_score(domain, data['score'], data['trend'])
        print(f"\n  {domain.capitalize()}:")
        print(f"    Level: {interpretation['level']}")
        print(f"    Trend: {interpretation['trend_level']}")
        print(f"    Clinical Notes: {interpretation['clinical_notes']}")
