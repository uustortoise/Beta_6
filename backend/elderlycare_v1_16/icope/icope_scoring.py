"""
ICOPE Scoring System
Implements WHO ICOPE framework for aging assessment
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class ICOPE_Scoring:
    """ICOPE scoring system for aging assessment"""
    
    def __init__(self):
        # WHO ICOPE domains with weights
        self.domains = {
            'locomotion': {
                'score': 0,
                'weight': 0.25,
                'trend': 0,
                'description': 'Mobility, balance, and physical function',
                'indicators': ['walking_speed', 'balance_time', 'chair_stand']
            },
            'cognition': {
                'score': 0,
                'weight': 0.20,
                'trend': 0,
                'description': 'Memory, attention, and executive function',
                'indicators': ['memory_score', 'attention_span', 'processing_speed']
            },
            'psychological': {
                'score': 0,
                'weight': 0.15,
                'trend': 0,
                'description': 'Mood, anxiety, and psychological well-being',
                'indicators': ['mood_score', 'anxiety_level', 'social_engagement']
            },
            'sensory': {
                'score': 0,
                'weight': 0.15,
                'trend': 0,
                'description': 'Vision, hearing, and sensory function',
                'indicators': ['vision_acuity', 'hearing_threshold', 'tactile_sensitivity']
            },
            'vitality': {
                'score': 0,
                'weight': 0.25,
                'trend': 0,
                'description': 'Nutrition, energy, and overall vitality',
                'indicators': ['nutrition_score', 'energy_level', 'sleep_quality']
            }
        }
        
        # Historical data for trend analysis
        self.historical_scores = {}
        
    def calculate_domain_scores(self, sensor_data: Dict, sleep_data: Dict, 
                               anomaly_data: Dict) -> Dict:
        """
        Calculate ICOPE domain scores from available data
        
        Args:
            sensor_data: Sensor readings from the main app
            sleep_data: Sleep analysis results
            anomaly_data: Anomaly detection results
            
        Returns:
            Dictionary with domain scores and trends
        """
        # Calculate locomotion score (based on movement patterns)
        locomotion_score = self._calculate_locomotion_score(sensor_data, anomaly_data)
        
        # Calculate cognition score (simulated for now, can be enhanced)
        cognition_score = self._calculate_cognition_score(sensor_data)
        
        # Calculate psychological score (based on sleep and activity patterns)
        psychological_score = self._calculate_psychological_score(sleep_data, sensor_data)
        
        # Calculate sensory score (simulated, can integrate actual sensor data)
        sensory_score = self._calculate_sensory_score(sensor_data)
        
        # Calculate vitality score (based on sleep quality and activity)
        vitality_score = self._calculate_vitality_score(sleep_data, sensor_data)
        
        # Update domain scores
        self.domains['locomotion']['score'] = locomotion_score
        self.domains['cognition']['score'] = cognition_score
        self.domains['psychological']['score'] = psychological_score
        self.domains['sensory']['score'] = sensory_score
        self.domains['vitality']['score'] = vitality_score
        
        # Calculate trends if historical data exists
        self._calculate_trends()
        
        return self.domains
    
    def _calculate_locomotion_score(self, sensor_data: Dict, anomaly_data: Dict) -> float:
        """Calculate locomotion score based on movement patterns"""
        score = 70.0  # Base score
        
        if 'motion' in sensor_data:
            motion_data = sensor_data.get('motion', [])
            if motion_data:
                # Higher movement = better locomotion (within limits)
                avg_motion = np.mean(motion_data)
                score += min(20, avg_motion * 10)
        
        if 'anomalies' in anomaly_data:
            # Fewer anomalies = better health
            anomaly_count = len(anomaly_data.get('anomalies', []))
            score -= min(15, anomaly_count * 2)
        
        # Ensure score is within bounds
        return max(0, min(100, score))
    
    def _calculate_cognition_score(self, sensor_data: Dict) -> float:
        """Calculate cognition score (simulated for now)"""
        score = 75.0  # Base score
        
        # Simulate based on activity patterns
        if 'activity_patterns' in sensor_data:
            patterns = sensor_data.get('activity_patterns', {})
            if patterns:
                # More varied patterns = better cognition
                pattern_variety = len(patterns)
                score += min(15, pattern_variety * 3)
        
        # Add some randomness for demonstration
        score += np.random.uniform(-5, 5)
        
        return max(0, min(100, score))
    
    def _calculate_psychological_score(self, sleep_data: Dict, sensor_data: Dict) -> float:
        """Calculate psychological score based on sleep and activity"""
        score = 65.0  # Base score
        
        # Sleep quality impact
        if 'sleep_metrics' in sleep_data:
            sleep_metrics = sleep_data.get('sleep_metrics', {})
            efficiency = sleep_metrics.get('efficiency', 0.7)
            score += efficiency * 20
        
        # Activity level impact
        if 'activity_level' in sensor_data:
            activity = sensor_data.get('activity_level', 0.5)
            score += min(15, activity * 20)
        
        return max(0, min(100, score))
    
    def _calculate_sensory_score(self, sensor_data: Dict) -> float:
        """Calculate sensory score (simulated)"""
        score = 80.0  # Base score
        
        # Simulate based on environmental interaction
        if 'environment_interaction' in sensor_data:
            interaction = sensor_data.get('environment_interaction', 0.6)
            score += interaction * 15
        
        # Add some randomness for demonstration
        score += np.random.uniform(-8, 8)
        
        return max(0, min(100, score))
    
    def _calculate_vitality_score(self, sleep_data: Dict, sensor_data: Dict) -> float:
        """Calculate vitality score based on sleep and energy patterns"""
        score = 60.0  # Base score
        
        # Sleep duration impact
        if 'sleep_metrics' in sleep_data:
            sleep_metrics = sleep_data.get('sleep_metrics', {})
            duration = sleep_metrics.get('duration_hours', 6)
            if 7 <= duration <= 9:
                score += 20  # Optimal sleep
            elif 6 <= duration < 7 or 9 < duration <= 10:
                score += 10  # Acceptable sleep
            else:
                score -= 10  # Poor sleep
        
        # Energy patterns
        if 'energy_patterns' in sensor_data:
            energy = sensor_data.get('energy_patterns', {}).get('average', 0.5)
            score += energy * 20
        
        return max(0, min(100, score))
    
    def _calculate_trends(self):
        """Calculate trends based on historical data"""
        current_time = datetime.now()
        time_key = current_time.strftime('%Y-%m-%d')
        
        # Store current scores
        current_scores = {domain: data['score'] for domain, data in self.domains.items()}
        self.historical_scores[time_key] = current_scores
        
        # Calculate trends if we have historical data
        if len(self.historical_scores) > 1:
            # Get previous scores (most recent before today)
            sorted_dates = sorted(self.historical_scores.keys())
            if len(sorted_dates) >= 2:
                prev_date = sorted_dates[-2]
                prev_scores = self.historical_scores[prev_date]
                
                # Calculate trend (positive = improvement, negative = decline)
                for domain in self.domains:
                    current_score = current_scores.get(domain, 0)
                    prev_score = prev_scores.get(domain, 0)
                    trend = current_score - prev_score
                    self.domains[domain]['trend'] = trend
    
    def get_overall_score(self) -> float:
        """Calculate overall ICOPE aging score"""
        total_score = 0
        total_weight = 0
        
        for domain, data in self.domains.items():
            total_score += data['score'] * data['weight']
            total_weight += data['weight']
        
        if total_weight > 0:
            return total_score / total_weight
        return 0
    
    def get_aging_interpretation(self) -> str:
        """Get interpretation of overall aging score"""
        overall_score = self.get_overall_score()
        
        if overall_score >= 80:
            return "Excellent aging trajectory"
        elif overall_score >= 70:
            return "Good aging trajectory"
        elif overall_score >= 60:
            return "Moderate aging trajectory"
        elif overall_score >= 50:
            return "Concerning aging trajectory"
        else:
            return "Critical aging trajectory - intervention needed"
    
    def get_domain_recommendations(self, domain: str) -> List[str]:
        """Get recommendations for specific domain"""
        recommendations = {
            'locomotion': [
                "Engage in daily walking or light exercise",
                "Practice balance exercises",
                "Consider physical therapy if mobility declines"
            ],
            'cognition': [
                "Engage in mentally stimulating activities",
                "Practice memory exercises",
                "Maintain social connections"
            ],
            'psychological': [
                "Practice stress management techniques",
                "Maintain regular sleep schedule",
                "Engage in enjoyable activities"
            ],
            'sensory': [
                "Regular vision and hearing check-ups",
                "Use appropriate sensory aids if needed",
                "Maintain good lighting and reduce background noise"
            ],
            'vitality': [
                "Maintain balanced nutrition",
                "Ensure adequate sleep (7-9 hours)",
                "Stay hydrated and maintain energy levels"
            ]
        }
        
        return recommendations.get(domain, ["No specific recommendations available"])


if __name__ == "__main__":
    # Test the scoring system
    scoring = ICOPE_Scoring()
    
    # Test with sample data
    sample_sensor_data = {
        'motion': [0.1, 0.2, 0.15, 0.3, 0.25],
        'activity_patterns': {'morning': 'active', 'afternoon': 'rest', 'evening': 'active'},
        'activity_level': 0.7,
        'environment_interaction': 0.8,
        'energy_patterns': {'average': 0.6}
    }
    
    sample_sleep_data = {
        'sleep_metrics': {
            'efficiency': 0.85,
            'duration_hours': 7.5
        }
    }
    
    sample_anomaly_data = {
        'anomalies': ['low_activity', 'irregular_sleep']
    }
    
    scores = scoring.calculate_domain_scores(sample_sensor_data, sample_sleep_data, sample_anomaly_data)
    
    print("ICOPE Domain Scores:")
    for domain, data in scores.items():
        print(f"  {domain.capitalize()}: {data['score']:.1f}/100 (Trend: {data['trend']:+.1f})")
    
    print(f"\nOverall Aging Score: {scoring.get_overall_score():.1f}/100")
    print(f"Interpretation: {scoring.get_aging_interpretation()}")
