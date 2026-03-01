"""
ICOPE Service - Integrated Care for Older People Assessment

This service calculates ICOPE domain scores based on ADL (Activities of Daily Living)
data and provides trend analysis by comparing with historical assessments.

Domains:
- Locomotion: Mobility and movement patterns
- Vitality: Activity level and energy
- Cognition: Routine consistency and predictability
- Psychological: Social engagement and shared space usage
- Sensory: Environmental awareness (light variance proxy)
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd

from .base_service import BaseService

logger = logging.getLogger(__name__)


class ICOPEService(BaseService):
    """
    Manages ICOPE (Integrated Care for Older People) assessments.
    
    Calculates 5-domain vitality scores from ADL data and provides
    trend analysis by comparing with historical assessments.
    """
    
    # Domain weights for overall score calculation
    DOMAIN_WEIGHTS = {
        'locomotion': 0.25,
        'vitality': 0.25,
        'cognition': 0.20,
        'psychological': 0.15,
        'sensory': 0.15
    }
    
    # Rooms considered "social" for psychological score
    SOCIAL_ROOMS = ['living', 'living_room', 'livingroom', 'kitchen', 'dining']
    
    # Activities considered sedentary for vitality calculation
    SEDENTARY_ACTIVITIES = ['sleep', 'nap', 'inactive', 'sit', 'sitting', 'none']

    def calculate_and_save(
        self, 
        elder_id: str, 
        prediction_results: Dict[str, pd.DataFrame],
        assessment_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate ICOPE scores from prediction results and save to database.
        
        Args:
            elder_id: The elder's unique identifier
            prediction_results: Dict of room_name -> DataFrame with predictions
            assessment_date: Optional date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Dict containing the calculated ICOPE assessment, or None on failure
        """
        if not prediction_results:
            logger.warning(f"No prediction results provided for {elder_id}")
            return None
            
        assessment_date = assessment_date or datetime.now().strftime("%Y-%m-%d")
        
        # Aggregate all predictions into a single DataFrame
        all_preds = []
        for room_name, df in prediction_results.items():
            if isinstance(df, pd.DataFrame) and 'predicted_activity' in df.columns:
                temp = df.copy()
                temp['room'] = room_name
                all_preds.extend(temp.to_dict('records'))
        
        if not all_preds:
            logger.warning(f"No valid predictions found for {elder_id}")
            return None
            
        icope_df = pd.DataFrame(all_preds)
        
        # Calculate domain scores
        scores = self._calculate_domain_scores(icope_df)
        
        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(scores)
        
        # Calculate trend by comparing with previous assessment
        trend = self._calculate_trend(elder_id, overall_score)
        
        # Generate recommendations based on scores
        recommendations = self._generate_recommendations(scores)
        
        # Build assessment object
        assessment = {
            'elder_id': elder_id,
            'assessment_date': assessment_date,
            'locomotion_score': scores['locomotion'],
            'cognition_score': scores['cognition'],
            'vitality_score': scores['vitality'],
            'psychological_score': scores['psychological'],
            'sensory_score': scores['sensory'],
            'overall_score': overall_score,
            'trend': trend,
            'recommendations': recommendations
        }
        
        # Save to database
        self._save_assessment(assessment)
        
        logger.info(
            f"ICOPE Assessment for {elder_id}: Overall={overall_score}, "
            f"Trend={trend}, Loc={scores['locomotion']}, Vit={scores['vitality']}"
        )
        
        return assessment

    def recalculate_for_date(self, elder_id: str, record_date: str) -> Optional[Dict[str, Any]]:
        """
        Recalculate ICOPE scores for a specific date using existing ADL history.
        
        Useful for backfill or re-calibration after formula changes.
        
        Args:
            elder_id: The elder's unique identifier
            record_date: The date to recalculate (YYYY-MM-DD)
            
        Returns:
            Dict containing the recalculated ICOPE assessment, or None on failure
        """
        # Query ADL history for the specified date
        with self.db.get_connection() as conn:
            columns = ['timestamp', 'room', 'predicted_activity', 'confidence']
            rows = conn.execute('''
                SELECT timestamp, room, activity_type as predicted_activity, confidence
                FROM adl_history
                WHERE elder_id = ? AND record_date = ?
            ''', (elder_id, record_date)).fetchall()
        
        if not rows:
            logger.warning(f"No ADL history found for {elder_id} on {record_date}")
            return None
        
        # Convert to DataFrame format expected by calculate_and_save
        df = pd.DataFrame(self._rows_to_dicts(rows, columns))
        prediction_results = {'all_rooms': df}
        
        return self.calculate_and_save(elder_id, prediction_results, record_date)

    def get_latest_assessment(self, elder_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent ICOPE assessment for an elder."""
        with self.db.get_connection() as conn:
            columns = [
                'id',
                'elder_id',
                'assessment_date',
                'locomotion_score',
                'cognition_score',
                'psychological_score',
                'sensory_score',
                'vitality_score',
                'overall_score',
                'recommendations',
                'trend',
            ]
            row = conn.execute('''
                SELECT id, elder_id, assessment_date, locomotion_score, cognition_score,
                       psychological_score, sensory_score, vitality_score, overall_score,
                       recommendations, trend
                FROM icope_assessments
                WHERE elder_id = ? 
                ORDER BY assessment_date DESC LIMIT 1
            ''', (elder_id,)).fetchone()
            
            if row:
                return self._rows_to_dicts([row], columns)[0]
            return None

    def get_assessment_history(
        self, 
        elder_id: str, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get ICOPE assessment history for the past N days."""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        with self.db.get_connection() as conn:
            columns = [
                'id',
                'elder_id',
                'assessment_date',
                'locomotion_score',
                'cognition_score',
                'psychological_score',
                'sensory_score',
                'vitality_score',
                'overall_score',
                'recommendations',
                'trend',
            ]
            rows = conn.execute('''
                SELECT id, elder_id, assessment_date, locomotion_score, cognition_score,
                       psychological_score, sensory_score, vitality_score, overall_score,
                       recommendations, trend
                FROM icope_assessments
                WHERE elder_id = ? AND assessment_date >= ?
                ORDER BY assessment_date DESC
            ''', (elder_id, cutoff_date)).fetchall()
            
            return self._rows_to_dicts(rows, columns)

    def _calculate_domain_scores(self, icope_df: pd.DataFrame) -> Dict[str, int]:
        """Calculate individual domain scores from ADL data."""
        total_events = len(icope_df)
        if total_events == 0:
            return {domain: 50 for domain in self.DOMAIN_WEIGHTS.keys()}
        
        # Ensure timestamp is datetime for transition analysis
        if 'timestamp' in icope_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(icope_df['timestamp']):
                icope_df['timestamp'] = pd.to_datetime(icope_df['timestamp'])
            icope_df = icope_df.sort_values('timestamp')
        
        scores = {}
        
        # 1. LOCOMOTION (Mobility)
        # Based on unique rooms visited and room-to-room transitions
        unique_rooms = icope_df['room'].nunique()
        transitions = 0
        if 'timestamp' in icope_df.columns:
            transitions = (icope_df['room'] != icope_df['room'].shift()).sum()
        
        # Score: Base 60 + (rooms * 5) + (transitions / 5), capped at 100
        locomotion = min(100, 60 + (unique_rooms * 5) + int(transitions / 5))
        scores['locomotion'] = int(locomotion)
        
        # 2. VITALITY (Activity Level)
        # Based on proportion of non-sedentary events
        activity_col = icope_df['predicted_activity'].astype(str).str.lower()
        active_events = icope_df[~activity_col.isin(self.SEDENTARY_ACTIVITIES)].shape[0]
        # Score: Base 50 + scaled active events, capped at 100
        vitality = min(100, 50 + int(active_events / 10))
        scores['vitality'] = int(vitality)
        
        # 3. COGNITION (Routine Consistency)
        # Based on prediction confidence - low_confidence indicates inconsistency
        low_conf_count = (activity_col == 'low_confidence').sum()
        cognition = int(max(0, 100 - (low_conf_count / total_events * 100)))
        scores['cognition'] = int(cognition)
        
        # 4. PSYCHOLOGICAL (Social Engagement)
        # Based on time spent in social/shared spaces OR being outdoors ('out')
        room_col = icope_df['room'].astype(str).str.lower()
        # Include 'out' activity as social engagement
        potential_social_mask = room_col.isin(self.SOCIAL_ROOMS) | (activity_col == 'out')
        social_events = icope_df[potential_social_mask].shape[0]
        psychological = min(100, 60 + int(social_events / total_events * 50))
        scores['psychological'] = int(psychological)
        
        # 5. SENSORY (Environmental Awareness)
        # Based on light sensor variance if available
        sensory = 90  # Default baseline
        light_col = next((c for c in icope_df.columns if c.lower() == 'light'), None)
        if light_col and icope_df[light_col].notna().any():
            light_std = icope_df[light_col].std()
            if light_std > 50:
                sensory = 95  # High variance = good environmental response
            elif light_std < 5:
                sensory = 75  # Low variance = potential concern
        scores['sensory'] = int(sensory)
        
        return scores

    def _calculate_overall_score(self, scores: Dict[str, int]) -> int:
        """Calculate weighted overall score from domain scores."""
        total = 0
        weight_sum = 0
        
        for domain, weight in self.DOMAIN_WEIGHTS.items():
            if domain in scores:
                total += scores[domain] * weight
                weight_sum += weight
        
        return int(total / weight_sum) if weight_sum > 0 else 50

    def _calculate_trend(self, elder_id: str, current_score: int) -> str:
        """
        Calculate trend by comparing current score with previous week's assessment.
        
        Returns: 'improving', 'stable', or 'declining'
        """
        # Get assessment from approximately 7 days ago
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        with self.db.get_connection() as conn:
            row = conn.execute('''
                SELECT overall_score FROM icope_assessments 
                WHERE elder_id = ? AND assessment_date <= ?
                ORDER BY assessment_date DESC LIMIT 1
            ''', (elder_id, week_ago)).fetchone()
        
        if not row:
            # No previous assessment, default to stable
            return 'stable'
        
        previous_score = row[0]
        diff = current_score - previous_score
        
        # Threshold: 5 points = significant change
        if diff >= 5:
            return 'improving'
        elif diff <= -5:
            return 'declining'
        else:
            return 'stable'

    def _generate_recommendations(self, scores: Dict[str, int]) -> List[str]:
        """Generate actionable recommendations based on domain scores."""
        recommendations = []
        
        if scores.get('locomotion', 100) < 70:
            recommendations.append("Increase daily room-to-room movement.")
        if scores.get('vitality', 100) < 70:
            recommendations.append("Schedule active periods during the day.")
        if scores.get('cognition', 100) < 70:
            recommendations.append("Review daily routine for consistency.")
        if scores.get('psychological', 100) < 70:
            recommendations.append("Encourage time in social areas (living room, kitchen).")
        if scores.get('sensory', 100) < 70:
            recommendations.append("Check lighting conditions and environmental stimulation.")
        
        return recommendations

    def _save_assessment(self, assessment: Dict[str, Any]) -> int:
        """Save ICOPE assessment to database."""
        with self.db.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO icope_assessments 
                (elder_id, assessment_date, overall_score, locomotion_score, cognition_score, 
                 vitality_score, psychological_score, sensory_score, trend, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                assessment['elder_id'],
                assessment['assessment_date'],
                assessment['overall_score'],
                assessment['locomotion_score'],
                assessment['cognition_score'],
                assessment['vitality_score'],
                assessment['psychological_score'],
                assessment['sensory_score'],
                assessment['trend'],
                json.dumps(assessment['recommendations'])
            ))
            conn.commit()
            return cursor.lastrowid
    @staticmethod
    def _rows_to_dicts(rows, columns: List[str]) -> List[Dict[str, Any]]:
        records = []
        for row in rows:
            try:
                records.append({col: row[col] for col in columns})
            except (TypeError, KeyError, IndexError):
                records.append(dict(zip(columns, row)))
        return records
