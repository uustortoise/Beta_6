import json
import logging
from typing import Dict, Any, Optional
from .base_service import BaseService

logger = logging.getLogger(__name__)

class SleepService(BaseService):
    """
    Manages Sleep Analysis Data.
    """

    def save_sleep_analysis(self, elder_id: str, analysis: Dict[str, Any], date_str: str) -> int:
        """
        Save a daily sleep analysis record.
        """
        duration = analysis.get('total_duration_hours', analysis.get('duration_hours', 0))
        efficiency = analysis.get('sleep_efficiency', analysis.get('efficiency', 0))
        if isinstance(efficiency, (int, float)) and 0 < efficiency <= 1:
            efficiency *= 100
        quality = analysis.get('quality_score', analysis.get('quality', 0))
        stages = analysis.get('stages_summary', analysis.get('stages', {})) # Light, Deep, REM
        insights = analysis.get('insights', [])

        with self.db.get_connection() as conn:
            # Delete existing analysis for this date to prevent duplicates
            conn.execute('DELETE FROM sleep_analysis WHERE elder_id = ? AND analysis_date = ?', (elder_id, date_str))
            
            cursor = conn.execute('''
                INSERT INTO sleep_analysis 
                (elder_id, analysis_date, duration_hours, efficiency_percent, sleep_stages, quality_score, insights)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                elder_id, 
                date_str, 
                duration, 
                efficiency, 
                json.dumps(stages), 
                quality, 
                json.dumps(insights)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_latest_sleep(self, elder_id: str) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            columns = [
                'id',
                'elder_id',
                'analysis_date',
                'duration_hours',
                'efficiency_percent',
                'sleep_stages',
                'quality_score',
                'insights',
            ]
            row = conn.execute('''
                SELECT id, elder_id, analysis_date, duration_hours, efficiency_percent,
                       sleep_stages, quality_score, insights
                FROM sleep_analysis
                WHERE elder_id = ? 
                ORDER BY analysis_date DESC LIMIT 1
            ''', (elder_id,)).fetchone()
            
            if row:
                return self._row_to_dict(row, columns)
            return None
    @staticmethod
    def _row_to_dict(row, columns):
        try:
            return {col: row[col] for col in columns}
        except (TypeError, KeyError, IndexError):
            return dict(zip(columns, row))
