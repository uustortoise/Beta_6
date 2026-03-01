"""
TrajectoryEngine - Cross-Room Activity Tracking (v2 - Refined)

This module analyzes ADL events across all rooms to generate movement trajectories.
It identifies when an elder moves from one room to another and creates a unified
path representing their movement through the house.

Algorithm (v2):
1. Fetch all ADL events for a day, sorted by timestamp
2. Consolidate consecutive same-room events into "room visits"
3. Create trajectories from sequences of room visits
4. A trajectory ends when there's a gap > threshold OR activity becomes inactive for extended period
"""

import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "residents_master_data.db"


class TrajectoryEngine:
    """
    Analyzes room-level ADL predictions to identify cross-room movement patterns.
    """
    
    def __init__(self, db_path=None):
        # Import shared database utility
        from utils.intelligence_db import (
            coerce_timestamp_column,
            get_intelligence_db,
            query_to_dataframe,
        )
        self._get_connection = get_intelligence_db
        self._query_to_dataframe = query_to_dataframe
        self._coerce_timestamp_column = coerce_timestamp_column
        
        # Configuration
        self.gap_threshold_minutes = 10   # Gap to end a trajectory
        self.min_visit_events = 3         # Min events in a room to count as a "visit"
        self.inactive_activities = ['inactive', 'low_confidence', 'unknown', None, '']
        
    def analyze_day(self, elder_id: str, date_str: str) -> list:
        """
        Analyze a full day of ADL events to extract trajectories.
        """
        logger.info(f"[TrajectoryEngine] Analyzing trajectories for {elder_id} on {date_str}")
        
        try:
            with self._get_connection() as conn:
                # 1. Fetch all ADL events for the day
                query = """
                    SELECT timestamp, room, activity_type, confidence
                    FROM adl_history
                    WHERE elder_id = ? AND record_date = ?
                    ORDER BY timestamp
                """
                df = self._query_to_dataframe(conn, query, (elder_id, date_str))
                
                if df.empty:
                    logger.warning(f"No ADL events found for {elder_id} on {date_str}")
                    return []
                
                df = self._coerce_timestamp_column(
                    df,
                    "timestamp",
                    f"TrajectoryEngine {elder_id}/{date_str}",
                )
                if df.empty:
                    logger.warning(f"No valid timestamp rows for {elder_id} on {date_str}")
                    return []
                
                # 2. Filter to "active" events only
                active_df = df[~df['activity_type'].isin(self.inactive_activities)].copy()
                
                if active_df.empty:
                    logger.info(f"No active events found for {elder_id} on {date_str}")
                    return []
                
                # 3. Consolidate into room visits
                visits = self._consolidate_room_visits(active_df)
                
                # 4. Build trajectories from visits
                trajectories = self._build_trajectories_from_visits(visits)
                
                # 5. Save to database
                self._save_trajectories(conn, elder_id, date_str, trajectories)
                
                logger.info(f"[TrajectoryEngine] Generated {len(trajectories)} trajectories for {elder_id}")
                return trajectories
                
        except Exception as e:
            logger.error(f"[TrajectoryEngine] Error analyzing {elder_id}: {e}", exc_info=True)
            return []
    
    def _consolidate_room_visits(self, df: pd.DataFrame) -> list:
        """
        Consolidate consecutive same-room events into "visits".
        A visit is a contiguous period spent in one room.
        """
        visits = []
        
        if df.empty:
            return visits
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        current_room = None
        visit_start = None
        visit_events = []
        
        for idx, row in df.iterrows():
            room = row['room']
            ts = row['timestamp']
            activity = row['activity_type']
            confidence = row.get('confidence', 1.0)
            
            if current_room is None:
                # Start first visit
                current_room = room
                visit_start = ts
                visit_events = [{'ts': ts, 'activity': activity, 'confidence': confidence}]
            elif room == current_room:
                # Continue current visit
                visit_events.append({'ts': ts, 'activity': activity, 'confidence': confidence})
            else:
                # Room changed - save current visit and start new one
                if len(visit_events) >= self.min_visit_events:
                    visits.append({
                        'room': current_room,
                        'start_time': visit_start,
                        'end_time': visit_events[-1]['ts'],
                        'duration_minutes': (visit_events[-1]['ts'] - visit_start).total_seconds() / 60.0,
                        'primary_activity': self._get_primary_activity(visit_events),
                        'avg_confidence': sum(e['confidence'] for e in visit_events) / len(visit_events),
                        'event_count': len(visit_events)
                    })
                
                # Start new visit
                current_room = room
                visit_start = ts
                visit_events = [{'ts': ts, 'activity': activity, 'confidence': confidence}]
        
        # Don't forget the last visit
        if current_room is not None and len(visit_events) >= self.min_visit_events:
            visits.append({
                'room': current_room,
                'start_time': visit_start,
                'end_time': visit_events[-1]['ts'],
                'duration_minutes': (visit_events[-1]['ts'] - visit_start).total_seconds() / 60.0,
                'primary_activity': self._get_primary_activity(visit_events),
                'avg_confidence': sum(e['confidence'] for e in visit_events) / len(visit_events),
                'event_count': len(visit_events)
            })
        
        return visits
    
    def _get_primary_activity(self, events: list) -> str:
        """Get the most common activity in a set of events."""
        if not events:
            return 'unknown'
        activities = [e['activity'] for e in events]
        return max(set(activities), key=activities.count)
    
    def _build_trajectories_from_visits(self, visits: list) -> list:
        """
        Build trajectory paths from room visits.
        A trajectory is a sequence of visits without large time gaps.
        """
        trajectories = []
        
        if len(visits) < 2:
            return trajectories
        
        current_trajectory = [visits[0]]
        
        for i in range(1, len(visits)):
            prev_visit = visits[i-1]
            curr_visit = visits[i]
            
            # Check time gap
            gap_minutes = (curr_visit['start_time'] - prev_visit['end_time']).total_seconds() / 60.0
            
            if gap_minutes > self.gap_threshold_minutes:
                # Gap too large - save current trajectory and start new one
                if len(current_trajectory) >= 2:
                    trajectories.append(self._create_trajectory_record(current_trajectory))
                current_trajectory = [curr_visit]
            else:
                # Continue trajectory
                current_trajectory.append(curr_visit)
        
        # Save last trajectory
        if len(current_trajectory) >= 2:
            trajectories.append(self._create_trajectory_record(current_trajectory))
        
        return trajectories
    
    def _create_trajectory_record(self, visits: list) -> dict:
        """Create a trajectory record from a sequence of visits."""
        path = '->'.join(v['room'] for v in visits)
        room_sequence = [
            {
                'room': v['room'],
                'start': v['start_time'].isoformat(),
                'end': v['end_time'].isoformat(),
                'activity': v['primary_activity'],
                'duration_min': round(v['duration_minutes'], 1)
            }
            for v in visits
        ]
        
        return {
            'start_time': visits[0]['start_time'],
            'end_time': visits[-1]['end_time'],
            'path': path,
            'primary_activity': visits[-1]['primary_activity'],  # Activity at destination
            'room_sequence': room_sequence,
            'duration_minutes': (visits[-1]['end_time'] - visits[0]['start_time']).total_seconds() / 60.0,
            'confidence': sum(v['avg_confidence'] for v in visits) / len(visits),
            'visit_count': len(visits)
        }
    
    def _save_trajectories(self, conn, elder_id: str, date_str: str, trajectories: list):
        """Save trajectories to the database."""
        
        # Delete existing
        conn.execute(
            "DELETE FROM trajectory_events WHERE elder_id = ? AND record_date = ?",
            (elder_id, date_str)
        )
        
        for traj in trajectories:
            try:
                conn.execute("""
                    INSERT INTO trajectory_events 
                    (elder_id, start_time, end_time, path, primary_activity, 
                     room_sequence, duration_minutes, confidence, record_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    elder_id,
                    traj['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    traj['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    traj['path'],
                    traj['primary_activity'],
                    json.dumps(traj['room_sequence']),
                    round(traj['duration_minutes'], 2),
                    round(traj['confidence'], 3),
                    date_str
                ))
            except Exception as e:
                logger.error(f"Failed to insert trajectory: {e}")
        
        conn.commit()
        logger.info(f"Saved {len(trajectories)} trajectories to database")


def run_trajectory_analysis(elder_id: str, date_str: str, db_path=None) -> list:
    """Run trajectory analysis for a specific elder and date."""
    engine = TrajectoryEngine(db_path=db_path)
    return engine.analyze_day(elder_id, date_str)
