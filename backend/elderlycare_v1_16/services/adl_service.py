import json
import logging
import pandas as pd
from typing import Dict, Any, List
from .base_service import BaseService

logger = logging.getLogger(__name__)

class ADLService(BaseService):
    """
    Manages Activities of Daily Living (ADL) History.
    """

    def save_adl_events(self, elder_id: str, room: str, df: pd.DataFrame):
        """
        Batch save ADL events from a DataFrame.
        """
        events = df.to_dict('records')
        for event in events:
            # Ensure required fields
            event['room'] = room
            if 'activity' not in event and 'predicted_activity' in event:
                 event['activity'] = event['predicted_activity']
            self.save_adl_event(elder_id, event)

    def save_adl_event(self, elder_id: str, event: Dict[str, Any] | str, room: str = None, timestamp=None, confidence: float = None, **kwargs):
        """
        Save a single ADL event (e.g., a toilet visit).
        Skips if a manually corrected row (is_corrected=1) already exists for this timestamp/room.
        
        Timestamps are normalized to 10-second intervals for consistent matching.
        Automatically creates elder profile if it doesn't exist.
        """
        import json
        from utils.room_utils import normalize_timestamp, normalize_room_name

        # Backward compatibility:
        # - New API: save_adl_event(elder_id, event_dict)
        # - Legacy API: save_adl_event(elder_id, activity, room, timestamp, confidence=...)
        if isinstance(event, dict):
            event_payload = dict(event)
        else:
            event_payload = {
                "activity": event,
                "room": room,
                "timestamp": timestamp,
            }
            if confidence is not None:
                event_payload["confidence"] = confidence
            event_payload.update(kwargs)
        event = event_payload
        
        # Normalize timestamp to 10-second intervals for consistent matching
        ts = event['timestamp']
        ts_str = normalize_timestamp(ts)  # Floors to 10s boundary
        
        room = event.get('room', 'unknown')
        normalized_room = normalize_room_name(room)
        
        # Prepare sensor features payload (raw sensor context + model hint metadata).
        def _has_value(value) -> bool:
            if value is None:
                return False
            try:
                return not bool(pd.isna(value))
            except Exception:
                return True

        sensor_payload = {}
        raw_sensor_features = event.get('sensor_features')
        if isinstance(raw_sensor_features, str):
            try:
                raw_sensor_features = json.loads(raw_sensor_features)
            except (TypeError, ValueError, json.JSONDecodeError):
                raw_sensor_features = None
        if isinstance(raw_sensor_features, dict):
            for key, value in raw_sensor_features.items():
                if _has_value(value):
                    sensor_payload[key] = value

        sensor_keys = ['motion', 'temperature', 'light', 'sound', 'co2', 'humidity']
        for key in sensor_keys:
            val = event.get(key)
            if _has_value(val):
                try:
                    sensor_payload[key] = float(val)
                except (TypeError, ValueError):
                    sensor_payload[key] = val

        top1_label = event.get('predicted_top1_label')
        if _has_value(top1_label):
            sensor_payload['predicted_top1_label'] = str(top1_label)

        top1_prob = event.get('predicted_top1_prob')
        if _has_value(top1_prob):
            sensor_payload['predicted_top1_prob'] = float(top1_prob)

        top2_label = event.get('predicted_top2_label')
        if _has_value(top2_label):
            sensor_payload['predicted_top2_label'] = str(top2_label)

        top2_prob = event.get('predicted_top2_prob')
        if _has_value(top2_prob):
            sensor_payload['predicted_top2_prob'] = float(top2_prob)

        top2_label_raw = event.get('predicted_top2_label_raw')
        if _has_value(top2_label_raw):
            sensor_payload['predicted_top2_label_raw'] = str(top2_label_raw)

        top2_prob_raw = event.get('predicted_top2_prob_raw')
        if _has_value(top2_prob_raw):
            sensor_payload['predicted_top2_prob_raw'] = float(top2_prob_raw)

        low_conf_threshold = event.get('low_confidence_threshold')
        if _has_value(low_conf_threshold):
            sensor_payload['low_confidence_threshold'] = float(low_conf_threshold)

        low_conf_hint = event.get('low_confidence_hint_label')
        if _has_value(low_conf_hint):
            sensor_payload['low_confidence_hint_label'] = str(low_conf_hint)

        is_low_conf = event.get('is_low_confidence')
        if _has_value(is_low_conf):
            sensor_payload['is_low_confidence'] = bool(is_low_conf)

        sensor_features = json.dumps(sensor_payload) if sensor_payload else None
        
        with self.db.get_connection() as conn:
            # Auto-create elder profile if it doesn't exist (prevents FOREIGN KEY error)
            # Using ON CONFLICT DO NOTHING for PostgreSQL compatibility
            conn.execute('''
                INSERT INTO elders (elder_id, full_name) 
                VALUES (?, ?)
                ON CONFLICT (elder_id) DO NOTHING
            ''', (elder_id, elder_id))

            # Check if a corrected row already exists for this timestamp/room
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM adl_history 
                WHERE elder_id = ? 
                  AND timestamp = ? 
                  AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                  AND is_corrected = 1
            ''', (elder_id, ts_str, normalized_room))
            
            if cursor.fetchone():
                # Skip - don't overwrite manually corrected data
                return
            
            # Delete any existing uncorrected row for this timestamp/room to avoid duplicates
            cursor.execute('''
                DELETE FROM adl_history 
                WHERE elder_id = ? 
                  AND timestamp = ? 
                  AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                  AND is_corrected = 0
            ''', (elder_id, ts_str, normalized_room))
            
            # Insert the new prediction with sensor features (validated)
            from utils.segment_utils import validate_activity_for_room
            raw_activity = event.get('activity')
            if pd.isna(raw_activity):
                raw_activity = 'unknown'
            raw_activity = str(raw_activity).strip()
            if raw_activity.lower() in {'', 'nan', 'none'}:
                raw_activity = 'unknown'
            canonical_activity = 'toilet' if str(raw_activity).strip().lower() == 'toileting' else raw_activity
            validated_activity = validate_activity_for_room(canonical_activity, room)
            
            conn.execute('''
                INSERT INTO adl_history 
                (elder_id, record_date, timestamp, activity_type, duration_minutes, confidence, room, is_anomaly, is_corrected, sensor_features)
                VALUES (?, DATE(?), ?, ?, ?, ?, ?, ?, 0, ?)
            ''', (
                elder_id,
                ts_str,
                ts_str,
                validated_activity,  # Use validated activity
                event.get('duration_minutes', event.get('duration', 0)),
                event.get('confidence', 0.0),
                room,  # Preserve original room name
                int(event.get('is_anomaly', False)),
                sensor_features
            ))
            conn.commit()

    def get_nighttime_activity(self, elder_id: str, date_str: str) -> List[Dict]:
        """
        Get activities between 10PM previous day and 7AM current day (example logic).
        For simplicity, let's query specific types like 'toileting'.
        """
        # Implementation depends on exact definitions of "Night"
        pass
    
    def get_todays_events(self, elder_id: str) -> List[Dict]:
         with self.db.get_connection() as conn:
            columns = [
                'id',
                'elder_id',
                'record_date',
                'timestamp',
                'activity_type',
                'duration_minutes',
                'confidence',
                'room',
                'is_anomaly',
                'is_corrected',
                'sensor_features',
            ]
            rows = conn.execute('''
                SELECT id, elder_id, record_date, timestamp, activity_type, duration_minutes,
                       confidence, room, is_anomaly, is_corrected, sensor_features
                FROM adl_history
                WHERE elder_id = ? AND record_date = CURRENT_DATE
                ORDER BY timestamp DESC
            ''', (elder_id,)).fetchall()
            return self._rows_to_dicts(rows, columns)

    def delete_correction(self, correction_id: int, deleted_by: str = 'user') -> Dict[str, Any]:
        """
        Soft-delete a correction from the audit trail.
        
        This implements: "ML forgets, Audit remembers"
        - correction_history: marked as is_deleted=1 (kept for audit)
        - adl_history: reset is_corrected=0 (removed from ML training)
        
        Args:
            correction_id: ID of the correction_history row to delete
            deleted_by: Username/identifier of who performed the deletion
            
        Returns:
            Dict with deletion status and affected row counts
        """
        from datetime import datetime
        from utils.room_utils import normalize_room_name
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Fetch the correction details before marking as deleted
            cursor.execute('''
                SELECT elder_id, room, timestamp_start, timestamp_end, new_activity, is_deleted
                FROM correction_history 
                WHERE id = ?
            ''', (correction_id,))
            
            row = cursor.fetchone()
            if not row:
                return {'success': False, 'error': 'Correction not found', 'correction_id': correction_id}
            
            elder_id, room, ts_start, ts_end, activity, already_deleted = row
            
            if already_deleted:
                return {'success': False, 'error': 'Correction already deleted', 'correction_id': correction_id}
            
            normalized_room = normalize_room_name(room)
            
            # 2. Cascade reset: Set is_corrected=0 for affected adl_history rows
            # This removes them from ML Golden Samples
            cursor.execute('''
                UPDATE adl_history 
                SET is_corrected = 0
                WHERE elder_id = ?
                  AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                  AND is_corrected = 1
            ''', (elder_id, normalized_room, ts_start, ts_end))
            
            rows_reset = cursor.rowcount
            
            # 3. Soft-delete the correction_history entry (keep for audit)
            cursor.execute('''
                UPDATE correction_history 
                SET is_deleted = 1, 
                    deleted_at = ?,
                    deleted_by = ?
                WHERE id = ?
            ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), deleted_by, correction_id))
            
            # 4. Regenerate Segments for consistency (within transaction)
            if isinstance(ts_start, str):
                record_date = ts_start.split(' ')[0]
            else:
                record_date = ts_start.strftime('%Y-%m-%d')
                
            from utils.segment_utils import regenerate_segments
            regenerate_segments(elder_id, room, record_date, conn)
            
            conn.commit()
            
            logger.info(f"Soft-deleted correction {correction_id}: reset {rows_reset} adl_history rows for {elder_id}/{room} and regenerated segments.")
            
            return {
                'success': True,
                'correction_id': correction_id,
                'elder_id': elder_id,
                'room': room,
                'timestamp_range': f"{ts_start} to {ts_end}",
                'activity_removed': activity,
                'adl_rows_reset': rows_reset,
                'deleted_by': deleted_by
            }

    def get_correction_history(self, elder_id: str, include_deleted: bool = False) -> List[Dict]:
        """
        Get correction history for an elder.
        
        Args:
            elder_id: Elder identifier
            include_deleted: If True, includes soft-deleted corrections (for full audit)
            
        Returns:
            List of correction records
        """
        with self.db.get_connection() as conn:
            columns = [
                'id',
                'adl_history_id',
                'elder_id',
                'room',
                'timestamp_start',
                'timestamp_end',
                'old_activity',
                'new_activity',
                'rows_affected',
                'corrected_by',
                'corrected_at',
                'is_deleted',
                'deleted_at',
                'deleted_by',
            ]
            if include_deleted:
                query = '''
                    SELECT id, adl_history_id, elder_id, room, timestamp_start, timestamp_end,
                           old_activity, new_activity, rows_affected, corrected_by,
                           corrected_at, is_deleted, deleted_at, deleted_by
                    FROM correction_history
                    WHERE elder_id = ?
                    ORDER BY corrected_at DESC
                '''
            else:
                query = '''
                    SELECT id, adl_history_id, elder_id, room, timestamp_start, timestamp_end,
                           old_activity, new_activity, rows_affected, corrected_by,
                           corrected_at, is_deleted, deleted_at, deleted_by
                    FROM correction_history
                    WHERE elder_id = ? AND is_deleted = 0
                    ORDER BY corrected_at DESC
                '''
            rows = conn.execute(query, (elder_id,)).fetchall()
            return self._rows_to_dicts(rows, columns)
    @staticmethod
    def _rows_to_dicts(rows, columns: List[str]) -> List[Dict[str, Any]]:
        records = []
        for row in rows:
            try:
                records.append({col: row[col] for col in columns})
            except (TypeError, KeyError, IndexError):
                records.append(dict(zip(columns, row)))
        return records
