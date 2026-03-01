"""
Segment generation utilities.
Single source of truth for generating and updating activity_segments from adl_history.
"""

import pandas as pd
import logging
from typing import Optional

try:
    from backend.utils.intelligence_db import coerce_timestamp_column, query_to_dataframe
except ImportError:
    from utils.intelligence_db import coerce_timestamp_column, query_to_dataframe

logger = logging.getLogger(__name__)

# Room-specific activity validation mapping
# Defines which activities are valid for each room
# Updated to include 'out' for all rooms (person can leave any room)
ROOM_ACTIVITY_VALIDATION = {
    'livingroom': ['inactive', 'unoccupied', 'livingroom_normal_use', 'watch_tv', 'low_confidence', 'unknown', 'out'],
    'bedroom': ['inactive', 'unoccupied', 'sleep', 'nap', 'bedroom_normal_use', 'room_normal_use', 'change_clothes', 'low_confidence', 'unknown', 'out'],
    'bathroom': ['inactive', 'unoccupied', 'bathroom_normal_use', 'shower', 'toilet', 'low_confidence', 'unknown', 'out'],
    'kitchen': ['inactive', 'unoccupied', 'kitchen_normal_use', 'cooking', 'washing_dishes', 'low_confidence', 'unknown', 'out'],
    'entrance': ['inactive', 'unoccupied', 'out', 'low_confidence', 'unknown'],
    # Default fallback for unknown rooms - allow basic activities
    'default': ['inactive', 'unoccupied', 'low_confidence', 'unknown', 'out']
}

# Activity normalization mapping (applied BEFORE validation)
# Maps variant spellings/names to canonical form
ACTIVITY_NORMALIZATION = {
    'kitchen normal use': 'kitchen_normal_use',
    'room_normal_use': 'bedroom_normal_use',
    '5area no action': 'inactive',
    '5area_no_action': 'inactive',
    '5 area no action': 'inactive',
}

# Activities that should NOT break sleep/nap continuity
# Transitions within this set are treated as part of same sleep period
# Added 'unoccupied' (Beta 5.5 Fix) to handle sensor dropouts during sleep
SLEEP_CONTINUATION_ACTIVITIES = frozenset(['sleep', 'nap', 'low_confidence', 'unknown', 'inactive', 'unoccupied'])

# Maximum gap (in seconds) allowed within SLEEP_CONTINUATION to still merge
# Prevents merging morning and evening sleep (2 hours = 7200 seconds)
SLEEP_CONTINUATION_MAX_GAP = 7200

def normalize_activity_name(activity: str) -> str:
    """
    Normalize activity name to canonical form.
    Handles spaces vs underscores, variant spellings, etc.
    """
    if activity in ACTIVITY_NORMALIZATION:
        return ACTIVITY_NORMALIZATION[activity]
    return activity


def should_break_segment(prev_activity: str, curr_activity: str, time_gap_seconds: float) -> bool:
    """
    Determine if activity transition should break current segment.
    
    This function focuses on TIME GAP detection. Activity-based merging
    is handled separately in merge_sleep_segments() post-processing.
    
    Rules:
    1. Always break if time gap > SLEEP_CONTINUATION_MAX_GAP (2h) for sleep-related
    2. Always break if time gap > TIME_GAP_THRESHOLD (5 min) for other activities
    3. Break on any activity change (merging handled post-process)
    
    Args:
        prev_activity: Previous activity type
        curr_activity: Current activity type
        time_gap_seconds: Time gap between consecutive rows
        
    Returns:
        bool: True if should start new segment, False if continue current
    """
    # Standard time gap threshold
    TIME_GAP_THRESHOLD = 300  # 5 minutes
    
    # For sleep-related activities, use extended gap threshold for time gaps only
    if prev_activity in SLEEP_CONTINUATION_ACTIVITIES and curr_activity in SLEEP_CONTINUATION_ACTIVITIES:
        if time_gap_seconds > SLEEP_CONTINUATION_MAX_GAP:
            return True
        # If same activity and within extended gap, don't break
        if prev_activity == curr_activity:
            return False
    
    # For all other cases, break on activity change
    if prev_activity != curr_activity:
        return True
    
    # Break on standard time gap
    if time_gap_seconds > TIME_GAP_THRESHOLD:
        return True
    
    return False


# Maximum interruption duration (in minutes) to still merge with sleep
# Interruptions longer than this are considered "awakening"
SLEEP_MERGE_MAX_INTERRUPTION = 30  # 30 minutes


def merge_sleep_segments(segments: list) -> list:
    """
    Post-process segments to merge adjacent sleep periods with brief interruptions.
    
    This merges patterns like:
      sleep(5h) → low_confidence(10min) → sleep(2h) 
    into:
      sleep(7h 10min)
    
    But keeps separate:
      sleep(5h) → low_confidence(2h) → sleep(3h)
    as:
      sleep(5h), inactive(2h), sleep(3h)
    
    Args:
        segments: List of segment dicts from generate_segments
        
    Returns:
        list: Merged segment list
    """
    if len(segments) <= 1:
        return segments
    
    merged = []
    i = 0
    
    while i < len(segments):
        current = segments[i].copy()
        
        # Check if current is a definite sleep activity (sleep or nap)
        if current['activity_type'] in ('sleep', 'nap'):
            # Look ahead for merging opportunities
            while i + 2 < len(segments):
                interruption = segments[i + 1]
                next_seg = segments[i + 2]
                
                # Check if: current is sleep, next is interruption, then sleep again
                is_brief_interruption = (
                    interruption['activity_type'] in SLEEP_CONTINUATION_ACTIVITIES and
                    interruption['duration_minutes'] <= SLEEP_MERGE_MAX_INTERRUPTION and
                    next_seg['activity_type'] in ('sleep', 'nap')
                )
                
                if is_brief_interruption:
                    # Merge: extend current to include interruption and next sleep
                    current['end_time'] = next_seg['end_time']
                    current['duration_minutes'] = (
                        pd.to_datetime(current['end_time']) - 
                        pd.to_datetime(current['start_time'])
                    ).total_seconds() / 60.0
                    current['event_count'] = (
                        current.get('event_count', 0) + 
                        interruption.get('event_count', 0) + 
                        next_seg.get('event_count', 0)
                    )
                    # Skip the interruption and next segment
                    i += 2
                    logger.debug(f"Merged sleep segment: absorbed {interruption['duration_minutes']:.1f}min interruption")
                else:
                    # Can't merge further
                    break
        
        merged.append(current)
        i += 1
    
    return merged

def validate_activity_for_room(activity: str, room: str) -> str:
    """
    Validate that an activity is appropriate for the given room.
    
    Args:
        activity: Activity type string
        room: Room name
        
    Returns:
        Validated activity (unchanged if valid, 'inactive' if invalid)
    """
    from backend.utils.room_utils import normalize_room_name
    
    # Guard missing/NaN labels before room-level validation.
    if activity is None or pd.isna(activity):
        return 'unknown'
    activity_str = str(activity).strip()
    if activity_str.lower() in {'', 'nan', 'none'}:
        return 'unknown'

    # First, normalize the activity name
    normalized_activity = normalize_activity_name(activity_str)
    normalized_room = normalize_room_name(room)
    
    # Get valid activities for this room, fallback to default
    valid_activities = ROOM_ACTIVITY_VALIDATION.get(normalized_room, ROOM_ACTIVITY_VALIDATION['default'])
    
    # Check if normalized activity is valid
    if normalized_activity in valid_activities:
        return normalized_activity  # Return normalized form
    
    # Special case: 'nap' should only be in bedroom
    if normalized_activity == 'nap' and normalized_room != 'bedroom':
        logger.warning(f"Invalid activity '{activity}' for room '{room}'. Changing to 'inactive'.")
        return 'inactive'
    
    # Special case: 'sleep' should only be in bedroom
    if normalized_activity == 'sleep' and normalized_room != 'bedroom':
        logger.warning(f"Invalid activity '{activity}' for room '{room}'. Changing to 'inactive'.")
        return 'inactive'
    
    # For other invalid activities, default to 'inactive'
    logger.warning(f"Activity '{activity}' not in valid list for room '{room}'. Changing to 'inactive'.")
    return 'inactive'


def regenerate_segments(elder_id: str, room: str, record_date: str, conn=None) -> int:
    """
    Regenerate activity_segments for a specific elder/room/date from adl_history.
    
    This is the SINGLE SOURCE OF TRUTH for segment generation.
    It reads from adl_history (which includes corrections) and regenerates
    the corresponding activity_segments entries.
    
    Args:
        elder_id: Elder identifier
        room: Room name (will be normalized for matching)
        record_date: Date string in YYYY-MM-DD format
        conn: Optional existing database connection (will create if not provided)
        
    Returns:
        Number of segments created
    """
    from backend.utils.room_utils import normalize_room_name
    from pathlib import Path
    import sqlite3
    
    # Import adapter for fallback
    try:
        from backend.db.legacy_adapter import LegacyDatabaseAdapter
    except ImportError:
        from elderlycare_v1_16.database import db as adapter
    else:
        adapter = LegacyDatabaseAdapter()
    
    # Resolve DB path relative to this file
    # segment_utils.py is in backend/utils/, DB is in data/processed/
    backend_dir = Path(__file__).parent.parent
    project_dir = backend_dir.parent
    DB_PATH = str(project_dir / 'data' / 'processed' / 'residents_master_data.db')
    
    normalized_room = normalize_room_name(room)
    close_conn = False
    conn_ctx = None
    
    try:
        if conn is None:
            # Use adapter to get appropriate connection (Postgres or SQLite)
            conn_ctx = adapter.get_connection()
            conn = conn_ctx.__enter__()
            close_conn = True
        # Fetch from adl_history (includes corrections)
        db_df = query_to_dataframe(conn, '''
            SELECT timestamp, activity_type, confidence, is_corrected
            FROM adl_history 
            WHERE elder_id = ? 
              AND record_date = ?
              AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
            ORDER BY timestamp
        ''', (elder_id, record_date, normalized_room))
        
        if db_df.empty:
            logger.warning(f"No data for {elder_id}/{room}/{record_date}")
            return 0
        
        # Ensure timestamp is datetime and sorted
        db_df = coerce_timestamp_column(
            db_df,
            "timestamp",
            f"Segment regeneration {elder_id}/{room}/{record_date}",
        )
        if db_df.empty:
            logger.warning(f"No valid timestamp rows for {elder_id}/{room}/{record_date}")
            return 0

        db_df = db_df.sort_values('timestamp')
        
        # Add time difference between rows
        db_df['time_diff'] = db_df['timestamp'].diff().dt.total_seconds().fillna(0)
        
        # Activity-aware grouping: use should_break_segment for intelligent breaks
        # This prevents fragmentation of sleep periods by low_confidence/inactive
        prev_activities = db_df['activity_type'].shift().fillna('')
        
        # Vectorized check for segment breaks
        db_df['should_break'] = [
            should_break_segment(prev, curr, gap) 
            for prev, curr, gap in zip(prev_activities, db_df['activity_type'], db_df['time_diff'])
        ]
        
        # First row always starts a new group
        db_df.iloc[0, db_df.columns.get_loc('should_break')] = True
        
        # Cumulative sum creates group IDs
        db_df['group'] = db_df['should_break'].cumsum()
        
        segments = []
        for group_id, group_df in db_df.groupby('group'):
            # Use the first activity in the group for labeling
            # Post-processing (merge_sleep_segments) handles intelligent merging
            activity = group_df['activity_type'].iloc[0]
            start_time = group_df['timestamp'].min()
            end_time = group_df['timestamp'].max() + pd.Timedelta(seconds=10)
            
            # Check if any row in this segment is corrected
            has_correction = group_df['is_corrected'].any() if 'is_corrected' in group_df.columns else False
            
            # Validate activity for room (unless corrected)
            if has_correction:
                # Keep corrected activity as-is (user knows what they're doing)
                validated_activity = activity
            else:
                validated_activity = validate_activity_for_room(activity, room)
            
            duration_minutes = (end_time - start_time).total_seconds() / 60.0
            avg_confidence = group_df['confidence'].mean() if 'confidence' in group_df.columns else 1.0
            event_count = len(group_df)
            
            # Activity-specific max duration limits (in minutes)
            # Sleep and out can legitimately span many hours
            MAX_DURATION = {
                'sleep': 720,      # 12 hours - normal overnight sleep
                'nap': 240,        # 4 hours - daytime nap
                'out': 1440,       # 24 hours - gone all day
                'inactive': 480,   # 8 hours - person may be inactive for extended periods
                'default': 240     # 4 hours - other activities
            }
            max_allowed = MAX_DURATION.get(validated_activity.lower(), MAX_DURATION['default'])
            
            # Split overly long segments into chunks instead of skipping
            if duration_minutes > max_allowed:
                logger.info(f"Splitting long segment {validated_activity} in {room}: {duration_minutes:.1f} min into {max_allowed} min chunks")
                
                # Create chunked segments
                chunk_start = start_time
                remaining_minutes = duration_minutes
                chunk_count = 0
                
                while remaining_minutes > 0:
                    chunk_duration = min(remaining_minutes, max_allowed)
                    chunk_end = chunk_start + pd.Timedelta(minutes=chunk_duration)
                    chunk_count += 1
                    
                    segments.append({
                        'elder_id': elder_id,
                        'room': room,
                        'activity_type': validated_activity,
                        'start_time': chunk_start.strftime('%Y-%m-%d %H:%M:%S'),
                        'end_time': chunk_end.strftime('%Y-%m-%d %H:%M:%S'),
                        'duration_minutes': round(chunk_duration, 2),
                        'avg_confidence': round(avg_confidence, 3) if not pd.isna(avg_confidence) else 1.0,
                        'event_count': int(event_count * (chunk_duration / duration_minutes)),
                        'record_date': record_date,
                        'is_corrected': 1 if has_correction else 0,
                        'correction_source': 'manual' if has_correction else None
                    })
                    
                    chunk_start = chunk_end
                    remaining_minutes -= chunk_duration
                
                logger.info(f"Created {chunk_count} segments from long {validated_activity} period")
            else:
                segments.append({
                    'elder_id': elder_id,
                    'room': room,  # Preserve original case
                    'activity_type': validated_activity,
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_minutes': round(duration_minutes, 2),
                    'avg_confidence': round(avg_confidence, 3) if not pd.isna(avg_confidence) else 1.0,
                    'event_count': event_count,
                    'record_date': record_date,
                    'is_corrected': 1 if has_correction else 0,
                    'correction_source': 'manual' if has_correction else None
                })
        
        # Post-process: merge adjacent sleep segments with brief interruptions
        segments = merge_sleep_segments(segments)
        
        # Delete old segments and insert new
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM activity_segments 
            WHERE elder_id = ? 
              AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
              AND record_date = ?
        ''', (elder_id, normalized_room, record_date))
        
        for seg in segments:
            cursor.execute('''
                INSERT INTO activity_segments 
                (elder_id, room, activity_type, start_time, end_time, 
                 duration_minutes, avg_confidence, event_count, record_date,
                 is_corrected, correction_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                seg['elder_id'], seg['room'], seg['activity_type'],
                seg['start_time'], seg['end_time'], seg['duration_minutes'],
                seg['avg_confidence'], seg['event_count'], seg['record_date'],
                seg['is_corrected'], seg['correction_source']
            ))
        
        conn.commit()
        logger.info(f"Regenerated {len(segments)} segments for {elder_id}/{room}/{record_date} (validated activities)")
        return len(segments)
        
    finally:
        if close_conn and conn_ctx:
            # If we created the context, exit it (commits/closes)
            conn_ctx.__exit__(None, None, None)
