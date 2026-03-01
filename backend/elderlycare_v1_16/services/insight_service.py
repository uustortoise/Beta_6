import json
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .base_service import BaseService
from .profile_service import ProfileService
from .sleep_service import SleepService
from .adl_service import ADLService

logger = logging.getLogger(__name__)


def _env_enabled(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_float(var_name: str, default: float) -> float:
    raw = os.getenv(var_name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


class InsightService(BaseService):
    """
    The 'Brain' of the platform.
    Analyzes cross-domain data to generate actionable alerts based on configurable rules.
    """
    
    def __init__(self):
        super().__init__()
        self.profile_svc = ProfileService()
        self.sleep_svc = SleepService()
        self.adl_svc = ADLService()

    def run_daily_analysis(self, elder_id: str, analysis_date: Optional[str] = None):
        """
        Run all enabled analysis rules for a specific elder.
        """
        logger.info(f"Running daily analysis for {elder_id}...")
        
        # 1. Gather Context (Snapshot of current state)
        target_date = str(analysis_date).strip() if analysis_date else datetime.now().strftime("%Y-%m-%d")
        profile = self.profile_svc.get_profile(elder_id)
        latest_sleep = self.sleep_svc.get_latest_sleep(elder_id)
        
        context = self._build_context(elder_id, profile, latest_sleep, target_date)
        
        # 2. Fetch Rules
        rules = self._fetch_enabled_rules()
        
        # 3. Execute Rules
        alerts = []
        for rule in rules:
            try:
                if self._evaluate_rule(rule, context):
                    alert = self._create_alert(elder_id, rule, context)
                    alerts.append(alert)
            except Exception as e:
                logger.error(f"Error evaluating rule '{rule['rule_name']}': {e}")

        # 3b. Built-in safety detector: prolonged risky fallen-state signal.
        try:
            fallen_alerts = self._detect_fallen_state_alerts(elder_id, target_date)
            alerts.extend(fallen_alerts)
        except Exception as e:
            logger.error(f"Error evaluating fallen-state alerts for '{elder_id}': {e}")
        
        # 4. Save Alerts
        self._save_alerts(elder_id, alerts, alert_date=target_date)
        return alerts

    def _fetch_enabled_rules(self) -> List[Dict]:
        with self.db.get_connection() as conn:
            columns = (
                'id',
                'rule_name',
                'required_condition',
                'conditions',
                'alert_message',
                'alert_severity',
                'enabled',
                'created_at',
            )
            rows = conn.execute(f"SELECT {', '.join(columns)} FROM alert_rules_v2 WHERE enabled = 1").fetchall()
            rules = []
            for row in rows:
                try:
                    rules.append({col: row[col] for col in columns})
                except (TypeError, KeyError, IndexError):
                    rules.append(dict(zip(columns, row)))
            return rules

    def _build_context(self, elder_id: str, profile: Dict, sleep: Dict, date_str: str) -> Dict:
        """
        Pre-calculate metrics to avoid re-querying for every rule.
        """
        # Sleep Metrics
        sleep_stages = json.loads(sleep.get('sleep_stages', '{}')) if sleep else {}
        deep_pct = sleep_stages.get('Deep', 0.0)
        # Handle decimal vs percentage representation (e.g. 0.15 vs 15.0)
        if deep_pct <= 1.0 and deep_pct > 0: deep_pct *= 100
            
        # ADL Metrics (Queries)
        night_toilet = self._count_adl_event(elder_id, date_str, 'toileting')
        # Assuming 'night_motion' maps to specific ADL types or just activity counts at night
        night_motion = self._count_adl_event(elder_id, date_str, 'motion', is_night=True)
        day_activity = self._count_adl_event(elder_id, date_str, 'motion', is_night=False)
        
        return {
            "profile": profile,
            "metrics": {
                "sleep_duration": float(sleep.get('duration_hours', 0) if sleep else 0),
                "deep_sleep_pct": float(deep_pct),
                "sleep_efficiency": float(sleep.get('efficiency_percent', 0) if sleep else 0),
                "night_toilet_visits": night_toilet,
                "night_motion_events": night_motion,
                "day_activity_count": day_activity
            },
            "elder_id": elder_id,
            "date_str": date_str
        }

    def _evaluate_rule(self, rule: Dict, context: Dict) -> bool:
        # 1. Check Disease Condition
        if rule['required_condition']:
            conditions = context['profile'].get('medical_history', {}).get('chronic_conditions', [])
            # Search partial match case-insensitive
            req = rule['required_condition'].lower()
            validation_list = [c.lower() for c in conditions]
            # Simple check: is the required condition substring in any of the patient's conditions?
            if not any(req in c for c in validation_list):
                return False

        # 2. Parse Conditions JSON
        try:
            cond_data = json.loads(rule['conditions'])
            logic = cond_data.get('logic', 'AND')
            sub_rules = cond_data.get('rules', [])
            
            results = []
            for sub in sub_rules:
                results.append(self._evaluate_condition(sub, context))
            
            if logic == 'AND':
                return all(results)
            else: # OR
                return any(results)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in rule {rule['rule_name']}")
            return False

    def _evaluate_condition(self, condition: Dict, context: Dict) -> bool:
        metric_key = condition['metric']
        operator = condition['operator']
        target_value = condition['value']
        value_type = condition['value_type']
        
        current_value = context['metrics'].get(metric_key, 0)
        
        # Handle Averages
        if value_type.startswith('avg_'):
            days = int(value_type.split('_')[1].replace('d',''))
            avg_value = self._get_historical_average(context['elder_id'], metric_key, days)
            
            # Target becomes the baseline average + the offset percentage/value
            # If target_value is like 10 (meaning +10) or -10 (meaning -10)
            # For simplicity, let's treat target_value as PERCENTAGE deviation for averages
            # e.g. value=50 means > 50% of average? No, usually means > (Avg + 50) or > (Avg * 1.5)
            # Let's interpret value as: The Threshold is Avg + Value. 
            # If metrics are percentages (like deep sleep), Value is absolute diff (Avg 15 + -5 = 10)
            threshold = avg_value + target_value
        else:
            threshold = target_value

        # Comparison
        if operator == 'greater_than': return current_value > threshold
        if operator == 'less_than': return current_value < threshold
        if operator == 'equals': return current_value == threshold
        if operator == 'greater_equal': return current_value >= threshold
        if operator == 'less_equal': return current_value <= threshold
        return False

    def _get_historical_average(self, elder_id: str, metric_key: str, days: int) -> float:
        # This is a placeholder for complex historical queries.
        # Efficient implementation would aggregate in SQL.
        # For now, returning dummy averages to allow code testing without full data population
        # In prod: Query adl_history/sleep_analysis for avg over last X days
        return 10.0 # Dummy baseline

    def _count_adl_event(self, elder_id: str, date_str: str, activity_type: str, is_night: bool = False) -> int:
        # Simplified query
        with self.db.get_connection() as conn:
            # Note: This simply counts rows on that calendar date.
            # Production usually needs time-of-day filtering.
            query = """
                SELECT COUNT(*) as count FROM adl_history 
                WHERE elder_id = ? AND record_date = ? 
            """
            params = [elder_id, date_str]
            
            if activity_type == 'toileting':
                # Support both legacy and canonical labels.
                query += " AND activity_type IN ('toileting', 'toilet')"
            elif activity_type == 'motion':
                # Excluding specific named activities to count general motion? 
                # Or looking for 'Bedroom' activity? Let's assume raw count.
                pass 
                
            row = conn.execute(query, params).fetchone()
            return row[0] if row else 0

    def _create_alert(self, elder_id: str, rule: Dict, context: Dict) -> Dict:
        # Build a message by formatting string
        # e.g. "High risk: {night_toilet_visits} toilet visits"
        msg = rule['alert_message'] or f"Alert triggered: {rule['rule_name']}"
        try:
            # Python string format allows {key} placeholders
            msg = msg.format(**context['metrics'])
        except Exception:
            pass # Keep original if format fails
            
        return {
            "alert_type": "health_risk",
            "severity": rule['alert_severity'],
            "title": rule['rule_name'],
            "message": msg,
            "recommendations": ["Review recent activity", "Check vital signs"]
        }

    def _normalize_token(self, value: Any) -> str:
        return (
            str(value or "")
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

    def _is_lying_like_activity(self, activity_type: Any) -> bool:
        token = self._normalize_token(activity_type)
        if not token:
            return False
        keywords = {
            "lying",
            "lyingdown",
            "liedown",
            "lie",
            "fallen",
            "fall",
            "onfloor",
            "floor",
        }
        return any(k in token for k in keywords)

    def _get_segment_motion_mean(
        self,
        elder_id: str,
        room: str,
        start_time: Any,
        end_time: Any,
    ) -> Optional[float]:
        from utils.room_utils import normalize_room_name

        room_key = normalize_room_name(room)
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT sensor_features
                FROM adl_history
                WHERE elder_id = ?
                  AND LOWER(REPLACE(REPLACE(room, ' ', ''), '_', '')) = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                  AND sensor_features IS NOT NULL
                ORDER BY timestamp
                """,
                (elder_id, room_key, str(start_time), str(end_time)),
            ).fetchall()

        motion_values: List[float] = []
        for row in rows:
            payload_raw = None
            try:
                payload_raw = row["sensor_features"]
            except Exception:
                try:
                    payload_raw = row[0]
                except Exception:
                    payload_raw = None
            if not payload_raw:
                continue
            try:
                payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            motion_raw = payload.get("motion")
            try:
                if motion_raw is not None:
                    motion_values.append(float(motion_raw))
            except (TypeError, ValueError):
                continue

        if not motion_values:
            return None
        return float(sum(motion_values) / len(motion_values))

    def _detect_fallen_state_alerts(self, elder_id: str, target_date: str) -> List[Dict]:
        """
        State-based safety detector:
        room in risky_rooms + lying-like state + sustained duration (+ optional low-motion)
        """
        if not _env_enabled("ENABLE_FALLEN_STATE_ALERTS", default=True):
            return []

        from utils.room_utils import normalize_room_name

        risky_rooms_raw = os.getenv(
            "FALLEN_STATE_RISK_ROOMS",
            "kitchen,bathroom,entrance,livingroom",
        )
        risky_rooms = {
            normalize_room_name(token)
            for token in str(risky_rooms_raw).split(",")
            if str(token).strip()
        }
        warning_minutes = max(1.0, _env_float("FALLEN_STATE_WARNING_MINUTES", 2.0))
        critical_minutes = max(warning_minutes, _env_float("FALLEN_STATE_CRITICAL_MINUTES", 5.0))
        require_low_motion = _env_enabled("FALLEN_STATE_REQUIRE_LOW_MOTION", default=False)
        max_motion = _env_float("FALLEN_STATE_MAX_MOTION", 0.08)

        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT room, activity_type, start_time, end_time, duration_minutes, avg_confidence
                FROM activity_segments
                WHERE elder_id = ?
                  AND record_date = ?
                ORDER BY room, duration_minutes DESC, start_time ASC
                """,
                (elder_id, target_date),
            ).fetchall()

        if not rows:
            return []

        by_room_best: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            try:
                room = row["room"]
                activity_type = row["activity_type"]
                start_time = row["start_time"]
                end_time = row["end_time"]
                duration_minutes = float(row["duration_minutes"] or 0.0)
                avg_confidence = float(row["avg_confidence"] or 0.0)
            except Exception:
                room = row[0]
                activity_type = row[1]
                start_time = row[2]
                end_time = row[3]
                duration_minutes = float(row[4] or 0.0)
                avg_confidence = float(row[5] or 0.0)

            room_key = normalize_room_name(room)
            if room_key not in risky_rooms:
                continue
            if not self._is_lying_like_activity(activity_type):
                continue
            if duration_minutes < warning_minutes:
                continue

            motion_mean = self._get_segment_motion_mean(
                elder_id=elder_id,
                room=room,
                start_time=start_time,
                end_time=end_time,
            )
            if require_low_motion and motion_mean is None:
                continue
            if require_low_motion and motion_mean is not None and motion_mean > max_motion:
                continue

            current = by_room_best.get(room_key)
            if current is None or duration_minutes > float(current["duration_minutes"]):
                by_room_best[room_key] = {
                    "room": str(room),
                    "activity_type": str(activity_type),
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "duration_minutes": float(duration_minutes),
                    "avg_confidence": float(avg_confidence),
                    "motion_mean": motion_mean,
                }

        alerts: List[Dict] = []
        for _, seg in sorted(by_room_best.items(), key=lambda kv: kv[0]):
            duration_minutes = float(seg["duration_minutes"])
            severity = "critical" if duration_minutes >= critical_minutes else "high"
            room_name = str(seg["room"])
            start_txt = str(seg["start_time"])
            motion_mean = seg.get("motion_mean")
            motion_txt = (
                f", avg motion {float(motion_mean):.3f}"
                if motion_mean is not None
                else ""
            )
            alerts.append(
                {
                    "alert_type": "safety_fallen_state",
                    "severity": severity,
                    "title": f"Possible Fall State: Prolonged Lying in {room_name}",
                    "message": (
                        f"Detected a prolonged lying state in {room_name} for {duration_minutes:.1f} minutes "
                        f"starting at {start_txt}{motion_txt}."
                    ),
                    "recommendations": [
                        "Check resident status immediately.",
                        "Call or visit to confirm safety.",
                        "Escalate to emergency contact if unresponsive.",
                    ],
                }
            )
        return alerts

    def _alert_exists(self, conn, elder_id: str, alert_date: str, alert: Dict[str, Any]) -> bool:
        row = conn.execute(
            """
            SELECT id
            FROM alerts
            WHERE elder_id = ?
              AND alert_date = ?
              AND alert_type = ?
              AND severity = ?
              AND title = ?
              AND message = ?
            LIMIT 1
            """,
            (
                elder_id,
                alert_date,
                str(alert.get("alert_type") or "health_risk"),
                str(alert.get("severity") or "medium"),
                str(alert.get("title") or "Alert"),
                str(alert.get("message") or ""),
            ),
        ).fetchone()
        return row is not None

    def _save_alerts(self, elder_id: str, alerts: List[Dict], alert_date: Optional[str] = None):
        if not alerts: return
        target_date = str(alert_date).strip() if alert_date else datetime.now().strftime("%Y-%m-%d")
        with self.db.get_connection() as conn:
            for alert in alerts:
                if self._alert_exists(conn, elder_id, target_date, alert):
                    continue
                conn.execute('''
                    INSERT INTO alerts (elder_id, alert_date, alert_type, severity, title, message, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    elder_id,
                    target_date,
                    alert.get('alert_type', 'health_risk'),
                    alert.get('severity', 'medium'),
                    alert.get('title', 'Alert'),
                    alert.get('message', ''),
                    json.dumps(alert.get('recommendations', []))
                ))
            conn.commit()
        logger.info(f"Generated {len(alerts)} alerts for {elder_id}")
