import sys
import os
import json
import pandas as pd
import logging
import fcntl
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Load env variables (important for DB connection)
load_dotenv()

# Import Configuration
from elderlycare_v1_16.config.settings import DATA_ROOT, RAW_DATA_DIR, ARCHIVE_DATA_DIR

# Import Core Platform & ML
try:
    from elderlycare_v1_16.platform import ElderlyCarePlatform
    from ml.pipeline import UnifiedPipeline
except ImportError as e:
    logger.error(f"Failed to import local elderlycare modules: {e}")
    sys.exit(1)

# Import New Services
from elderlycare_v1_16.services.profile_service import ProfileService
from elderlycare_v1_16.services.sleep_service import SleepService
from elderlycare_v1_16.services.adl_service import ADLService
from elderlycare_v1_16.services.insight_service import InsightService 
from ml.household_analyzer import HouseholdAnalyzer 
from ml.sleep_analyzer import SleepAnalyzer
from utils.elder_id_utils import (
    apply_canonical_alias_map,
    choose_canonical_elder_id,
    elder_id_lineage_matches,
    parse_elder_id_from_filename,
)

# Intelligence Phase Add-Ons (Beta 5)
try:
    from intelligence.fusion import run_trajectory_analysis
    from intelligence.patterns import run_pattern_analysis
    from intelligence.context import run_context_analysis
    INTELLIGENCE_ENABLED = True
except ImportError as e:
    INTELLIGENCE_ENABLED = False
    logger.warning(f"Intelligence modules not available: {e}")


# Legacy imports (if needed for transition)
# from data_manager import DataManager 

def _known_elder_ids_for_canonicalization() -> set[str]:
    """
    Discover existing elder IDs so incoming filename aliases can be canonicalized
    (for example HK0011_jessica -> HK001_jessica when HK001_jessica already exists).
    """
    known: set[str] = set()
    backend_dir = Path(__file__).resolve().parent

    for models_root in (
        backend_dir / "models",
        backend_dir / "models_beta6_registry_v2",
    ):
        if not models_root.exists():
            continue
        for child in models_root.iterdir():
            if child.is_dir():
                name = str(child.name).strip()
                if name:
                    known.add(name)

    for data_root in (RAW_DATA_DIR, ARCHIVE_DATA_DIR):
        try:
            root_path = Path(data_root)
        except Exception:
            continue
        if not root_path.exists():
            continue
        for suffix in (".xlsx", ".xls", ".parquet"):
            for file_path in root_path.rglob(f"*{suffix}"):
                if not file_path.is_file():
                    continue
                if "_train" not in file_path.name.lower():
                    continue
                inferred = parse_elder_id_from_filename(file_path.name)
                if inferred:
                    known.add(inferred)

    return known


def get_elder_id_from_filename(filename: str) -> str:
    parsed = apply_canonical_alias_map(parse_elder_id_from_filename(filename))
    known = _known_elder_ids_for_canonicalization()
    lineage_candidates = [parsed]
    lineage_candidates.extend(
        apply_canonical_alias_map(candidate)
        for candidate in sorted(known)
        if elder_id_lineage_matches(parsed, candidate)
    )
    canonical = choose_canonical_elder_id(lineage_candidates)
    return apply_canonical_alias_map(canonical or parsed)


def _env_enabled(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _normalize_activity_token(value) -> str:
    return str(value or "").strip().lower()


def _infer_occupancy_prob(activity_label: str, confidence: float) -> float:
    label = _normalize_activity_token(activity_label)
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.5
    conf = float(np.clip(conf, 0.0, 1.0))
    if label in {"unoccupied", "inactive", "out"}:
        return float(np.clip(1.0 - conf, 0.0, 1.0))
    if label in {"unknown", "low_confidence"}:
        return 0.5
    return float(np.clip(max(conf, 0.5), 0.0, 1.0))


def _pre_persistence_arbitration_enabled() -> bool:
    if os.getenv("ENABLE_PRE_PERSISTENCE_ARBITRATION") is not None:
        return _env_enabled("ENABLE_PRE_PERSISTENCE_ARBITRATION", default=False)
    return _env_enabled("ENABLE_BETA6_AUTHORITY", default=False)


def _apply_event_decoder_stability(room_name: str, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Optional room-level stability smoothing before cross-room arbitration.
    """
    if not _env_enabled("ENABLE_PRE_PERSISTENCE_EVENT_DECODER", default=False):
        return pred_df, 0
    if not {"timestamp", "predicted_activity"}.issubset(pred_df.columns):
        return pred_df, 0
    if "predicted_top1_label" not in pred_df.columns or "predicted_top1_prob" not in pred_df.columns:
        return pred_df, 0

    try:
        from ml.event_decoder import DecoderConfig, EventDecoder

        n = len(pred_df)
        if n == 0:
            return pred_df, 0

        timestamps = pd.to_datetime(pred_df["timestamp"], errors="coerce")
        if timestamps.isna().all():
            return pred_df, 0

        labels: set[str] = set(
            _normalize_activity_token(x)
            for x in pred_df.get("predicted_top1_label", [])
            if _normalize_activity_token(x)
        )
        labels.update(
            _normalize_activity_token(x)
            for x in pred_df.get("predicted_top2_label", [])
            if _normalize_activity_token(x)
        )
        labels.update(
            _normalize_activity_token(x)
            for x in pred_df.get("predicted_activity", [])
            if _normalize_activity_token(x)
        )
        labels.discard("unoccupied")
        if not labels:
            return pred_df, 0

        activity_probs: dict[str, np.ndarray] = {
            label: np.zeros(n, dtype=float) for label in sorted(labels)
        }
        top1_prob = pd.to_numeric(pred_df.get("predicted_top1_prob"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        top2_prob = pd.to_numeric(pred_df.get("predicted_top2_prob"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        confidence = pd.to_numeric(pred_df.get("confidence"), errors="coerce").fillna(0.5).to_numpy(dtype=float)

        top1_label = pred_df.get("predicted_top1_label", pd.Series(index=pred_df.index, dtype=object)).astype(str).str.strip().str.lower()
        top2_label = pred_df.get("predicted_top2_label", pd.Series(index=pred_df.index, dtype=object)).astype(str).str.strip().str.lower()
        pred_label = pred_df.get("predicted_activity", pd.Series(index=pred_df.index, dtype=object)).astype(str).str.strip().str.lower()

        for idx in range(n):
            label1 = _normalize_activity_token(top1_label.iloc[idx])
            label2 = _normalize_activity_token(top2_label.iloc[idx])
            if label1 in activity_probs:
                activity_probs[label1][idx] = max(activity_probs[label1][idx], float(np.clip(top1_prob[idx], 0.0, 1.0)))
            if label2 in activity_probs:
                activity_probs[label2][idx] = max(activity_probs[label2][idx], float(np.clip(top2_prob[idx], 0.0, 1.0)))
            chosen = _normalize_activity_token(pred_label.iloc[idx])
            if chosen in activity_probs:
                activity_probs[chosen][idx] = max(activity_probs[chosen][idx], float(np.clip(confidence[idx], 0.0, 1.0)))

        occupancy_probs = np.asarray(
            [
                _infer_occupancy_prob(activity_label=pred_label.iloc[idx], confidence=confidence[idx])
                for idx in range(n)
            ],
            dtype=float,
        )

        decoder = EventDecoder(
            DecoderConfig(
                occupancy_on_threshold=float(os.getenv("PRE_PERSIST_DECODER_ON_THRESHOLD", "0.60")),
                occupancy_off_threshold=float(os.getenv("PRE_PERSIST_DECODER_OFF_THRESHOLD", "0.40")),
                hysteresis_min_windows=max(1, int(os.getenv("PRE_PERSIST_DECODER_MIN_WINDOWS", "2"))),
                use_unknown_fallback=True,
            )
        )
        decoded = decoder.decode_to_dataframe(
            occupancy_probs=occupancy_probs,
            activity_probs=activity_probs,
            timestamps=[pd.Timestamp(ts).to_pydatetime() for ts in timestamps],
            room_name=room_name,
        )
        if decoded.empty or "predicted_label" not in decoded.columns:
            return pred_df, 0

        out = pred_df.copy()
        new_labels = decoded["predicted_label"].astype(str).to_numpy(dtype=object)
        old_labels = out["predicted_activity"].astype(str).to_numpy(dtype=object)
        if len(new_labels) != len(old_labels):
            return pred_df, 0
        out["predicted_activity"] = new_labels
        if "confidence" in out.columns and "confidence" in decoded.columns:
            out["confidence"] = pd.to_numeric(decoded["confidence"], errors="coerce").fillna(out["confidence"])
        changes = int(np.sum(new_labels != old_labels))
        return out, changes
    except Exception as e:
        logger.warning(f"Pre-persistence EventDecoder skipped for {room_name}: {e}")
        return pred_df, 0


def _apply_pre_persistence_arbitration(prediction_results: dict) -> tuple[dict, dict]:
    """
    Resolve cross-room contradictions before persisting ADL rows.
    """
    if not isinstance(prediction_results, dict) or not prediction_results:
        return prediction_results, {"status": "skipped", "reason": "empty_prediction_results"}

    adjusted: dict = {}
    decoder_changes = 0
    for room_name, pred_df in prediction_results.items():
        if not isinstance(pred_df, pd.DataFrame):
            adjusted[room_name] = pred_df
            continue
        if not {"timestamp", "predicted_activity"}.issubset(pred_df.columns):
            adjusted[room_name] = pred_df
            continue
        room_df = pred_df.copy()
        room_df["timestamp"] = pd.to_datetime(room_df["timestamp"], errors="coerce")
        room_df = room_df[room_df["timestamp"].notna()].copy()
        if room_df.empty:
            adjusted[room_name] = room_df
            continue
        room_df, room_changes = _apply_event_decoder_stability(room_name, room_df)
        decoder_changes += int(room_changes)
        adjusted[room_name] = room_df

    fusion_inputs: dict[str, pd.DataFrame] = {}
    timestamp_set: set[pd.Timestamp] = set()
    for room_name, room_df in adjusted.items():
        if not isinstance(room_df, pd.DataFrame) or room_df.empty:
            continue
        if not {"timestamp", "predicted_activity"}.issubset(room_df.columns):
            continue
        confidence_series = pd.to_numeric(room_df.get("confidence"), errors="coerce").fillna(0.5)
        fusion_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(room_df["timestamp"], errors="coerce"),
                "predicted_label": room_df["predicted_activity"].astype(str),
                "confidence": confidence_series.astype(float),
            }
        ).dropna(subset=["timestamp"])
        if fusion_df.empty:
            continue
        fusion_df["occupancy_prob"] = fusion_df.apply(
            lambda row: _infer_occupancy_prob(row.get("predicted_label"), row.get("confidence")),
            axis=1,
        )
        fusion_df = fusion_df.sort_values("timestamp", kind="stable").reset_index(drop=True)
        fusion_inputs[room_name] = fusion_df
        timestamp_set.update(pd.to_datetime(fusion_df["timestamp"], errors="coerce").dropna().tolist())

    if len(fusion_inputs) < 2 or not timestamp_set:
        return adjusted, {
            "status": "skipped",
            "reason": "insufficient_room_inputs",
            "rooms": sorted(list(fusion_inputs.keys())),
            "decoder_changes": int(decoder_changes),
        }

    try:
        from ml.home_empty_fusion import HomeEmptyFusion

        fusion = HomeEmptyFusion()
        timestamps = sorted(pd.Timestamp(ts).to_pydatetime() for ts in timestamp_set)
        fused_predictions = fusion.fuse(fusion_inputs, timestamps)
    except Exception as e:
        logger.warning(f"Pre-persistence HomeEmptyFusion skipped: {e}")
        return adjusted, {
            "status": "error",
            "reason": "fusion_error",
            "error": f"{type(e).__name__}: {e}",
            "decoder_changes": int(decoder_changes),
        }

    non_exclusive_labels = {"inactive", "unoccupied", "out", "unknown", "low_confidence"}
    contradictions = 0
    adjustments = 0
    alignment_seconds = max(1, int(os.getenv("PRE_PERSIST_ARBITRATION_ALIGNMENT_SECONDS", "15")))

    for item in fused_predictions:
        ts = pd.Timestamp(item.timestamp)
        occupied_states = []
        for state in item.room_states:
            label = _normalize_activity_token(state.activity_label)
            if not bool(state.is_occupied):
                continue
            if label in non_exclusive_labels:
                continue
            occupied_states.append(state)

        if len(occupied_states) <= 1:
            continue

        contradictions += 1
        winner = max(
            occupied_states,
            key=lambda state: (float(state.occupancy_prob), float(state.confidence)),
        )

        for state in occupied_states:
            if state.room_name == winner.room_name:
                continue
            room_df = adjusted.get(state.room_name)
            if not isinstance(room_df, pd.DataFrame) or room_df.empty:
                continue
            deltas = (pd.to_datetime(room_df["timestamp"], errors="coerce") - ts).abs()
            if deltas.isna().all():
                continue
            nearest_idx = deltas.idxmin()
            nearest_delta = deltas.loc[nearest_idx]
            if pd.isna(nearest_delta) or nearest_delta > pd.Timedelta(seconds=alignment_seconds):
                continue
            current_label = _normalize_activity_token(room_df.at[nearest_idx, "predicted_activity"])
            if current_label in non_exclusive_labels:
                continue
            room_df.at[nearest_idx, "predicted_activity"] = "unoccupied"
            if "confidence" in room_df.columns:
                room_df.at[nearest_idx, "confidence"] = min(
                    float(pd.to_numeric(room_df.at[nearest_idx, "confidence"], errors="coerce") or 0.5),
                    0.5,
                )
            adjustments += 1

    return adjusted, {
        "status": "ok",
        "rooms": sorted(list(fusion_inputs.keys())),
        "timestamps": int(len(timestamp_set)),
        "contradiction_timestamps": int(contradictions),
        "adjustments": int(adjustments),
        "decoder_changes": int(decoder_changes),
    }

def generate_activity_segments(elder_id: str, prediction_results: dict):
    """
    Generate activity_segments from adl_history data.
    
    IMPORTANT: Queries adl_history directly for ALL rooms to ensure no rooms
    are missed (prediction_results may be incomplete or filtered).
    Updated to handle MULTIPLE dates if the file spans efficiently.
    """
    from utils.segment_utils import regenerate_segments
    from backend.db.legacy_adapter import LegacyDatabaseAdapter
    
    # Extract ALL unique dates from prediction_results
    unique_dates = set()
    try:
        for room_name, pred_df in prediction_results.items():
            if 'timestamp' in pred_df.columns and not pred_df.empty:
                # Ensure datetime conversion
                if not pd.api.types.is_datetime64_any_dtype(pred_df['timestamp']):
                    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
                
                # Get unique dates
                dates = pred_df['timestamp'].dt.date.unique()
                for d in dates:
                    unique_dates.add(d.isoformat())
    except Exception as e:
        logger.warning(f"Error extracting dates for segment generation: {e}")
    
    if not unique_dates:
        logger.warning("No valid timestamps found in prediction_results, skipping segment generation")
        return
    
    logger.info(f"Targeting segment regeneration for dates: {sorted(list(unique_dates))}")
    
    try:
        with LegacyDatabaseAdapter().get_connection() as conn:
            cursor = conn.cursor()
            
            # For each unique date, regenerate segments
            for record_date in sorted(list(unique_dates)):
                
                # Query adl_history directly for ALL rooms with data for this elder/date
                cursor.execute('''
                    SELECT DISTINCT room FROM adl_history 
                    WHERE elder_id = ? AND record_date = ?
                ''', (elder_id, record_date))
                rooms_in_db = [row[0] for row in cursor.fetchall()]
                
                if not rooms_in_db:
                    continue

                logger.info(f"Regenerating segments for {record_date} ({len(rooms_in_db)} rooms)")
                
                for room_name in rooms_in_db:
                    try:
                        count = regenerate_segments(elder_id, room_name, record_date, conn)
                        # logger.info(f"  -> Generated {count} segments for {room_name}")
                    except Exception as e:
                        logger.error(f"Failed to generate segments for {room_name} on {record_date}: {e}")
                        
    except Exception as e:
        logger.error(f"Failed to query rooms from adl_history: {e}")

def process_file(file_path: Path):
    logger.info(f"Processing file: {file_path}")
    filename = file_path.name
    elder_id = get_elder_id_from_filename(filename)
    logger.info(f"Identified Elder ID: {elder_id}")

    # Initialize Services
    profile_svc = ProfileService()
    sleep_svc = SleepService()
    adl_svc = ADLService()
    insight_svc = InsightService()

    # 1. Profile Check (Ensure exists)
    # If not exists, we create a stub so FKs work
    if not profile_svc.get_profile(elder_id):
        logger.info(f"Checking profile for {elder_id}: Creating default.")
        profile_svc.create_or_update_elder(elder_id, {
            "personal_info": {"full_name": elder_id}
        })
    
    # 2. ML Pipeline execution
    pipeline = UnifiedPipeline()
    prediction_results = {}
    platform = None
    
    try:
        from utils.data_loader import load_sensor_data
        
        # Check for pre-computed predictions
        is_precomputed = False
        try:
            # We don't want to load specific rows for parquet easily without pyarrow/fastparquet specific kwargs
            # So we load the file using our loader which handles details
            loaded_data = load_sensor_data(file_path)
            
            # Check if any room has 'predicted_activity'
            for df in loaded_data.values():
                if 'predicted_activity' in df.columns:
                    is_precomputed = True
                    break
        except Exception as e:
            logger.warning(f"Could not preview file content: {e}")

        if is_precomputed:
             logger.info("File appears to contain pre-computed predictions. Using as-is.")
             platform = ElderlyCarePlatform()
             # Platform might need update too? platform.load_excel_data is exclusive for Excel?
             # Let's assume prediction_results IS the loaded_data for now
             prediction_results = loaded_data
        else:
             if "_train" in filename.lower():
                  prediction_results, _ = pipeline.train_and_predict(str(file_path), elder_id)
                  # Ground Truth is now handled directly in pipeline.train_and_predict()
                  # (Beta 5.5: uses 'activity' column directly, no prediction step for training files)
             else:
                  prediction_results = pipeline.predict(str(file_path), elder_id)
             platform = pipeline.platform
        
        # 3. Save Predictions to DB as ADL Events
        if prediction_results:
            logger.info(f"Predictions generated for {len(prediction_results)} rooms")
            if _pre_persistence_arbitration_enabled():
                try:
                    prediction_results, arbitration_report = _apply_pre_persistence_arbitration(
                        prediction_results
                    )
                    logger.info(
                        "Pre-persistence arbitration applied for %s: %s",
                        elder_id,
                        arbitration_report,
                    )
                except Exception as e:
                    logger.exception(
                        "Pre-persistence arbitration failed for %s: %s",
                        elder_id,
                        e,
                    )
                    if _env_enabled("PRE_PERSISTENCE_ARBITRATION_FAIL_CLOSED", default=False):
                        raise
                    logger.warning(
                        "Continuing with original prediction persistence path because "
                        "PRE_PERSISTENCE_ARBITRATION_FAIL_CLOSED=false."
                    )
            
            # Save each room's predictions to adl_events table
            for room_name, pred_df in prediction_results.items():
                if not isinstance(pred_df, pd.DataFrame):
                    logger.warning(f"Skipping {room_name}: predictions are not a DataFrame (got {type(pred_df)})")
                    continue
                    
                if 'timestamp' in pred_df.columns and 'predicted_activity' in pred_df.columns:
                    # Save ALL predictions (including "inactive") to capture room usage patterns
                    # UI can filter "inactive" if needed via timeline controls
                    active_df = pred_df.copy()
                    
                    logger.info(f"Saving {len(active_df)} ADL events for {room_name} (including inactive)")
                    for _, row in active_df.iterrows():
                        # Convert timestamp to string for SQLite
                        ts = row['timestamp']
                        timestamp_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
                        
                        # Prepare event data including sensor columns for DB persistence
                        event = {
                            'room': room_name,
                            'activity': row['predicted_activity'],
                            'timestamp': timestamp_str,
                            'confidence': row.get('confidence', 1.0)
                        }

                        # Carry model suggestion metadata for low-confidence review UI.
                        hint_cols = [
                            'predicted_top1_label',
                            'predicted_top1_prob',
                            'predicted_top2_label',
                            'predicted_top2_prob',
                            'low_confidence_threshold',
                            'low_confidence_hint_label',
                            'is_low_confidence',
                        ]
                        for col in hint_cols:
                            if col in row and pd.notna(row[col]):
                                event[col] = row[col]
                        
                        # Add sensor columns if present in the row
                        sensor_cols = ['motion', 'temperature', 'light', 'sound', 'co2', 'humidity']
                        for col in sensor_cols:
                            if col in row:
                                event[col] = row[col]
                            elif col in active_df.columns:
                                event[col] = active_df.loc[row.name, col] if row.name in active_df.index else None

                        try:
                            adl_svc.save_adl_event(elder_id, event)
                        except Exception as e:
                            logger.error(f"Failed to save ADL event: {e}")
                            continue

            # Generate activity segments for efficient timeline display
            generate_activity_segments(elder_id, prediction_results)
            
            # Run Data Integrity Check (Post-Processing Hook)
            try:
                from scripts.check_integrity import run_integrity_check
                record_date = None
                # Extract date from first prediction result
                for room_df in prediction_results.values():
                    if isinstance(room_df, pd.DataFrame) and 'timestamp' in room_df.columns and len(room_df) > 0:
                        record_date = pd.to_datetime(room_df['timestamp'].iloc[0]).strftime('%Y-%m-%d')
                        break
                
                if record_date and not run_integrity_check(elder_id, record_date):
                    logger.warning(f"⚠️ Data integrity issues detected for {elder_id} on {record_date}. Check logs.")
                elif record_date:
                    logger.info(f"✓ Data integrity check passed for {elder_id}")
            except Exception as e:
                logger.error(f"Integrity check failed: {e}")

    except Exception as e:
        logger.error(f"Error in ML pipeline: {e}", exc_info=True)
        # Continue to archiving



    # 4. Run Analyzers & Save to Services
    if prediction_results:
        logger.info("Running downstream analyzers and saving to DB...")
        target_date = None
        for _, df in prediction_results.items():
            if isinstance(df, pd.DataFrame) and 'timestamp' in df.columns and not df.empty:
                target_date = pd.to_datetime(df['timestamp'].iloc[0]).strftime('%Y-%m-%d')
                break
        
        # Generate Sleep Analysis using centralized SleepAnalyzer
        logger.info("Generating sleep analysis (Standardized)...")
        try:
            analyzer = SleepAnalyzer()
            
            # --- Motion Data Normalization (External Review Recommendation) ---
            # Use centralized MotionDataNormalizer for consistent handling across pipelines
            try:
                from utils.motion_normalizer import MotionDataNormalizer
                from utils.data_loader import load_sensor_data
                raw_data_source = load_sensor_data(file_path)
                
                for room, pred_df in prediction_results.items():
                    if not pred_df.empty:
                        raw_df = raw_data_source.get(room)
                        normalized_df, quality = MotionDataNormalizer.normalize_for_sleep_analysis(
                            pred_df,
                            source=f"process_data:{room}",
                            raw_data_source=raw_df
                        )
                        prediction_results[room] = normalized_df
            except Exception as e:
                logger.warning(f"Motion normalization failed: {e}. Sleep analysis may use heuristics.")
            # -----------------------------------------------------------

            sleep_analysis = analyzer.analyze_from_predictions(elder_id, prediction_results)
            
            if sleep_analysis:
                date_str = target_date or pd.Timestamp.now().strftime("%Y-%m-%d")
                sleep_svc.save_sleep_analysis(elder_id, sleep_analysis, date_str)
                logger.info(f"Saved standardized sleep analysis: {sleep_analysis['total_duration_hours']}h, Quality: {sleep_analysis['quality_score']}")
            else:
                logger.warning("No sleep events found in prediction results.")
        except Exception as e:
            logger.error(f"Error in SleepAnalyzer: {e}", exc_info=True)
        
        # Generate ICOPE Assessment from ADL data
        logger.info("Generating ICOPE assessment from ADL data...")
        try:
            from elderlycare_v1_16.services.icope_service import ICOPEService
            icope_svc = ICOPEService()
            icope_result = icope_svc.calculate_and_save(elder_id, prediction_results)
            if icope_result:
                logger.info(f"ICOPE Assessment saved: Overall={icope_result['overall_score']}, Trend={icope_result['trend']}")
        except Exception as e:
            logger.error(f"Error generating ICOPE assessment: {e}", exc_info=True)
        
        # Run Health Insights Engine
        logger.info("Running Health Insights Engine...")
        try:
            alerts = insight_svc.run_daily_analysis(elder_id, analysis_date=target_date)
            if alerts:
                logger.info(f"Generated {len(alerts)} health alerts.")
        except Exception as e:
            logger.error(f"Error running insights: {e}")
            
        # Run Household Analysis (Empty Home Detection)
        logger.info("Running Household Analysis (Global State)...")
        try:
            h_analyzer = HouseholdAnalyzer()
            
            if target_date:
                h_analyzer.analyze_day(elder_id, target_date)
            else:
                logger.warning("Could not determine date for Household Analysis.")
                
        except Exception as e:
            logger.error(f"Error running Household Analysis: {e}", exc_info=True)
        # ============================================================
        # INTELLIGENCE PHASE: Trajectory Analysis (Beta 5 Add-On)
        # ============================================================
        if INTELLIGENCE_ENABLED and target_date:
            logger.info("Running Trajectory Analysis (Cross-Room Tracking)...")
            try:
                trajectories = run_trajectory_analysis(elder_id, target_date)
                if trajectories:
                    logger.info(f"Generated {len(trajectories)} movement trajectories.")
            except Exception as e:
                logger.error(f"Error running Trajectory Analysis: {e}", exc_info=True)
        # ============================================================
        # INTELLIGENCE PHASE: Pattern Analysis (Beta 5 Add-On)
        # ============================================================
        if INTELLIGENCE_ENABLED and target_date:
            logger.info("Running Pattern Analysis (Routine Anomaly Detection)...")
            try:
                anomalies = run_pattern_analysis(elder_id, target_date)
                if anomalies:
                    logger.info(f"Detected {len(anomalies)} routine anomalies.")
                    for a in anomalies:
                        logger.info(f"  - {a['description']}: {a['observed_value']} (normal: {a['baseline_value']})")
            except Exception as e:
                logger.error(f"Error running Pattern Analysis: {e}", exc_info=True)




    
    # 7. Archive (always archive, even if processing had issues)
    archive_file(file_path, ARCHIVE_DATA_DIR)

def archive_file(file_path: Path, archive_root: Path):
    """
    Archive an input file to the archive directory.
    Converts Excel files to Parquet format for storage efficiency (~90% size reduction).
    Implements hash-based deduplication to avoid storing identical files.
    
    HARDENED: Added file existence checks and improved error handling to prevent
    'File not found' errors during concurrent or sequential processing.
    """
    # Early exit if file doesn't exist (may have been processed by another instance)
    if not file_path.exists():
        logger.warning(f"File already processed or missing: {file_path}")
        return
    
    try:
        from utils.data_loader import load_sensor_data, save_to_parquet
        import hashlib
        
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        dest_dir = archive_root / today_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert xlsx to parquet for storage efficiency
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            # Load the Excel data
            data = load_sensor_data(file_path)
            
            # Determine parquet filename
            parquet_name = file_path.stem + '.parquet'
            dest_path = dest_dir / parquet_name
            
            # Optimized deduplication: Only check if destination exists
            if dest_path.exists():
                # Simple content comparison instead of hashing all files
                if _compare_parquet_content(data, dest_path):
                    logger.info(f"Duplicate detected: {file_path.name} matches existing archive. Skipping.")
                    try:
                        file_path.unlink()
                    except FileNotFoundError:
                        logger.warning(f"File already removed: {file_path}")
                    return
                
                # Not a duplicate, create timestamped version
                timestamp = pd.Timestamp.now().strftime("%H%M%S")
                dest_path = dest_dir / f"{file_path.stem}_{timestamp}.parquet"
            
            # Save as parquet
            save_to_parquet(data, dest_path)
            logger.info(f"Archived as Parquet: {dest_path}")
            
            # Delete original xlsx with safety check
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted original xlsx: {file_path}")
            except FileNotFoundError:
                logger.warning(f"Original file already removed: {file_path}")
        else:
            # For non-Excel files, just move them
            import shutil
            dest_path = dest_dir / file_path.name
            if dest_path.exists():
                # Simple hash comparison for non-Excel
                try:
                    new_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    existing_hash = hashlib.sha256(dest_path.read_bytes()).hexdigest()
                    if new_hash == existing_hash:
                        logger.info(f"Duplicate detected: {file_path.name}. Skipping archive.")
                        try:
                            file_path.unlink()
                        except FileNotFoundError:
                            pass
                        return
                except FileNotFoundError:
                    logger.warning(f"File comparison failed, proceeding with archive")
                timestamp = pd.Timestamp.now().strftime("%H%M%S")
                dest_path = dest_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            try:
                if file_path.exists():
                    shutil.move(str(file_path), str(dest_path))
                    logger.info(f"Archived file to {dest_path}")
            except FileNotFoundError:
                logger.warning(f"Source file not found during move: {file_path}")
            
    except Exception as e:
        logger.error(f"Failed to archive file {file_path}: {e}")

def _compare_parquet_content(new_data: dict, existing_path: Path) -> bool:
    """Compare loaded data dict with existing parquet file content."""
    try:
        from utils.data_loader import load_sensor_data
        existing_data = load_sensor_data(existing_path)
        
        # Compare keys
        if set(new_data.keys()) != set(existing_data.keys()):
            return False
        
        # Compare row counts per room
        for room in new_data.keys():
            if len(new_data[room]) != len(existing_data[room]):
                return False
        
        return True
    except:
        return False


def main():
    """
    Main entry point for the data processing service.
    
    HARDENED: Added file locking to prevent multiple instances from processing
    the same files simultaneously, which can cause 'File not found' errors.
    """
    lock_file_path = RAW_DATA_DIR / ".process_data.lock"
    lock_file = None
    
    try:
        # Acquire exclusive lock
        lock_file = open(lock_file_path, 'w')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.info("Acquired process lock.")
        except IOError:
            logger.warning("Another instance of process_data.py is already running. Exiting.")
            return
        
        logger.info("Starting Beta 5 Data Processing Service (DB-Centric)...")
        
        # Support both Excel and Parquet
        files = list(RAW_DATA_DIR.glob("*.xlsx")) + list(RAW_DATA_DIR.glob("*.parquet"))
        
        if not files:
            logger.warning(f"No .xlsx or .parquet files found in {RAW_DATA_DIR}")
            return
            
        # Sort files: Training files first, then by name/date
        def file_sort_key(f):
            name = f.name.lower()
            # "train" priority: 0, others: 1
            priority = 0 if "_train" in name else 1
            return (priority, name)
            
        files.sort(key=file_sort_key)
        
        for file in files:
            # Double-check file exists before processing (may have been handled by previous iteration)
            if file.exists():
                process_file(file)
            else:
                logger.info(f"Skipping already processed file: {file}")
    
    finally:
        # Release lock
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                lock_file_path.unlink(missing_ok=True)
                logger.info("Released process lock.")
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")

if __name__ == "__main__":
    main()
