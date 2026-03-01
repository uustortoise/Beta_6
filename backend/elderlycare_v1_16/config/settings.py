"""
Configuration settings for Elderly Care Platform v2.0 (Beta 5.5 - Transformer Prototype)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Base Paths ---
# Calculate paths relative to this file's location for portability
# Path: settings.py -> config/ -> elderlycare_v1_16/ -> backend/ -> Beta_5.5/
_settings_file = Path(__file__).resolve()
_config_dir = _settings_file.parent          # config/
_elderlycare_dir = _config_dir.parent         # elderlycare_v1_16/
BACKEND_DIR = _elderlycare_dir.parent         # backend/
PROJECT_ROOT = BACKEND_DIR.parent             # Beta_5.5/
BASE_DIR = PROJECT_ROOT                       # Alias for compatibility

# Load environment variables.
# Project-level .env stays supported, but backend/.env should be able to
# override DB settings for local backend runtime consistency.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env", override=True)

# --- Data Paths (Relative to PROJECT_ROOT) ---
# Environment variables can override these, but must be absolute paths
def _resolve_data_path(env_var: str, default_relative: str) -> Path:
    """Resolve a data path from env var or default relative to PROJECT_ROOT."""
    env_value = os.getenv(env_var)
    if env_value:
        p = Path(env_value)
        return p if p.is_absolute() else (PROJECT_ROOT / env_value).resolve()
    return (PROJECT_ROOT / default_relative).resolve()

DATA_ROOT = _resolve_data_path("DATA_ROOT", "data")
RAW_DATA_DIR = _resolve_data_path("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = _resolve_data_path("PROCESSED_DATA_DIR", "data/processed")
ARCHIVE_DATA_DIR = _resolve_data_path("ARCHIVE_DATA_DIR", "data/archive")

# Database Path - Always relative to data/processed
DB_PATH = _resolve_data_path("DB_PATH", "data/processed/residents_master_data.db")

# PostgreSQL Configuration (Feature Flagged)
# POSTGRES_ONLY: When True, disables SQLite entirely and uses PostgreSQL as the sole database.
# Beta 5.5+: PostgreSQL is now the ONLY database. SQLite is fully deprecated.
def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

POSTGRES_ONLY = _env_flag("POSTGRES_ONLY", True)
USE_POSTGRESQL = _env_flag("USE_POSTGRESQL", True)
POSTGRES_CONFIG = {
    'host': os.getenv("POSTGRES_HOST", "localhost"),
    'port': int(os.getenv("POSTGRES_PORT", "5432")),
    'user': os.getenv("POSTGRES_USER", "postgres"),
    'password': os.getenv("POSTGRES_PASSWORD", "password"),
    'dbname': os.getenv("POSTGRES_DB", "elderlycare"),
    'minconn': 1,
    'maxconn': 10
}

# Models Directory
MODELS_DIR = _resolve_data_path("MODEL_DIR", "backend/models")
ANOMALY_MODELS_DIR = MODELS_DIR / "anomaly"

# Logs Directory
LOGS_DIR = BACKEND_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants & Defaults ---

# Data Processing
# Physical sensors that must exist in raw Excel files
DEFAULT_PHYSICAL_SENSORS = [
    'motion', 'temperature', 'light', 'sound', 'co2', 'humidity', 'vibration'
]

# Temporal features (generated from timestamp, not in raw files)
DEFAULT_TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'day_period']

# All sensor columns (physical + temporal) used for model training/prediction
DEFAULT_SENSOR_COLUMNS = DEFAULT_PHYSICAL_SENSORS + DEFAULT_TEMPORAL_FEATURES

DEFAULT_DATA_INTERVAL = '10s'
DEFAULT_SEQUENCE_WINDOW = '10min'  # Extended from 1min to capture longer activity patterns

# Denoising (Beta 5.5: Less aggressive for Transformer - allows attention to handle noise)
DEFAULT_DENOISING_METHOD = 'hampel'
DEFAULT_DENOISING_WINDOW = 3  # 3-sample window (30 seconds) - reduced from 5 to preserve short activity bursts
DEFAULT_DENOISING_THRESHOLD = 4.0  # Increased from 3.0 for Transformer (attention handles moderate noise)

# Model Hyperparameters (Defaults)
DEFAULT_CONV_FILTERS_1 = 64
DEFAULT_CONV_FILTERS_2 = 32
DEFAULT_LSTM_UNITS = 64
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_EPOCHS = 5
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "3.0"))

# Sleep Analysis Configuration
SLEEP_CONFIG = {
    'min_sleep_duration_min': 60,  # Minimum minutes to be considered valid sleep
    'motion_thresholds': {
        'deep': 0.05,    # Motion < 0.05 = Deep Sleep
        'light': 0.2,    # Motion 0.05-0.2 = Light Sleep
        'rem': 0.5       # Motion 0.2-0.5 = REM; >= 0.5 = Awake
    },
    'duration_penalties': {
        'short_severe': 0.7,     # < 5 hours
        'short_moderate': 0.9,   # 5-7 hours
        'long_moderate': 0.95,   # > 9 hours
        'optimal': 1.0           # 7-9 hours
    },
    'fallback_stage_ratios': {
        'Light': 0.55,
        'Deep': 0.15,
        'REM': 0.20,
        'Awake': 0.10
    },
    # Motion Data Requirements (External Review Recommendation)
    # Documents expected motion data format for sleep analysis consistency
    'motion_data_requirements': {
        'expected_range': (0.0, 1.0),  # Raw sensor values expected
        'zscore_detection': {
            'enabled': True,
            'action': 'use_heuristics'  # What to do when Z-score data detected
        },
        'missing_action': 'use_heuristics',  # Fallback when no motion data
        'log_validation': True  # Log validation results for transparency
    }
}

# Activity Analysis
ADL_CONFIDENCE_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD

# Logging
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(BACKEND_DIR / "logs")))
