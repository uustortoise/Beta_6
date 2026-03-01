
import sys
import os
from pathlib import Path

# Ensure we can import from backend
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent / "backend"
sys.path.append(str(backend_dir))

# Mock env var to test consistency
os.environ["DB_PATH"] = "data/processed/TEST_DB.db"

def verify_paths():
    results = {}
    
    try:
        from config import DB_PATH as CONFIG_DB_PATH
        results['config'] = CONFIG_DB_PATH
    except ImportError as e:
        results['config'] = f"ImportError: {e}"

    try:
        from elderlycare_v1_16.config.settings import DB_PATH as SETTINGS_DB_PATH
        results['settings'] = SETTINGS_DB_PATH
    except ImportError as e:
        results['settings'] = f"ImportError: {e}"

    try:
        from ml.household_analyzer import HouseholdAnalyzer
        # Inspect the default arg if possible or check if we can inspect the module
        import ml.household_analyzer as ha
        results['household_analyzer'] = ha.DB_PATH
    except ImportError as e:
        results['household_analyzer'] = f"ImportError: {e}"

    try:
        from intelligence.patterns.watchdog import DEFAULT_DB_PATH as WATCHDOG_DB_PATH
        results['watchdog'] = WATCHDOG_DB_PATH
    except ImportError as e:
        results['watchdog'] = f"ImportError: {e}"

    # Check consistency
    reference = results.get('config')
    print(f"Reference Path (config): {reference}")
    
    all_match = True
    for module, path in results.items():
        print(f"{module}: {path}")
        if path != reference:
            print(f"❌ MISMATCH: {module} != config")
            all_match = False
        else:
            print(f"✅ MATCH: {module}")

    if all_match:
        print("\nSUCCESS: All paths match!")
        sys.exit(0)
    else:
        print("\nFAILURE: Path mismatch detected.")
        sys.exit(1)

if __name__ == "__main__":
    verify_paths()
