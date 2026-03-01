import sys
import os
import json
import logging
from pathlib import Path

# Setup paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BACKEND_DIR))

from elderlycare_v1_16.database import db
from elderlycare_v1_16.services.profile_service import ProfileService
from elderlycare_v1_16.config.settings import DATA_ROOT

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Migration")

def migrate_profiles():
    """
    Migrate 'enhanced_profile.json' and 'profile.json' to the SQLite DB.
    """
    # Initialize Service
    profile_service = ProfileService() 
    
    # Path to processed data where resident folders live
    processed_dir = DATA_ROOT / "processed"
    
    if not processed_dir.exists():
        logger.error(f"Processed dir not found: {processed_dir}")
        return

    # Iterate through all resident folders
    for elder_dir in processed_dir.iterdir():
        if not elder_dir.is_dir() or elder_dir.name.startswith('.'):
            continue
        
        elder_id = elder_dir.name
        logger.info(f"Migrating data for {elder_id}...")
        
        # 1. Look for Enhanced Profile (v1.11)
        enhanced_path = elder_dir / "profile.json" # In Beta_2 it was named profile.json but structure varied
        
        # Note: In Beta_2's ElderDataManager, "profiles/enhanced_profile.json" was the new standard
        # but "resident_db/profile.json" existed too.
        # Let's check `profiles` subdir first.
        profile_subdir = elder_dir / "profiles"
        v11_profile = profile_subdir / "enhanced_profile.json"
        
        profile_data = {}
        
        if v11_profile.exists():
            logger.info("  Found Enhanced Profile (v1.11)")
            with open(v11_profile, 'r') as f:
                profile_data = json.load(f)
        elif enhanced_path.exists():
            logger.info("  Found Standard Profile.json")
            with open(enhanced_path, 'r') as f:
                data = json.load(f)
                # Check if it has 'personal_info' key (Enhanced) or just flat keys (Legacy)
                if 'personal_info' in data:
                    profile_data = data
                else:
                    # Convert Legacy to Enhanced structure
                    profile_data = {
                        "personal_info": {
                            "full_name": data.get('name', elder_id),
                            "date_of_birth": None, # Legacy didn't have this usually
                            "gender": data.get('gender'),
                        },
                        "medical_history": {
                            "chronic_conditions": [], # Legacy "notes" might have this
                            "medications": [], 
                            "allergies": []
                        }
                    }
                    if 'notes' in data:
                         profile_data['personal_info']['notes'] = data['notes']
        
        if profile_data:
            try:
                profile_service.create_or_update_elder(elder_id, profile_data)
                logger.info(f"  SUCCESS: Profile migrated for {elder_id}")
            except Exception as e:
                logger.error(f"  FAILED: Could not migrate profile for {elder_id}: {e}")
        else:
            logger.warning(f"  No profile data found for {elder_id}")

    # Future: Migrate Sleep/ADL JSONs (Not strictly required for this step as we can re-process raw files or just start fresh)
    # But for "Self-contained", we should migrate if easily possible. 
    # Skipping heavy time-series migration for this specific step to keep it efficient. 

if __name__ == "__main__":
    # Ensure DB is init
    from scripts.init_db import init
    init()
    
    migrate_profiles()
    print("Migration complete.")
