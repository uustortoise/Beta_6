
import os
import sys
import argparse
import joblib
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from elderlycare_v1_16.config.settings import DATA_ROOT

MODELS_DIR = Path(os.path.join(os.path.dirname(__file__), '../models'))

def check_models(elder_id):
    elder_model_dir = MODELS_DIR / elder_id
    if not elder_model_dir.exists():
        print(f"No models found for {elder_id}")
        return

    print(f"Checking models for: {elder_id}")
    print("-" * 40)

    # Find label encoders
    encoders = list(elder_model_dir.glob("*_label_encoder.pkl"))
    if not encoders:
        print("No label encoders found.")
        return

    for enc_file in encoders:
        room_name = enc_file.name.replace("_label_encoder.pkl", "")
        try:
            encoder = joblib.load(enc_file)
            classes = encoder.classes_
            print(f"Room: {room_name}")
            print(f"  Trained Classes: {list(classes)}")
        except Exception as e:
            print(f"  Error loading {enc_file.name}: {e}")
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_models(sys.argv[1])
    else:
        # Default try finding one
        dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir()]
        if dirs:
             check_models(dirs[0].name)
        else:
             print("No elder models found.")
