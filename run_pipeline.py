import sys
import os
from pathlib import Path

# Add backend to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend'))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from ml.pipeline import UnifiedPipeline
from elderlycare_v1_16.config.logging_config import setup_logging

# Setup logging
setup_logging("backend/logs/manual_run.log")

def main():
    pipeline = UnifiedPipeline(enable_denoising=True)
    
    # Path to the training file we copied
    train_file = "data/raw/HK001_jessica_train_17dec2025.xlsx"
    
    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        return

    print("Starting training...")
    predictions, trained_rooms = pipeline.train_and_predict(train_file, "HK001")
    
    print(f"Training completed for rooms: {[r['room'] for r in trained_rooms]}")
    
    # Now run prediction on the input file
    input_file = "data/raw/HK001_jessica_input_18dec 2025.xlsx"
    if os.path.exists(input_file):
        print(f"Running prediction on {input_file}...")
        results = pipeline.predict(input_file, "HK001")
        print(f"Prediction completed. Results for rooms: {list(results.keys())}")
        
        # Save results (optional, for verification)
        for room, df in results.items():
            out_path = f"data/processed/prediction_{room}_HK001.csv"
            df.to_csv(out_path)
            print(f"Saved prediction for {room} to {out_path}")
    else:
        print(f"Input file {input_file} not found.")

if __name__ == "__main__":
    main()
