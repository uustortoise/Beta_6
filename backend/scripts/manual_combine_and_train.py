
import sys
import os
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.pipeline import UnifiedPipeline
from utils.data_loader import load_sensor_data, save_to_parquet
from elderlycare_v1_16.config.settings import DATA_ROOT

logger = UnifiedPipeline().logger

def main():
    elder_id = "HK001_jessica"
    
    files = [
        DATA_ROOT / "archive/2026-01-14/HK001_jessica_train_17dec2025.parquet",
        DATA_ROOT / "archive/2026-01-14/HK001_jessica_input_1jan2026.parquet",
        DATA_ROOT / "archive/2026-01-15/HK001_jessica_input_2jan2026.parquet"
    ]
    
    print(f"Loading {len(files)} files...")
    
    combined_data = {}
    
    for f in files:
        if not f.exists():
            print(f"ERROR: File not found: {f}")
            return
            
        print(f"Loading {f.name}...")
        data = load_sensor_data(f)
        for room, df in data.items():
            if room not in combined_data:
                combined_data[room] = []
            combined_data[room].append(df)
            
    # Combine
    final_data = {}
    for room, dfs in combined_data.items():
        print(f"Merging {len(dfs)} dataframes for {room}...")
        merged = pd.concat(dfs, ignore_index=True)
        # Ensure timestamp sort
        if 'timestamp' in merged.columns:
            merged['timestamp'] = pd.to_datetime(merged['timestamp'])
            merged = merged.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        final_data[room] = merged
        
    # Save temp file
    temp_file = DATA_ROOT / "raw/temp_combined_retrain.parquet"
    print(f"Saving combined data to {temp_file}...")
    save_to_parquet(final_data, temp_file)
    
    # Train
    print("Starting Training Pipeline (this applies Class Weights + Golden Samples)...")
    pipeline = UnifiedPipeline()
    results, metrics = pipeline.train_and_predict(temp_file, elder_id)
    
    print("\nTraining Complete!")
    print("Metrics:")
    for m in metrics:
        print(f"  Room: {m['room']}, Acc: {m['accuracy']:.2%}, Epochs: {m['epochs']}")
        
    # Cleanup
    if temp_file.exists():
        temp_file.unlink()
        print("Temp file cleaned up.")

if __name__ == "__main__":
    main()
