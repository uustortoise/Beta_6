import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from data_manager import DataManager

def main():
    parser = argparse.ArgumentParser(description="Export Resident Data from SQLite to Excel")
    parser.add_argument("--resident", required=True, help="Resident ID (e.g., resident_99)")
    parser.add_argument("--start", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--output", help="Specific output filename")
    
    args = parser.parse_args()
    
    # Initialize Data Manager
    data_root = Path(current_dir) / "../data"
    data_mgr = DataManager(data_root)
    
    # Run Export
    print(f"Starting export for {args.resident}...")
    success = data_mgr.export_data_to_excel(
        resident_id=args.resident,
        start_date=args.start,
        end_date=args.end,
        output_file=Path(args.output) if args.output else None
    )
    
    if success:
        print("✅ Export completed successfully.")
    else:
        print("❌ Export failed or no data found.")

if __name__ == "__main__":
    main()
