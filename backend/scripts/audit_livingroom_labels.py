#!/usr/bin/env python3
"""
LivingRoom Label Audit Script
=============================

This script analyzes raw training data (Excel files) to identify "Phantom Gaps":
periods where the LivingRoom sensors strongly indicate human presence (motion, light, etc.)
but the manual labels claim the room (or the entire house) is unoccupied.

Usage:
  python audit_livingroom_labels.py --dir "/path/to/training_files"

Outputs a chronological report of highly suspicious 'unoccupied' windows.
"""

import os
import glob
import argparse
import warnings
import pandas as pd
import numpy as np

# Suppress pandas warnings for clean CLI output
warnings.filterwarnings('ignore')

ROOMS = ['Bedroom', 'LivingRoom', 'Kitchen', 'Bathroom', 'Entrance']

def load_and_preprocess_room(filepath, room_name):
    """Loads a specific room sheet and standardizes columns."""
    try:
        df = pd.read_excel(filepath, sheet_name=room_name)
    except Exception:
        return None # Sheet doesn't exist or is unreadable
    
    # Standardize column types
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['activity'] = df['activity'].astype(str).str.strip().str.lower()
    
    # Numeric conversions with fallback
    for s in ['motion', 'co2', 'light', 'sound', 'humidity', 'temperature']:
        if s in df.columns:
            df[s] = pd.to_numeric(df[s], errors='coerce')
            
    df['occ'] = (df['activity'] != 'unoccupied').astype(int)
    return df

def audit_file(filepath, min_duration_minutes=5):
    """Audits a single Excel file for LivingRoom contradictions."""
    filename = os.path.basename(filepath)
    room_dfs = {}
    
    for room in ROOMS:
        df = load_and_preprocess_room(filepath, room)
        if df is not None:
            room_dfs[room] = df
            
    if 'LivingRoom' not in room_dfs:
        print(f"  [SKIPPED] {filename}: No LivingRoom sheet found.")
        return 0, 0
        
    lr = room_dfs['LivingRoom']
    
    # 1. Flag suspicious 'unoccupied' windows in LivingRoom
    lr_unocc = lr[lr['activity'] == 'unoccupied'].copy()
    
    # Define empirical presence triggers based on Dec audits
    lr_unocc['has_motion'] = (lr_unocc['motion'] > 0.5).astype(int)
    lr_unocc['has_light'] = (lr_unocc['light'] > 500).astype(int)
    lr_unocc['has_sound'] = (lr_unocc['sound'] > 4.2).astype(int)
    lr_unocc['has_co2'] = (lr_unocc['co2'] > 3100).astype(int)
    
    # Require at least 2 strong triggers to flag a window
    lr_unocc['score'] = (lr_unocc['has_motion'] + 
                         lr_unocc['has_light'] + 
                         lr_unocc['has_sound'] + 
                         lr_unocc['has_co2'])
    
    suspicious_lr = lr_unocc[lr_unocc['score'] >= 2]
    total_suspicious_windows = len(suspicious_lr)
    
    if total_suspicious_windows == 0:
        print(f"  [CLEAN] {filename}: No severe label contradictions found.")
        return 0, 0
        
    # 2. Cross-reference other rooms to ensure the person wasn't elsewhere
    merged = lr[['timestamp', 'activity', 'occ', 'motion', 'light', 'co2', 'sound']].copy()
    merged.columns = ['timestamp', 'LR_act', 'LR_occ', 'LR_motion', 'LR_light', 'LR_co2', 'LR_sound']
    
    for room in ['Bedroom', 'Kitchen', 'Bathroom', 'Entrance']:
        if room in room_dfs:
            rdf = room_dfs[room][['timestamp', 'activity', 'occ']].copy()
            rdf.columns = ['timestamp', f'{room[:3]}_act', f'{room[:3]}_occ']
            merged = pd.merge(merged, rdf, on='timestamp', how='left')
    
    susp_ts = set(suspicious_lr['timestamp'].values)
    susp_merged = merged[merged['timestamp'].isin(susp_ts)].copy()
    
    # Check if ANY other room was labeled as occupied at the exact same timestamp
    other_occ_cols = [c for c in merged.columns if c.endswith('_occ') and c != 'LR_occ']
    susp_merged['any_other_occ'] = susp_merged[other_occ_cols].sum(axis=1) > 0
    
    # "Phantom Gaps": LR sensors scream occupied AND labeler says nowhere in the house is occupied
    phantom = susp_merged[~susp_merged['any_other_occ']]
    
    if len(phantom) == 0:
        print(f"  [MINOR] {filename}: LR suspicious, but occupant located in another room (sensor leakage/transition).")
        return total_suspicious_windows, 0
        
    # 3. Group phantom windows into continuous episodes
    phantom = phantom.sort_values('timestamp')
    ts_arr = phantom['timestamp'].values
    
    eps = []
    ep_start = 0
    for i in range(1, len(ts_arr)):
        # If gap between flagged windows is > 2 minutes, break the episode
        gap = (ts_arr[i] - ts_arr[i-1]) / np.timedelta64(1, 'm')
        if gap > 2:
            eps.append((ep_start, i))
            ep_start = i
    eps.append((ep_start, len(ts_arr)))
    
    print(f"\\n{'='*80}")
    print(f"AUDIT ALERTS: {filename}")
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  - Total Suspicious Windows (LR): {total_suspicious_windows}")
    print(f"  - Phantom Windows (All rooms unocc): {len(phantom)}")
    print(f"\\nHighly probable missed occupancy periods (Duration >= {min_duration_minutes} min):")
    
    episodes_reported = 0
    total_missed_minutes = 0
    
    for s, e in eps:
        ep = phantom.iloc[s:e]
        dur_min = (len(ep) * 10) / 60  # Assuming 10-second windows
        
        if dur_min >= min_duration_minutes:
            episodes_reported += 1
            total_missed_minutes += dur_min
            t0 = str(ep['timestamp'].iloc[0])[11:19]
            t1 = str(ep['timestamp'].iloc[-1])[11:19]
            
            mot_mean = ep['LR_motion'].mean()
            mot_max = ep['LR_motion'].max()
            lgt_mean = ep['LR_light'].mean()
            co2_mean = ep['LR_co2'].mean()
            snd_mean = ep['LR_sound'].mean()
            
            # Highlight severe evidence
            alert_flags = []
            if mot_mean > 1.0 or mot_max > 50: alert_flags.append('MOTION')
            if lgt_mean > 800: alert_flags.append('LIGHTS_ON')
            if co2_mean > 3200: alert_flags.append('HIGH_CO2')
            flag_str = "+".join(alert_flags) if alert_flags else "AMBIENT_ELEVATED"
            
            print(f"  [X] {t0} → {t1} ({dur_min:4.1f} min)")
            print(f"      Evidence: {flag_str}")
            print(f"      Metrics:  Motion_Avg={mot_mean:.1f} (Max={mot_max:.0f}), Light={lgt_mean:.0f}, CO2={co2_mean:.0f}")
            print(f"      Action:   Change ALL windows here to `livingroom_normal_use`\\n")
            
    if episodes_reported == 0:
        print(f"  No continuous episodes >= {min_duration_minutes} minutes found (mostly isolated noisy windows).")
        
    return total_suspicious_windows, total_missed_minutes

def main(data_dir, min_duration):
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return
        
    files = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))
    if not files:
        print(f"No Excel files found in '{data_dir}'.")
        return
        
    print(f"Starting LivingRoom Label Audit on {len(files)} files...")
    print(f"Directory: {data_dir}\\n")
    
    total_files_audited = 0
    total_phantom_mins_overall = 0
    
    for fpath in files:
        _, missed_mins = audit_file(fpath, min_duration)
        total_phantom_mins_overall += missed_mins
        total_files_audited += 1
        
    print(f"{'='*80}")
    print("AUDIT COMPLETE")
    print(f"Total files audited: {total_files_audited}")
    print(f"Total high-confidence missed occupancy: ~{total_phantom_mins_overall/60:.1f} hours")
    print(f"{'='*80}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audit LivingRoom manual labels against raw sensor data.")
    parser.add_argument("--dir", type=str, required=True, help="Path to directory containing training .xlsx files.")
    parser.add_argument("--min-duration", type=int, default=5, help="Minimum episode duration in minutes to flag (default: 5).")
    args = parser.parse_args()
    
    main(args.dir, args.min_duration)
