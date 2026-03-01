import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Room Occupancy Label Auditor", page_icon="🕵️", layout="wide")

ROOMS = ['Bedroom', 'LivingRoom', 'Kitchen', 'Bathroom', 'Entrance']

def load_and_preprocess_room(uploaded_file, room_name):
    """Loads a specific room sheet from an uploaded Excel file."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=room_name)
    except Exception:
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['activity'] = df['activity'].astype(str).str.strip().str.lower()
    
    for s in ['motion', 'co2', 'light', 'sound', 'humidity', 'temperature']:
        if s in df.columns:
            df[s] = pd.to_numeric(df[s], errors='coerce')
            
    df['occ'] = (df['activity'] != 'unoccupied').astype(int)
    return df

def get_room_suspicion_score(df, room):
    """Calculates a suspicion score (0-4) based on sensors, tuned per room."""
    has_motion = (df['motion'] > 0.5).astype(int)
    
    # NEW: Detect missing sleep blocks
    df['hour'] = df['timestamp'].dt.hour
    is_night = (df['hour'] >= 20) | (df['hour'] <= 9)
    # Sleep signature: night, pitch black, high CO2, occasional subtle motion
    is_sleeping = (is_night & (df['light'] <= 10) & (df['co2'] > 1200)).astype(int)
    
    if room == 'Bedroom':
        has_light = (df['light'] > 50).astype(int) # Lower threshold for bedside lamps
        has_sound = (df['sound'] > 4.0).astype(int)
        has_co2_or_hum = ((df['co2'] > 2800) | (df['humidity'] > 35)).astype(int)
        
        # If the sleep signature is strong, override and give it a high score
        base_score = has_motion + has_light + has_sound + has_co2_or_hum
        # Max of 4 to keep it within the 0-4 range
        return np.maximum(base_score, is_sleeping * 4)
        
    elif room in ['Kitchen', 'Bathroom']:
        has_light = (df['light'] > 200).astype(int)
        has_sound = (df['sound'] > 4.2).astype(int)
        has_humidity = (df['humidity'] > 40).astype(int) # Key for cooking/showering
        return has_motion + has_light + has_sound + has_humidity
        
    else: # LivingRoom, Entrance
        has_light = (df['light'] > 500).astype(int)
        has_sound = (df['sound'] > 4.2).astype(int)
        has_co2 = (df['co2'] > 3100).astype(int)
        
        # Sleep can happen on living room couch too
        base_score = has_motion + has_light + has_sound + has_co2
        return np.maximum(base_score, is_sleeping * 4)


def audit_file(uploaded_file, min_duration_minutes=5):
    """Analyzes all rooms in a file and returns a list of contradiction dicts."""
    filename = uploaded_file.name
    room_dfs = {}
    
    # Reset file pointer and load all available rooms
    uploaded_file.seek(0)
    for room in ROOMS:
        df = load_and_preprocess_room(uploaded_file, room)
        uploaded_file.seek(0)
        if df is not None:
            room_dfs[room] = df
            
    all_episodes = []
    
    # Build a master dataframe of occupancy across all rooms for cross-checking
    if not room_dfs:
        return []
        
    # Start with timestamps from any available room
    master_ts = list(room_dfs.values())[0]['timestamp'].copy()
    master_df = pd.DataFrame({'timestamp': master_ts})
    
    for r, rdf in room_dfs.items():
        occ_df = rdf[['timestamp', 'occ']].copy()
        occ_df.columns = ['timestamp', f'{r}_occ']
        master_df = pd.merge(master_df, occ_df, on='timestamp', how='left')
    
    # Audit each room individually against the "rest of house"
    for target_room, df_tgt in room_dfs.items():
        # Calculate specialized suspicion score on the full room dataframe first
        # This is critical so we don't lose the score column during merges later
        has_motion = (df_tgt['motion'] > 0.5).astype(int)
        df_tgt['hour'] = df_tgt['timestamp'].dt.hour
        is_night = (df_tgt['hour'] >= 20) | (df_tgt['hour'] <= 9)
        is_sleeping = (is_night & (df_tgt['light'] <= 10) & (df_tgt['co2'] > 1200)).astype(int)
        
        has_light = (df_tgt['light'] > (50 if target_room == 'Bedroom' else 200)).astype(int)
        has_sound = (df_tgt['sound'] > 4.0).astype(int)
        
        if target_room == 'Bedroom':
            has_co2_or_hum = ((df_tgt['co2'] > 2800) | (df_tgt['humidity'] > 35)).astype(int)
            base_score = has_motion + has_light + has_sound + has_co2_or_hum
            df_tgt['score'] = np.maximum(base_score, is_sleeping * 4)
        elif target_room in ['Kitchen', 'Bathroom']:
            has_humidity = (df_tgt['humidity'] > 40).astype(int)
            base_score = has_motion + has_light + has_sound + has_humidity
            df_tgt['score'] = base_score
        else: # LivingRoom, Entrance
            has_co2 = (df_tgt['co2'] > 3100).astype(int)
            base_score = has_motion + has_light + has_sound + has_co2
            df_tgt['score'] = np.maximum(base_score, is_sleeping * 4)
            
        # Then filter down to windows we want to audit
        if target_room == 'Bedroom':
            # Bridge over tiny sprinkles of 'sleep' labels to find massive unlabeled blocks
            df_unocc = df_tgt[df_tgt['activity'] != 'bedroom_normal_use'].copy()
        else:
            df_unocc = df_tgt[df_tgt['activity'] == 'unoccupied'].copy()
            
        suspicious = df_unocc[df_unocc['score'] >= 2]
        
        if len(suspicious) == 0:
            continue
            
        merged = df_tgt[['timestamp', 'activity', 'occ', 'motion', 'light', 'co2', 'sound', 'humidity', 'score']].copy()
        merged = pd.merge(merged, master_df, on='timestamp', how='left')
        
        susp_ts = set(suspicious['timestamp'].values)
        susp_merged = merged[merged['timestamp'].isin(susp_ts)].copy()
        
        # Check if ANY OTHER room was labeled as occupied at the same time
        other_occ_cols = [c for c in master_df.columns if c != 'timestamp' and c != f'{target_room}_occ']
        if other_occ_cols:
            susp_merged['any_other_occ'] = (susp_merged[other_occ_cols].sum(axis=1) > 0).astype(bool)
        else:
            susp_merged['any_other_occ'] = False
            
        # Build episodes from ALL suspicious windows, including those with temporary cross-room overlaps
        # This keeps brief bathroom trips from fracturing a 6-hour sleep block
        susp_merged = susp_merged.sort_values('timestamp')
        ts_arr = susp_merged['timestamp'].values
        
        eps = []
        ep_start = 0
        for i in range(1, len(ts_arr)):
            gap = (ts_arr[i] - ts_arr[i-1]) / np.timedelta64(1, 'm')
            
            # Gap tolerance depends on whether it looks like sleep
            hour = ts_arr[i].astype('datetime64[h]').astype(int) % 24
            is_night = (hour >= 20) or (hour <= 9)
            gap_tolerance = 20 if (is_night and target_room == 'Bedroom') else 2  # Very forgiving at night in bedroom
            
            if gap > gap_tolerance:
                eps.append((ep_start, i))
                ep_start = i
        eps.append((ep_start, len(ts_arr)))
        
        for s, e in eps:
            ep = susp_merged.iloc[s:e]
            dur_min = (len(ep) * 10) / 60
            
            # If the episode is mostly occupied by another room, it's not a phantom gap here
            other_occ_ratio = ep['any_other_occ'].astype(int).mean()
            
            if other_occ_ratio > 0.15:  # Tolerance for brief overlap (e.g. 5 min bathroom in 6hr sleep)
                continue
            
            if dur_min >= min_duration_minutes:
                t0 = ep['timestamp'].iloc[0]
                t1 = ep['timestamp'].iloc[-1]
                mot_mean = ep['motion'].mean()
                mot_max = ep['motion'].max()
                lgt_mean = ep['light'].mean()
                snd_mean = ep['sound'].mean()
                co2_mean = ep['co2'].mean()
                hum_mean = ep['humidity'].mean()
                
                # Check current labels
                unocc_ratio = (ep['activity'] == 'unoccupied').mean()
                sleep_ratio = (ep['activity'] == 'sleep').mean()
                
                # If it's already mostly labeled as sleep, it's not a missing label block
                if target_room == 'Bedroom' and sleep_ratio > 0.8:
                    continue
                    
                curr_label = 'Unoccupied'
                if target_room == 'Bedroom' and sleep_ratio > 0:
                    curr_label = f"Unoccupied ({unocc_ratio*100:.0f}%) / Sleep ({sleep_ratio*100:.0f}%)"
                
                flags = []
                # Check for sleep signature first
                hour = getattr(t0, 'hour', t0.components.hours if hasattr(t0, 'components') else 0)
                is_night = (hour >= 20) or (hour <= 9)
                if is_night and lgt_mean <= 15 and co2_mean > 1000 and mot_mean < 25:
                    flags.append('SLEEP_SIGNATURE')
                else:
                    if mot_mean > 1.0 or mot_max > 50: flags.append('MOTION')
                    if lgt_mean > (50 if target_room == 'Bedroom' else 800): flags.append('LIGHTS_ON')
                    if snd_mean > 4.5: flags.append('HIGH_SOUND')
                    if target_room in ['Bathroom', 'Kitchen'] and hum_mean > 45: flags.append('HIGH_HUMIDITY')
                    if target_room in ['LivingRoom', 'Bedroom'] and co2_mean > 3200: flags.append('HIGH_CO2')
                
                # Determine recommended label
                if 'SLEEP_SIGNATURE' in flags or (target_room == 'Bedroom' and lgt_mean < 50 and mot_mean < 25):
                    rec_label = 'sleep'
                elif target_room == 'Bedroom':
                    rec_label = 'bedroom_normal_use'
                else:
                    rec_label = f'{target_room.lower()}_normal_use'
                
                all_episodes.append({
                    'File': filename,
                    'Room': target_room,
                    'Start Time': str(t0),
                    'End Time': str(t1),
                    'Duration (min)': round(dur_min, 1),
                    'Flags': "+".join(flags) if flags else "AMBIENT_ELEVATED",
                    'Motion Avg': round(mot_mean, 2),
                    'Motion Max': round(mot_max, 2),
                    'Light Avg': round(lgt_mean, 0),
                    'CO2 Avg': round(co2_mean, 0),
                    'Hum Avg': round(hum_mean, 1),
                    'Current Label': curr_label,
                    'Recommended Label': rec_label
                })
                
    return all_episodes

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- UI Setup ---
st.title("🕵️ Cross-Room Label Auditor")
st.markdown("Upload raw training data (`.xlsx`) to detect **Phantom Gaps** across **ALL rooms** (Bedroom, LivingRoom, Kitchen, Bathroom). These are periods where sensors strongly detect occupancy, but the entire house is mislabeled as `unoccupied`.")

with st.sidebar:
    st.header("Settings")
    min_dur = st.number_input("Minimum Gap Duration (minutes)", min_value=1, value=5, help="Ignore mislabeled gaps shorter than this duration.")
    
    st.markdown("---")
    st.markdown("**Room Detection Thresholds:**")
    st.caption("- **LivingRoom**: High Light, CO2, Motion")
    st.caption("- **Bedroom**: Low Light + Respiration, OR 🌙 Nighttime Sleep Detection (CO2 > 1200 + Pitch Black)")
    st.caption("- **Kitchen/Bath**: High Humidity, Sound, Motion")

uploaded_files = st.file_uploader("Upload Training Excel Files", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Run House-Wide Audit", type="primary"):
        all_episodes = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Auditing file {i+1} of {len(uploaded_files)}: {file.name} (Checking all 5 rooms)...")
            eps = audit_file(file, min_duration_minutes=min_dur)
            all_episodes.extend(eps)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("Audit Complete!")
        
        if not all_episodes:
            st.success("🎉 No significant label contradictions found across any rooms in these files!")
        else:
            # Display results
            results_df = pd.DataFrame(all_episodes)
            results_df = results_df.sort_values(by="Duration (min)", ascending=False).reset_index(drop=True)
            
            # Reorder columns for readability
            cols = ['File', 'Room', 'Start Time', 'End Time', 'Duration (min)', 'Flags', 'Current Label', 'Recommended Label', 'Motion Max', 'Light Avg', 'CO2 Avg', 'Hum Avg']
            results_df = results_df[cols]
            
            total_missed = results_df['Duration (min)'].sum()
            
            # Summary metrics
            st.error(f"⚠️ Found {len(results_df)} mislabeled episodes across {len(uploaded_files)} files.")
            
            col1, col2 = st.columns(2)
            col1.metric(label="Total Missing Data", value=f"{total_missed / 60:.1f} Hours")
            
            # Break down by room
            room_counts = results_df['Room'].value_counts()
            breakdown_str = " | ".join([f"{r}: {c}" for r, c in room_counts.items()])
            col2.metric(label="Errors by Room", value=str(len(room_counts)) + " rooms", delta=breakdown_str, delta_color="off")
            
            st.subheader("🔥 Top 10 Most Severe Contradictions")
            st.dataframe(results_df.head(10), use_container_width=True)
            
            # Download full CSV
            st.markdown("---")
            st.subheader("📥 Download Complete Report")
            csv_data = convert_df_to_csv(results_df)
            
            st.download_button(
                label="Download Full Audit Report (CSV)",
                data=csv_data,
                file_name="cross_room_label_audit_report.csv",
                mime="text/csv",
                type="primary"
            )
            
            with st.expander("View Full House-Wide Data Table"):
                st.dataframe(results_df, use_container_width=True)
