# 🔄 Synchronization & Team Merging Guide

This guide explains how to synchronize your development environment across different locations (e.g., Home vs. Work) and how to merge training data when multiple team members are working on the same resident.

---

## 1. Environment Synchronization (`sync_data.sh`)

If you work on multiple machines, use `scripts/sync_data.sh` to keep your entire Beta 5 state (Database, Models, and Archives) in sync via a cloud service (like iCloud or Dropbox).

### How it Works
The script packages your essential data into a compressed archive on your Desktop. Once the cloud service syncs that file to your other machine, you can restore it there.

### Commands
```bash
# On Machine A (Saving your work)
./scripts/sync_data.sh backup

# On Machine B (Loading latest work)
./scripts/sync_data.sh restore
```

> [!IMPORTANT]
> Always run `backup` before leaving one machine and `restore` immediately upon arriving at the other to prevent data version conflicts.

---

## 2. Team Collaboration Workflows

When multiple team members are labeling and training data, use the following strategies:

### Scenario A: Different Residents
*Team Member A works on Elder X, Team Member B works on Elder Y.*

1. **Workflow**: Each member works independently.
2. **Merging**: Simply copy the specific resident folders:
   - `data/archive/[date]/[ResidentID]_*`
   - `backend/models/[ResidentID]/`
3. **Outcome**: No conflicts, as resident IDs are unique.

### Scenario B: Shared Resident (Split Dates)
*Team Member A labels Days 1-3, Team Member B labels Days 4-6 for the same Elder.*

To combine this work into a single "Super Model":
1. **Sync Files**: Collect all `_train.parquet` files from Team A and B into one machine's `data/archive` folder.
2. **Merge Script**: Use `backend/scripts/manual_combine_and_train.py`.
3. **Execution**:
   ```bash
   cd backend
   python3 scripts/manual_combine_and_train.py
   ```
4. **Internal Logic**: The script performs a `pd.concat`, sorts by timestamp, and **removes duplicates**. It then triggers a full retrospective training.

---

## 3. Step-by-Step Merging Example

**Goal**: Merge Team Member B's new training file into the Master machine.

1. **Extract**: Drop the new file into the Master machine: 
   `data/archive/2026-01-22/HK001_jessica_train_remote.parquet`
2. **Configure**: Open `backend/scripts/manual_combine_and_train.py` and update the `files` list:
   ```python
   files = [
       DATA_ROOT / "archive/2026-01-21/HK001_jessica_train_local.parquet",
       DATA_ROOT / "archive/2026-01-22/HK001_jessica_train_remote.parquet"
   ]
   ```
3. **Run**: Execute the script. 
4. **Verify**: Check the metrics output. The accuracy should reflect the combined knowledge of both labeling sessions.

---

## 🛠️ Troubleshooting
- **Missing Data**: If the database doesn't show merged segments, ensure you checked **"Train Retrospectively"** in the Correction Studio or ran the manual script above.
- **Port Conflicts**: If running on a new machine, ensure no other services are using port `3001` (Next.js) or `8502` (Streamlit).
