# Walkthrough - Training File Correction Feature

This walkthrough documents the implementation of the feature allowing users to correct ground-truth labels in training files directly within the Correction Studio.

## Problem Statement
Previously, the Correction Studio was hardcoded to only load "Input Files" (predictions). Users could not correct mislabeled ground-truth data in "Training Files" (`_train.xlsx`), even though the backend was capable of merging these corrections as Golden Samples for future retraining.

## Solution Overview
We modified `export_dashboard.py` to:
1.  **Enable File Type Selection**: Added a radio button to toggle between Predictions and Training Files.
2.  **Add Safety Warnings**: Displays a prominent warning when editing training files to inform users that changes will override ground truth.
3.  **Implement Label Fallback**: Added logic to pre-populate labels from the Excel file itself if no model predictions are found in the database.
4.  **Fix structural glitches**: Cleaned up deeply nested indentation issues and redundant logic blocks in the Correction Studio tab.

## Changes Made

### UI: File Type Toggle
Users can now switch context between correcting day-to-day predictions and improving historical ground-truth data.

```python
file_type_filter = st.radio(
    "File Type", 
    ["📥 Input Files (Predictions)", "📚 Training Files (Ground Truth)"],
    horizontal=True
)
```

### UI: Warning Banner
Editing training data is a high-impact action. A clear warning is shown to ensure user intent.

```python
if target_type == 'train':
    st.warning("⚠️ **Correcting Ground Truth**: Changes here will override original training labels on next retrain.")
```

### Data: Label Fallback Logic
Training files often haven't been processed into the `adl_history` database table yet. The system now gracefully falls back to using the labels already present in the Excel file as a starting point for correction.

```python
elif 'activity' in df.columns:
    df['activity'] = df['activity'].fillna('inactive')
    st.success("📚 Pre-populated labels from the original file.")
```

## Verification Results

### Code Review
- **`ml/pipeline.py`**: Confirmed `fetch_all_golden_samples` is called during training and correctly merges overrides into the training DataFrame.
- **`ml/utils.py`**: Confirmed `fetch_all_golden_samples` specifically targets rows where `is_corrected = 1`.
- **`export_dashboard.py`**: Verified that `is_corrected = 1` is set during Batch Apply, ensuring corrections are picked up during the next training run.

### Structural Integrity (Senior Engineer Review)
- Resolved all "Indentation Hell" issues in `export_dashboard.py`.
- Removed redundant `else` blocks that were causing Streamlit parsing errors.
- **Fixed logic bug**: Sensor widgets (multiselect, checkbox, charts) were outside their `if sensor_cols:` guard block. This would have caused crashes when loading files without sensor columns. Now properly nested.
- Standardized 4-space indentation across the entire Correction Studio block.

- Standardized 4-space indentation across the entire Correction Studio block.

### Critical Data Integrity Fixes (Post-Verification)
During end-to-end verification, we identified and fixed three critical issues:

1.  **Missing Elder (Foreign Key Violation)**: 
    - **Issue**: Correcting a training file for a new elder failed because the elder profile didn't exist in the database.
    - **Fix**: Added elder auto-registration (`INSERT ... ON CONFLICT DO NOTHING`) to `adl_service.py`, `prediction.py`, and `export_dashboard.py`.
    
2.  **SQLite Incompatibility**:
    - **Issue**: `INSERT OR IGNORE` (SQLite syntax) silently failed in PostgreSQL mode.
    - **Fix**: Replaced with PostgreSQL-compatible `INSERT ... ON CONFLICT DO NOTHING`.
    
3.  **Legacy Adapter Gap**:
    - **Issue**: `INSERT OR REPLACE` (SQLite upsert) was not translated by the adapter, causing silent failures for config/history updates.
    - **Fix**: Added regex-based translation to `INSERT ... ON CONFLICT DO UPDATE SET ...` in `legacy_adapter.py`.

## Conclusion
The Correction Studio is now a fully capable tool for both correcting model drift (predictions) and curating high-quality ground-truth datasets (training files). All changes are integrated with the existing Beta 5.5 Golden Sample and Retraining pipelines.
