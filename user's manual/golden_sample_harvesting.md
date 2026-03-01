# Beta 5.5 Golden Sample Harvesting Guide

## 1. Purpose
Collect high-quality corrected labels (golden samples) for model improvement and dataset curation.

Current program target:
- Golden sample collection cohort: `N=20` residents.

## 2. Preconditions
1. Runtime stack is healthy.
2. Corrections were saved in Streamlit and persisted to DB (`is_corrected=1`).
3. Label policy in `labeling_guide.md` has been followed.

## 3. Recommended Pre-Check
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5
./ops_doctor.sh running
```

## 4. Harvest Commands

### 4.1 Dry run (count only)
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
python3 scripts/harvest_gold_samples.py --dry-run
```

### 4.2 Full export
```bash
python3 scripts/harvest_gold_samples.py \
  --output /Users/dicksonng/DT/Development/Beta_5.5/data/golden_samples
```

### 4.3 Safe-only export
```bash
python3 scripts/harvest_gold_samples.py \
  --filter-safe-only \
  --output /Users/dicksonng/DT/Development/Beta_5.5/data/golden_samples/safe_only
```

### 4.4 Single resident export
```bash
python3 scripts/harvest_gold_samples.py \
  --elder-id HK0011_jessica \
  --output /Users/dicksonng/DT/Development/Beta_5.5/data/golden_samples/HK0011_jessica
```

## 5. Quality Criteria
A sample set is acceptable when:
1. timestamps are valid and ordered,
2. labels are known/registered,
3. correction provenance is retained,
4. no duplicated windows in exported output,
5. coverage includes target activities needed by downstream training.

## 6. Safe vs Caution Classes
- Safe-first classes for broad reuse: sleep, shower, clear bathroom/kitchen routines.
- Caution classes: ambiguous passive occupancy intervals (especially LivingRoom low-motion periods).
- If evidence is ambiguous, prefer `unknown` during correction rather than forcing room occupancy.

## 7. Integration Notes
- Golden samples override train-file/model labels in the priority chain.
- New corrections should be followed by smoke + matrix evaluation before promotion decisions.

## 8. Related Docs
- Label policy: `labeling_guide.md`
- E2E technical flow: `ml_adl_e2e_technical_flow.md`
- Ops SOP: `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/golden_sample_ops_sop_2026-02-25.md`
