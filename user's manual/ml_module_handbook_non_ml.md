# Beta 5.5 ML Handbook (Non-ML)

## 1. Audience
For operations, QA, product, and engineers who need system behavior clarity without deep model internals.

## 2. What Beta 5.5 Does
1. Ingests room sensor files.
2. Normalizes data to a stable timeline.
3. Trains or applies resident models.
4. Produces room activity timeline and ADL outputs.
5. Applies correction-driven relearning using golden samples.

## 3. End-to-End Process

### Input
- Training files: `*_train_*`
- Inference files: `*_input_*`
- Resident id parsed from filename.

### Processing
1. Watcher detects new file (`backend/run_daily_analysis.py`).
2. Loader parses room sheets and timestamps.
3. Data is cleaned/resampled to canonical 10s steps.
4. ML pipeline runs train+predict (train files) or predict-only (input files).
5. Predictions are written to `adl_history`.
6. Timeline segments are regenerated in `activity_segments`.
7. Downstream context (household, trajectory, alerts) is updated.

### Output
- Room timeline in UI/API
- ADL history for analytics and correction
- Derived care indicators (sleep, activity patterns, anomalies)

## 4. Label and Correction Model
Label authority chain:
1. Manual corrections (`is_corrected=1`)
2. Training file labels
3. Model predictions

Impact:
- corrected labels are protected from overwrite,
- retraining learns from corrections first,
- timeline quality improves over repeated correction cycles.

## 5. Event-First Evaluation (Promotion Path)
Runtime prediction and promotion evaluation are separate.

Promotion path uses:
- `validate_label_pack.py`
- `diff_label_pack.py`
- `run_event_first_smoke.py`
- `run_event_first_matrix.py`
- `event_first_go_no_go.yaml`

This provides deterministic PASS/FAIL decisions with explicit blocking reasons.

## 6. Current Reliability Status (Important)
- Bedroom/Kitchen/Bathroom/Entrance are generally strong.
- LivingRoom passive occupancy remains the hardest domain due to weak separability in low-motion periods.
- Release policy currently treats LivingRoom episode metrics as informational in Option Y, while keeping hard guards on core quality and MAE non-regression.

## 7. What Operators Should Monitor
Daily:
1. File ingestion success.
2. Prediction persistence to `adl_history`.
3. Timeline coherence in `activity_segments`.
4. Any smoke/matrix blocking reasons for new training packs.

Weekly:
1. Label drift and correction volume.
2. Timeline MAE trends (LivingRoom, Bedroom sleep).
3. Promotion decisions and gate outcomes.

## 8. Common Failure Classes
- Label pack invalid: schema, room, timestamp, or label violations.
- Smoke failure: correction evidence missing or day-level checks fail.
- Matrix failure: gate thresholds not met.
- Timeline regression: higher MAE despite similar F1/accuracy.

## 9. Adding New ADL Labels
Ops can add labels without core model code changes when done through registry/config:
1. Update `backend/config/adl_event_registry.v1.yaml`.
2. Ensure room validation alignment where required.
3. Validate pack and rerun smoke/matrix.

## 10. References
- Technical source-of-truth: `ml_adl_e2e_technical_flow.md`
- Ops manual: `operation_manual.md`
- Labeling guide: `labeling_guide.md`
- Golden sample process: `golden_sample_harvesting.md`
