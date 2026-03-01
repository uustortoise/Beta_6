# Beta 5.5 Imbalance Upgrade Run Log

- Date: 2026-02-19
- Workspace: `/Users/dicksonng/DT/Development/Beta_5.5`

## 1. Safety and Backup

- Backup archive created:
  - `/tmp/beta55_backups/Beta_5.5_20260219_224428.tar.gz`
  - SHA-256: `9f0e4e551e669896cbbf6c8c361177d4d4df820bd900bb55a14dfdac94eb12ae`
  - Size: `1.6G`

## 2. Implemented Changes

### Phase 1: coverage/support diagnostics

- Added data continuity + canonical-file audit payloads in backtest output:
  - `data_continuity_audit`
  - `splits[].train_day_continuity`
- Files changed:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`

### Phase 3: segment mode structural upgrade

- Upgraded segment feature extraction from minimal stats to enriched feature table:
  - occupancy shape stats
  - sensor stats (motion/light/co2/temperature/humidity/sound/vibration)
  - time encodings
  - activity-probability summaries
- Added learned segment classifier path (default-off) with deterministic fallback:
  - confidence gate
  - minimum support windows gate
  - fallback reason instrumentation
- Wired new segment classifier flags and debug outputs into backtest script.
- Files changed:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_features.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/ml/segment_classifier.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_features.py`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_segment_classifier.py`

## 3. Commands Executed and Results

1. Segment module tests
```bash
pytest -q backend/tests/test_segment_features.py backend/tests/test_segment_classifier.py backend/tests/test_segment_proposal.py backend/tests/test_segment_projection.py
```
- Result: `9 passed`

2. Backtest/support contract suites
```bash
pytest -q backend/tests/test_event_first_backtest_script.py backend/tests/test_event_first_backtest_aggregate.py backend/tests/test_d2_strict_splitseed_integration.py
```
- Result: `92 passed`

3. Full targeted suite
```bash
pytest -q backend/tests/test_segment_proposal.py backend/tests/test_segment_features.py backend/tests/test_segment_classifier.py backend/tests/test_segment_projection.py backend/tests/test_event_models.py backend/tests/test_timeline_gates.py backend/tests/test_event_first_backtest_script.py backend/tests/test_event_first_backtest_aggregate.py backend/tests/test_d2_strict_splitseed_integration.py
```
- Result: `130 passed, 1 warning`

4. Runtime smoke (default)
```bash
python3 backend/scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 6 --seed 11 --output "/tmp/ws6_beta55_upgrade_smoke_default_seed11_after_segment_patch.json"
```
- Result: output generated successfully with continuity audit payload.

5. Runtime smoke (segment learned classifier enabled)
```bash
python3 backend/scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 6 --seed 11 --enable-bedroom-livingroom-segment-mode --enable-bedroom-livingroom-segment-learned-classifier --segment-classifier-min-segments 2 --segment-classifier-confidence-floor 0.0 --segment-classifier-min-windows 1 --output "/tmp/ws6_beta55_upgrade_smoke_segment_learned_seed11.json"
```
- Result: output generated successfully with segment classifier debug fields.

## 4. Remaining for Full WS-6 Signoff

1. Execute full 3-seed matrix on intended day window (`min-day 4`, `max-day 8`) with selected candidate flags.
2. Run aggregate signoff and verify `eligible hard-gate = 30/30`.
3. Record final go/no-go evidence bundle paths in this log.

## 5. Strict Candidate Results (2026-02-19)

### Candidate A: segment mode + learned classifier (min_train_days=0)

- Output:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentlearned_20260219/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentlearned_20260219/ws6_signoff.json`
- Gate summary:
  - `hard_gate_checks_passed_eligible = 38`
  - `hard_gate_checks_total_eligible = 60`
  - `split_pass_rate = 0.6333`
- Room pass counts (full):
  - Bathroom `12/12`
  - Bedroom `2/12`
  - Entrance `12/12`
  - Kitchen `12/12`
  - LivingRoom `0/12`

### Candidate B: segment mode + learned classifier (min_train_days=3, anchor-comparable eligibility)

- Output:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentlearned_mintrain3_20260219/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentlearned_mintrain3_20260219/ws6_signoff.json`
- Gate summary:
  - `hard_gate_checks_passed_eligible = 20`
  - `hard_gate_checks_total_eligible = 30`
  - `hard_gate_checks_passed_full = 38`
  - `hard_gate_checks_total_full = 60`
- Room pass counts (eligible):
  - Bathroom `6/6`
  - Bedroom `2/6`
  - Entrance `6/6`
  - Kitchen `6/6`
  - LivingRoom `0/6`
- Top fail reasons:
  - `occupied_recall_lt_0.500`
  - `occupied_f1_lt_0.580`
  - `occupied_f1_lt_0.550`
  - `fragmentation_score_lt_0.450`

### Candidate C: segment mode + heuristic classifier (learned disabled, min_train_days=3)

- Output:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentheur_mintrain3_20260219/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentheur_mintrain3_20260219/ws6_signoff.json`
- Gate summary:
  - `hard_gate_checks_passed_eligible = 20`
  - `hard_gate_checks_total_eligible = 30`
  - `hard_gate_checks_passed_full = 38`
  - `hard_gate_checks_total_full = 60`
- Observation:
  - Same eligible/full result as Candidate B in this run family; learned classifier did not change hard-gate outcome yet.

### Candidate D: segment mode + Stage-A HGB (heuristic segment classifier, min_train_days=3)

- Output:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentheur_hgb_mintrain3_20260219/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentheur_hgb_mintrain3_20260219/ws6_signoff.json`
- Gate summary:
  - `hard_gate_checks_passed_eligible = 21`
  - `hard_gate_checks_total_eligible = 30`
  - `hard_gate_checks_passed_full = 39`
  - `hard_gate_checks_total_full = 60`
- Room pass counts (eligible):
  - Bathroom `6/6`
  - Bedroom `3/6`
  - Entrance `6/6`
  - Kitchen `6/6`
  - LivingRoom `0/6`
- Observation:
  - Slight uplift vs Candidates B/C (`20 -> 21` eligible passes), still below anchor (`23/30`) and far from target (`30/30`).

### Candidate E: segment mode + Stage-A HGB + learned segment classifier (min_train_days=3)

- Output:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentlearned_hgb_mintrain3_20260219/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_segmentlearned_hgb_mintrain3_20260219/ws6_signoff.json`
- Gate summary:
  - `hard_gate_checks_passed_eligible = 21`
  - `hard_gate_checks_total_eligible = 30`
  - `hard_gate_checks_passed_full = 39`
  - `hard_gate_checks_total_full = 60`
- Observation:
  - Same result as Candidate D; learned segment classifier still not moving hard-gate outcomes in strict WS-6.

### Candidate F: Phase-2 imbalance controls + HGB (no segment mode, min_train_days=3)

- Output:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_phase2_hgb_hn_replay_mintrain3_20260219/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_phase2_hgb_hn_replay_mintrain3_20260219/ws6_signoff.json`
- Enabled controls:
  - BL hard-negative mining
  - failure replay
  - livingroom occupied sample weight
  - BL Stage-A HGB
- Gate summary:
  - `hard_gate_checks_passed_eligible = 19`
  - `hard_gate_checks_total_eligible = 30`
  - `hard_gate_checks_passed_full = 37`
  - `hard_gate_checks_total_full = 60`
- Room pass counts (eligible):
  - Bathroom `6/6`
  - Bedroom `1/6`
  - Entrance `6/6`
  - Kitchen `6/6`
  - LivingRoom `0/6`
- Observation:
  - Worse than Candidates D/E and below anchor.

### Candidate G: Phase-2 + HGB + sequence Stage-A + hardgate tuning (min_train_days=3)

- Output:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_v9_phase2_hgb_seq_hgtune_mintrain3_20260219/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_v9_phase2_hgb_seq_hgtune_mintrain3_20260219/ws6_signoff.json`
- Gate summary:
  - `hard_gate_checks_passed_eligible = 20`
  - `hard_gate_checks_total_eligible = 30`
  - `hard_gate_checks_passed_full = 38`
  - `hard_gate_checks_total_full = 60`
- Observation:
  - No uplift vs current-code anchor-config rerun (`20/30`).

## 6. LivingRoom Recovery Sweeps

### Seed-11 broad sweep (10 variants)

- Summary file:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/lr_sweep_seed11_20260219/summary.json`
- Result:
  - Best variants were sequence-enabled (`v9`, `v10`) but still:
    - `livingroom_pass = 0/2` eligible splits
    - mean `occupied_recall ~= 0.472`
    - mean `occupied_f1 ~= 0.362`
  - No variant achieved a LivingRoom eligible hard-gate pass in seed 11.

### Seed-11 boundary-reweighting sweep (5 variants)

- Summary file:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/lr_sweep_boundary_seed11_20260219/summary.json`
- Result:
  - Enabling `--enable-bedroom-livingroom-boundary-reweighting` did not produce LivingRoom eligible pass in seed 11.
  - Top variant remained `b3_phase2_hgb_boundary_seq_hgtune` with same LivingRoom pass count (`0/2`).

## 7. Baseline Drift Check

- Current-code rerun with anchor-style flags:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/currentcode_anchorcfg_mintrain3_20260219/ws6_rolling.json`
  - Result: `20/30` eligible, `38/60` full.
- Historical anchor artifact:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_next_ab_min3_smooth_kitchen_tune/ws6_rolling.json`
  - Result: `23/30` eligible, `41/60` full.
- Interpretation:
  - There is run-environment/data drift vs historical anchor conditions. Current optimization should use today’s rerun baseline (`20/30`) for fair deltas.

## 8. Additional Exploration (Post-Drift)

### Sequence/Segment sweep promoted candidate (3 seeds)

- Candidate:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_v9_phase2_hgb_seq_hgtune_mintrain3_20260219/ws6_rolling.json`
- Result:
  - `20/30` eligible, `38/60` full.
  - No improvement vs current-code anchor-config rerun baseline.

### Boundary-reweighting validation

- Sweep summary:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/lr_sweep_boundary_seed11_20260219/summary.json`
- Result:
  - Turning on `--enable-bedroom-livingroom-boundary-reweighting` did not create LivingRoom eligible pass in seed 11.

### Transformer Stage-A probe (seed 11)

- Artifact:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/candidate_transformer_seed11_mintrain3_20260219.json`
- Result:
  - `6/10` eligible on seed 11 (worse than best seed-11 candidates at `7/10`).
  - LivingRoom eligible splits remained failing (`0/2`).

## 9. Failure Forensics Artifact

- Generated LivingRoom deep-dive report:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_livingroom_failure_forensics_2026-02-20.md`
- Generated ops-ready FN window CSV for label review:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/livingroom_day7_fn_windows_for_label_review_2026-02-20.csv`
- Highlights:
  - Persistent worst windows around day-7 LivingRoom long FN episodes.
  - `occupied_f1` floor remains the dominant blocker even when recall improves.

## 10. Ops Label-Correction Ingest (Implemented 2026-02-20)

### Scope

- Added correction-ingest helpers in:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`
    - `_load_activity_label_corrections(...)`
    - `_apply_activity_label_corrections(...)`
- Integrated correction flow into `run_backtest(...)` with default-off behavior:
  - new argument `label_corrections_csv: Optional[Path] = None`
  - correction load/apply executed after room-day data load and before split generation
  - report now includes:
    - `label_corrections.load`
    - `label_corrections.apply`
  - config hash payload now includes `label_corrections_csv`.
- Added CLI flag:
  - `--label-corrections-csv`
  - expected columns: `room,label,start_time,end_time[,day]`

### Validation

- Unit tests added in:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
    - path-missing default behavior
    - valid/invalid row parsing with day fallback
    - required-column validation
    - in-memory apply behavior + skip-reason accounting
- Regression suite:
  - `134 passed, 1 warning`
  - command:
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend && pytest -q \
  tests/test_segment_proposal.py \
  tests/test_segment_features.py \
  tests/test_segment_classifier.py \
  tests/test_segment_projection.py \
  tests/test_event_models.py \
  tests/test_timeline_gates.py \
  tests/test_event_first_backtest_script.py \
  tests/test_event_first_backtest_aggregate.py \
  tests/test_d2_strict_splitseed_integration.py
```

### Smoke Run

- Command:
```bash
python3 /Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 6 --seed 11 \
  --label-corrections-csv /tmp/label_corrections_smoke.csv \
  --output /tmp/ws6_beta55_upgrade_smoke_label_corrections.json
```
- Output:
  - `/tmp/ws6_beta55_upgrade_smoke_label_corrections.json`
- Key confirmation:
  - `label_corrections.load.rows_loaded = 1`
  - `label_corrections.apply.applied_windows = 1`
  - `label_corrections.apply.applied_rows = 3`

## 11. Stage-A Minute-Grid Pilot (BL Rooms, 3 Seeds)

- Detailed report:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_stagea_minute_pilot_2026-02-20.md`
- Aggregate artifacts:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/stagea_minute_pilot_20260220/variantA_baseline/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/stagea_minute_pilot_20260220/variantB_minute/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/stagea_minute_pilot_20260220/variantC_minute_plus/ws6_rolling.json`
- Summary:
  - Variant A (10s baseline): `21/30` eligible, `39/60` full.
  - Variant B (minute-grid): `18/30` eligible, `36/60` full.
  - Variant C (minute-grid + decoder/smoothing): `18/30` eligible, `36/60` full.
- Decision:
  - Minute-grid Stage-A remains default-off for now due net hard-gate regression.

## 12. Corrected-Data Matrix + Minute-Grid Isolation (2026-02-20)

- Detailed report:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_corrected_data_matrix_2026-02-20.md`
- Artifacts root:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/corrected_matrix_20260220`
- Smoke verification (ingest check):
  - Day-7 LivingRoom GT occupied rate moved from `5.36%` (uncorrected) to `37.17%` (corrected).
- Matrix summary:
  - Baseline corrected (Candidate D path): `24/30` eligible.
  - Learned classifier (`0.55` and `0.30`): no uplift vs baseline (`24/30`).
  - Minute-grid global: regression to `21/30` (Bedroom collapse).
  - Minute-grid LivingRoom-only isolation: recovered to `24/30` (no Bedroom collapse).

## 13. Bedroom Day-8 Audit + Fragmentation Sweep (2026-02-20)

### 13.1 Anchor Freeze

- New corrected-data anchor snapshot created:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/anchor_corrected_variantA_20260220`
- Includes:
  - `seed11/22/33.json`
  - `ws6_rolling.json`
  - `ws6_signoff.json`
  - `SHA256SUMS.txt`
  - `ANCHOR_METADATA.json`

### 13.2 Bedroom Day-8 Audit

- Report:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_bedroom_day8_audit_2026-02-20.md`
- Result:
  - No high-confidence auto-correction applied.
  - Empty correction rerun stayed at `24/30` eligible, `45/60` full.

### 13.3 LivingRoom Fragmentation Sweep

- Report:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_livingroom_fragmentation_sweep_2026-02-20.md`
- Seed-11 sweep root:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/fragmentation_sweep_seed11_20260220`
- Top-2 3-seed validation root:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/fragmentation_top2_3seed_20260220`
- Outcome:
  - `top2_frag_v3_hg_less_smooth`: `26/30` eligible, `47/60` full
  - `top2_frag_v6_pred_smooth`: `26/30` eligible, `47/60` full
- Recommended next candidate:
  - `top2_frag_v3_hg_less_smooth` (simpler config delta)
  - `--hard-gate-fragmentation-min-run-windows "bedroom=9,livingroom=6"`
  - `--hard-gate-fragmentation-gap-fill-windows "bedroom=6,livingroom=3"`


### 13.4 Strict Baseline-Bound Aggregate (Candidate `top2_frag_v3_hg_less_smooth`)

- Output artifacts:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/fragmentation_top2_3seed_20260220/top2_frag_v3_hg_less_smooth/ws6_rolling_strict_bound.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/fragmentation_top2_3seed_20260220/top2_frag_v3_hg_less_smooth/ws6_signoff_strict_bound.json`
- Baseline binding:
  - baseline version: `beta55_corrected_variantA_baseline_20260220`
  - baseline artifact path: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/anchor_corrected_variantA_20260220/ws6_signoff.json`
  - baseline artifact hash verified.
- Result:
  - `gate_decision = FAIL` (expected at this stage)
  - improvement remains: `26/30` eligible hard-gate vs anchor `24/30`
  - remaining blockers: Bedroom day-8 recall/F1/sleep and one LivingRoom day-8 fragmentation seed.


## 14. Anchor Promotion (2026-02-20)

- Promoted new working anchor:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/anchor_corrected_fragv3_20260220`
- Anchor source:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/fragmentation_top2_3seed_20260220/top2_frag_v3_hg_less_smooth`
- Anchor performance:
  - Eligible hard-gate: `26/30`
  - Full hard-gate: `47/60`
- Config delta vs prior corrected baseline:
  - `--hard-gate-fragmentation-min-run-windows "bedroom=9,livingroom=6"`
  - `--hard-gate-fragmentation-gap-fill-windows "bedroom=6,livingroom=3"`
- Included files:
  - `seed11/22/33.json`
  - `ws6_rolling.json`
  - `ws6_signoff.json`
  - `ws6_rolling_strict_bound.json`
  - `ws6_signoff_strict_bound.json`
  - `SHA256SUMS.txt`
  - `ANCHOR_METADATA.json`

