# Beta 5.5 Label Pack Arrival Execution (2026-02-23)

## Scope
End-to-end arrival-day execution for new 4-10 Dec training files:
1. Intake validation
2. Baseline-vs-new diff
3. Smoke gate
4. Full matrix (all configured variants, seeds 11/22/33)
5. Before/after summary (old baseline vs new pack)

## Input Data
- New pack (candidate): `/Users/dicksonng/DT/Development/New training files`
- Previous baseline pack: `/Users/dicksonng/DT/Development/New training files/corrected_clones`
- Elder: `HK0011_jessica`
- Day window: `4-10`

## Intake Results
### 1) Validate label pack
- Command: `python3 backend/scripts/validate_label_pack.py --pack-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 10 --output /tmp/beta55_arrival_validate_20260223.json`
- Result: `PASS`
- Report: `/tmp/beta55_arrival_validate_20260223.json`

### 2) Diff baseline vs candidate
- Command: `python3 backend/scripts/diff_label_pack.py --baseline-dir "/Users/dicksonng/DT/Development/New training files/corrected_clones" --candidate-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 10 --json-output /tmp/beta55_arrival_diff_20260223.json --csv-output /tmp/beta55_arrival_diff_20260223.csv`
- Result: `PASS`
- Report JSON: `/tmp/beta55_arrival_diff_20260223.json`
- Report CSV: `/tmp/beta55_arrival_diff_20260223.csv`
- Summary:
  - `windows_changed_total=32818`
  - `minutes_changed_total=5469.67`
  - `episodes_added_total=2554`
  - `episodes_removed_total=0`

## Smoke Result
### 3) Smoke gate
- Command: `python3 backend/scripts/run_event_first_smoke.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --day 7 --seed 11 --expectation-config backend/config/event_first_go_no_go.yaml --diff-report /tmp/beta55_arrival_diff_20260223.json --output /tmp/beta55_arrival_smoke_20260223.json`
- Result: `FAIL`
- Report: `/tmp/beta55_arrival_smoke_20260223.json`
- Blocking reason:
  - `occupied_rate_below_threshold:livingroom:day7`
- Check details:
  - Observed Day-7 LivingRoom occupied rate: `0.0718`
  - Configured minimum: `0.20`

## Full Matrix Results
### 4) Matrix execution
Due to wrapper hang behavior in `run_event_first_matrix.py` under this runtime, variants were executed with equivalent manual orchestration:
- For each configured variant:
  - run `run_event_first_backtest.py` for seeds `11,22,33`
  - aggregate via `aggregate_event_first_backtest.py`
  - evaluate go/no-go using `backend/config/event_first_go_no_go.yaml`

Artifacts root:
- `/tmp/beta55_arrival_manual_matrix_20260223`
- Manifest: `/tmp/beta55_arrival_manual_matrix_20260223/manifest.json`

Overall matrix status:
- `FAIL`

Per-variant go/no-go:
1. `anchor_top2_frag_v3`
   - Eligible passes: `36/60`
   - Blockers:
     - `livingroom_eligible_pass_count_min`
     - `bedroom_max_regression_splits`
     - `day8_bedroom_sleep_recall_min`
2. `frag_sweep_room_targeted`
   - Eligible passes: `39/60`
   - Blockers:
     - `livingroom_eligible_pass_count_min`
     - `bedroom_max_regression_splits`
     - `day7_livingroom_recall_min`
     - `day8_bedroom_sleep_recall_min`
     - `day8_livingroom_fragmentation_min`
3. `lr_only_minute_grid_optional`
   - Eligible passes: `37/60`
   - Blockers:
     - `livingroom_eligible_pass_count_min`
     - `bedroom_max_regression_splits`
     - `day7_livingroom_recall_min`
     - `day8_bedroom_sleep_recall_min`
     - `day8_livingroom_fragmentation_min`
4. `learned_segment_classifier_off`
   - Eligible passes: `37/60`
   - Blockers:
     - `livingroom_eligible_pass_count_min`
     - `bedroom_max_regression_splits`
     - `day7_livingroom_recall_min`
     - `day8_bedroom_sleep_recall_min`
     - `day8_livingroom_fragmentation_min`

## Before/After Comparison (Anchor)
### 5) Standardized report
- Command: `python3 backend/scripts/summarize_before_after.py --before-rolling /tmp/beta55_prearrival_matrix_full/pre_arrival_full/anchor_top2_frag_v3/rolling.json --before-signoff /tmp/beta55_prearrival_matrix_full/pre_arrival_full/anchor_top2_frag_v3/signoff.json --after-rolling /tmp/beta55_arrival_manual_matrix_20260223/anchor_top2_frag_v3/rolling.json --after-signoff /tmp/beta55_arrival_manual_matrix_20260223/anchor_top2_frag_v3/signoff.json --markdown-output /tmp/beta55_arrival_before_after_anchor_20260223.md --csv-output /tmp/beta55_arrival_before_after_anchor_20260223.csv`
- Output markdown: `/tmp/beta55_arrival_before_after_anchor_20260223.md`
- Output csv: `/tmp/beta55_arrival_before_after_anchor_20260223.csv`

Room metric deltas (after - before):
| Room | Delta Accuracy | Delta Macro F1 | Delta Occupied F1 | Delta Occupied Recall | Delta Fragmentation |
|---|---:|---:|---:|---:|---:|
| Bedroom | -0.0578 | -0.0776 | +0.0513 | +0.1155 | +0.1695 |
| LivingRoom | -0.0308 | -0.1656 | -0.3959 | -0.2323 | +0.2606 |
| Kitchen | +0.0058 | +0.0025 | -0.0013 | +0.0230 | +0.3654 |
| Bathroom | -0.0635 | +0.0020 | -0.1059 | -0.0336 | +0.1765 |
| Entrance | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## Decision
- Current state is **NO-GO** for promotion with this pack under current thresholds and model configs.
- Primary failures are concentrated in:
  - LivingRoom eligible pass floor
  - Bedroom regression guard
  - Day-8 Bedroom sleep recall floor
  - (most non-anchor variants) Day-7 LR recall and Day-8 LR fragmentation floor

## Recommended Next Steps
1. Re-run Day-7 smoke criterion with an updated expected floor for LivingRoom occupied-rate if 0.20 no longer reflects revised labeling policy.
2. Run focused Day-8 forensics on Bedroom sleep and LivingRoom fragmentation from anchor seed reports in `/tmp/beta55_arrival_manual_matrix_20260223/anchor_top2_frag_v3/seed_*.json`.
3. Keep `anchor_top2_frag_v3` as control baseline for next correction iteration; it remains the least-bad variant under current gates.
