# Beta 5.5 Label Pack Rerun Execution (2026-02-23, Updated 5-10 Dec)

## Input
- Candidate pack: `/Users/dicksonng/DT/Development/New training files`
- Baseline pack for diff: `/Users/dicksonng/DT/Development/New training files/corrected_clones`
- Elder: `HK0011_jessica`
- Day window: `4-10`

## 1) Intake Validation
- Output: `/tmp/beta55_arrival_validate_20260223_v2.json`
- Result: `PASS` (`violations=0`, `warnings=0`)

## 2) Baseline-vs-Candidate Diff
- Outputs:
  - `/tmp/beta55_arrival_diff_20260223_v2.json`
  - `/tmp/beta55_arrival_diff_20260223_v2.csv`
- Summary:
  - `windows_changed_total=40662`
  - `minutes_changed_total=6777.0`
  - `episodes_added_total=2659`
  - `episodes_removed_total=0`

## 3) Smoke Gate
- Output: `/tmp/beta55_arrival_smoke_20260223_v2.json`
- Result: `FAIL`
- Blocking reason:
  - `occupied_rate_below_threshold:livingroom:day7`
- Detail:
  - Day-7 LivingRoom occupied-rate observed `0.0718`
  - Configured minimum `0.20`

## 4) Full Matrix (4 variants x 3 seeds)
`run_event_first_matrix.py` was unstable in this environment (parent process hang), so matrix was executed via manual equivalent orchestration (same backtest + aggregate + go/no-go logic).

- Manifest: `/tmp/beta55_arrival_manual_matrix_20260223_v2/manifest.json`
- Overall status: `FAIL`

### Variant results
1. `anchor_top2_frag_v3`
   - Eligible passes: `47/60`
   - Blockers:
     - `livingroom_eligible_pass_count_min`
2. `frag_sweep_room_targeted`
   - Eligible passes: `48/60`
   - Blockers:
     - `livingroom_eligible_pass_count_min`
     - `day7_livingroom_recall_min`
     - `day8_livingroom_fragmentation_min`
3. `lr_only_minute_grid_optional`
   - Eligible passes: `46/60`
   - Blockers:
     - `livingroom_eligible_pass_count_min`
     - `bedroom_max_regression_splits`
     - `day7_livingroom_recall_min`
     - `day8_livingroom_fragmentation_min`
4. `learned_segment_classifier_off`
   - Eligible passes: `45/60`
   - Blockers:
     - `livingroom_eligible_pass_count_min`
     - `bedroom_max_regression_splits`
     - `day7_livingroom_recall_min`
     - `day8_livingroom_fragmentation_min`

## 5) Before/After (Anchor)
Compared against pre-arrival full 3-seed anchor baseline.

- Outputs:
  - `/tmp/beta55_arrival_before_after_anchor_20260223_v2.md`
  - `/tmp/beta55_arrival_before_after_anchor_20260223_v2.csv`

### Key deltas (after - before, anchor)
- Bedroom:
  - Accuracy `+0.0974`
  - Macro F1 `+0.1260`
  - Occupied F1 `+0.2942`
  - Occupied recall `+0.1866`
- LivingRoom:
  - Accuracy `-0.0308`
  - Macro F1 `-0.1656`
  - Occupied F1 `-0.3959`
  - Occupied recall `-0.2323`

## Evaluation
- The updated 5-10 Dec label pack materially improved Bedroom performance and removed multiple previous blockers.
- Promotion still blocked by LivingRoom criteria:
  - smoke occupied-rate policy floor for Day 7
  - matrix gate `livingroom_eligible_pass_count_min`
- Current best candidate remains `anchor_top2_frag_v3` for next iteration.
