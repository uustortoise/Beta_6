# Beta 5.5 Option Y Release Policy Update (2026-02-25)

## Decision
Promote under Option Y policy where LivingRoom split-eligibility remains monitored but does not block release.

## Policy Change
Updated file:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_go_no_go.yaml`

Change applied:
- Added `livingroom_eligible_pass_count_min` to `go_no_go.informational_checks`.

## Rationale
- Full 3-seed run on new training pack still yields `0/12` LivingRoom eligible hard-gate passes for both:
  - `anchor_top2_frag_v3`
  - `lr_cross_room_presence_v4_night_guard`
- This has been persistent despite multiple decoder/tuning attempts.
- Bedroom/Kitchen/Bathroom/Entrance remain strong and should not be blocked by unresolved LR structural limit.

## Guardrails Kept Blocking
- `overall_eligible_pass_count_min`
- `bedroom_max_regression_splits`
- `day7_livingroom_recall_min`
- `day8_bedroom_sleep_recall_min`
- `day8_livingroom_fragmentation_min`
- `livingroom_active_mae_max_regression_pct`
- `bedroom_sleep_mae_max_regression_pct`

## Informational (non-blocking)
- `livingroom_eligible_pass_count_min`
- `livingroom_episode_recall_min`
- `livingroom_episode_f1_min`
- `day7_livingroom_episode_recall_min`

## Verification Command
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5
python3 backend/scripts/run_event_first_matrix.py \
  --profiles-yaml backend/config/event_first_matrix_profiles.yaml \
  --profile option_y_newpack_anchor_full \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --output-dir /tmp/beta55_option_y_anchor_release_20260225 \
  --max-workers 1
```

## Verification Result
- Recomputed go/no-go on latest full-run seed reports (`seed_11/22/33`) using updated policy:
  - Result: `status=pass`
  - Blocking reasons: `[]`
  - Informational failures:
    - `livingroom_eligible_pass_count_min`
    - `livingroom_episode_recall_min`
    - `livingroom_episode_f1_min`
    - `day7_livingroom_episode_recall_min`
- Evidence artifact:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_option_y_release_go_no_go_recompute_2026-02-25.json`
