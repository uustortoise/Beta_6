# Beta 5.5 LivingRoom Night Guardrail Execution (2026-02-25)

## Objective
Implement and validate a targeted guardrail for the known overnight LivingRoom FP pattern:
- If Bedroom sleep is strongly active during night hours,
- and there is no Bedroom exit evidence,
- and no strong LivingRoom entry evidence,
then suppress LivingRoom occupied output (default label: `unknown`).

## Code Changes
1. Added a centralized hour-range helper and reused it:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_backtest.py`

2. Extended LivingRoom cross-room decoder with a night Bedroom-sleep guardrail:
- Function: `_apply_livingroom_cross_room_presence_decoder`
- Added:
  - night guard config params (hours, thresholds, confirmation windows)
  - Bedroom coverage/flatline sensor-health gating
  - suppression mode (`unknown` or `unoccupied`)
  - debug telemetry fields for auditability

3. Wired new params end-to-end:
- `run_backtest(...)` signature and run config payloads
- CLI args in `main()` parser
- backtest invocation argument mapping

4. Added profile for reproducible A/B execution:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_matrix_profiles.yaml`
- Variant: `lr_cross_room_presence_v4_night_guard`
- Profiles:
  - `lr_cross_room_presence_night_guard_quick`
  - `lr_cross_room_presence_night_guard_full`

5. Added unit tests:
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_event_first_backtest_script.py`
- New tests:
  - night guard suppresses overnight LR FP windows
  - night guard still allows strong LR entry (with consecutive-window confirmation)

## Validation Executed
### Unit tests
- `pytest -q tests/test_event_first_backtest_script.py -k "livingroom_cross_room_presence_decoder or parse_hour_range"`
  - Result: 6 passed
- `pytest -q tests/test_run_event_first_matrix.py`
  - Result: 3 passed

### Targeted backtest A/B (day-7-focused slice)
- Dataset: `/Users/dicksonng/DT/Development/New training files`
- Elder: `HK0011_jessica`
- Seed: `11`
- Window: days 6-7 (single split testing day 7)

Artifacts:
- Anchor: `/tmp/beta55_night_guard_ab_20260225/anchor_seed11_d6d7.json`
- Guarded v4: `/tmp/beta55_night_guard_ab_20260225/v4_night_guard_seed11_d6d7.json`
- Comparison CSV:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_lr_night_guard_d6d7_seed11_compare_2026-02-25.csv`

## Results (Day 7, Seed 11, d6-d7 slice)
1. Overnight target FP overlap (`2025-12-07 03:00:50–04:36:44`):
- Anchor: 95.90 min
- v4 night guard: 52.33 min
- Delta: -43.57 min (~45.4% reduction)

2. LivingRoom classification:
- `occupied_f1`: 0.1431 -> 0.2827 (improved)
- `occupied_recall`: 0.1780 -> 0.8314 (improved)
- `fragmentation_score`: 0.6190 -> 0.2442 (regressed vs fragmentation gate)
- `accuracy`: 0.8482 -> 0.7086 (regressed)
- `macro_f1`: 0.5252 -> 0.3710 (regressed)

3. Guardrail telemetry (v4):
- `night_bedroom_guard_applied`: true
- `night_bedroom_guard_suppressed_windows`: 149
- `night_bedroom_guard_unknown_windows`: 60
- `night_bedroom_guard_blocked_entries`: 639

## Interpretation
1. The guardrail successfully cuts the specific overnight FP block (the intended target).
2. It is currently too aggressive globally in this slice, shifting error balance toward precision/accuracy regression.
3. This indicates the guard should remain default-off and be tuned with tighter scope, not promoted as-is.

## Recommended Next Tuning Step
1. Keep guard enabled only for overnight hours (`22:00-06:00`) as implemented.
2. Reduce suppression aggressiveness by gating suppression to:
- only when LR currently predicted occupied from prior decoder state, and
- requiring sustained Bedroom sleep evidence (consecutive windows), not single-window checks.
3. Keep `unknown` fallback (do not force-room assignment).
4. Re-run one seed-11 quick A/B, then 3-seed if quick run improves both:
- overnight FP overlap and
- non-regression on accuracy/macro_f1/fragmentation.

## Constraint Encountered
A full 4-10 day rerun through this terminal environment did not reliably produce an artifact before process termination in-session. The day-7 focused slice was used to validate the guardrail mechanics and impact. Full 3-seed promotion assessment should run in the team’s stable host runner.
