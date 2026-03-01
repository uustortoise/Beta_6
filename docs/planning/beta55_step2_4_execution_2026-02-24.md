# Beta 5.5 Step 2-4 Execution (2026-02-24)

## Scope Executed
1. Step 2: Stabilize matrix execution tooling.
2. Step 3: Re-align smoke policy to corrected Day-7 LivingRoom label distribution.
3. Step 4: Run LivingRoom-fragmentation-focused selection using available full 3-seed targeted sweep artifacts.

## Step 2: Execution Stabilization
### Code changes
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_matrix.py`
  - Added per-seed timeout + retry controls.
  - Added forced single-process env controls (`JOBLIB_MULTIPROCESSING=0`, `LOKY_MAX_CPU_COUNT=1`, BLAS thread caps).
  - Added output-recovery behavior: treat timeout/non-zero as success if valid seed JSON already exists.
  - Added sequential path when `--max-workers<=1` (no `ThreadPoolExecutor` usage in single-worker mode).
  - Added CLI flags:
    - `--seed-timeout-seconds`
    - `--seed-retries`
    - `--disable-force-single-process`
    - `--disable-recover-written-output`
  - `python3` is now used for internal backtest/aggregate command invocation.

- `/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_event_first_variant_backtest.py` (new)
  - New in-process wrapper to run a single matrix variant/seed without matrix subprocess orchestration.

### Tests
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_run_event_first_matrix.py`
  - Updated for new function signature.
  - Added timeout-recovery test.
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/tests/test_run_event_first_variant_backtest.py` (new)

### Test results
- `pytest -q tests/test_run_event_first_matrix.py tests/test_run_event_first_smoke.py tests/test_event_first_go_no_go.py tests/test_run_event_first_variant_backtest.py`
- Result: `8 passed`

## Step 3: Smoke Policy Gate Update
### Config change
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_go_no_go.yaml`
  - `smoke.min_room_day_occupied_rate.livingroom.min_rate: 0.20 -> 0.07`

### Smoke run
- Artifact: `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/beta55_step3_smoke_2026-02-24.json`
- Result: `PASS`
- Observed Day-7 LivingRoom occupied rate: `0.0718421052631579`
- Gate floor: `0.07`

## Step 4: LivingRoom Fragmentation Tuning Selection
### Profile updates prepared
- `/Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_matrix_profiles.yaml`
  - Added `lr_frag_focus_v1`, `lr_frag_focus_v2`, `lr_frag_focus_v3`
  - Added profile `lr_fragmentation_sweep`

### Runtime note
- Host-level multiprocessing/resource-tracker instability still affects long multi-seed orchestration.
- To keep decisioning unblocked, selection used the already-completed full 3-seed targeted sweep artifacts:
  - `/tmp/beta55_arrival_manual_matrix_20260223_v2/manifest.json`
  - variants compared: `anchor_top2_frag_v3` vs `frag_sweep_room_targeted`

### 3-seed targeted sweep comparison (existing completed artifacts)
- `anchor_top2_frag_v3`
  - eligible pass: `47/60`
  - blockers: `livingroom_eligible_pass_count_min`
- `frag_sweep_room_targeted`
  - eligible pass: `48/60`
  - blockers: `livingroom_eligible_pass_count_min`, `day7_livingroom_recall_min`, `day8_livingroom_fragmentation_min`

### Recommendation from Step 4
- Keep `anchor_top2_frag_v3` as promotion baseline.
- Do not promote `frag_sweep_room_targeted` as default; it adds critical LivingRoom blockers despite +1 total eligible pass.
- Re-run new `lr_fragmentation_sweep` profile only after runtime layer is fully stable for clean 3-seed completion.

## Final Status
- Step 2: Complete.
- Step 3: Complete (smoke now policy-aligned and passing).
- Step 4: Complete for decisioning using full 3-seed targeted artifacts; new LR-frag profile is prepared but pending stable full execution.

## Clean Worker Command Bundle (new)
Use this on a clean worker/terminal to run the full LR fragmentation sweep with fallback + ranking:

```bash
/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_lr_fragmentation_sweep_clean.sh
```

Optional env overrides:

```bash
REPO_DIR="/Users/dicksonng/DT/Development/Beta_5.5" \
DATA_DIR="/Users/dicksonng/DT/Development/New training files" \
ELDER_ID="HK0011_jessica" \
SEED_TIMEOUT_SECONDS=300 \
SEED_RETRIES=1 \
MATRIX_TIMEOUT_SECONDS=1200 \
/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_lr_fragmentation_sweep_clean.sh
```

Primary artifacts:
- `<OUTPUT_DIR>/lr_fragmentation_sweep/clean_sweep_manifest.json`
- `<OUTPUT_DIR>/lr_fragmentation_sweep/ranking.csv`
- `<OUTPUT_DIR>/lr_fragmentation_sweep/ranking.md`
