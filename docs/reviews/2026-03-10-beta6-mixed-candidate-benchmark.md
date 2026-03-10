# Beta6 Mixed Candidate Benchmark

## Candidate Definition

Mixed candidate namespace:

- `Bathroom`: live `v35`
- `Bedroom`: earlier `v2` confidence-runtime fix (`v36`)
- `Entrance`: live `v26`
- `Kitchen`: live `v27`
- `LivingRoom`: fallback recalibration result (`v30`)

Benchmark:

- Source file: corrected `HK0011_jessica_train_17dec2025.xlsx`
- Output: `tmp/jessica_17dec_eval_candidate_mixed_20260310T090755Z/final_v3/comparison/summary.json`
- Baseline comparison target: `tmp/jessica_17dec_eval_candidate_blr_20260310T072625Z/baseline/comparison/summary.json`

Important correction:

- The first mixed-candidate benchmark was invalid for Bathroom.
- Root cause: this branch had not yet merged the Bathroom parity/runtime fixes from the local workspace, so registry discovery ignored/scrubbed the two-stage alias state incorrectly and the mixed namespace initially ran Bathroom on the wrong effective path.
- After merging the missing `legacy/registry.py` and `legacy/prediction.py` behavior and rerunning, the mixed candidate below is the first valid integrated result.

## Result

### Overall

- Baseline live-copy final:
  - accuracy `0.7334`
  - macro-F1 `0.3377`
- Mixed candidate final:
  - accuracy `0.7608`
  - macro-F1 `0.3922`

Raw-vs-final gap:

- Baseline:
  - accuracy gap `0.0731`
  - macro-F1 gap `0.1043`
- Mixed candidate:
  - accuracy gap `0.0426`
  - macro-F1 gap `0.0538`

Interpretation:

- The mixed candidate is materially better than the live-copy baseline.
- The final exported output is closer to raw top-1 than the live-copy baseline.
- Raw top-1 is slightly lower overall because Bathroom `v35` is a recall-favoring frontier, but the final exported output is materially stronger once Bedroom/LivingRoom confidence failures are removed and Bathroom parity is preserved.

### Room-level deltas

- Bathroom:
  - baseline final macro-F1 `0.3631`
  - mixed final macro-F1 `0.4200`
  - low-confidence rate `0.0077 -> 0.0044`
  - raw macro-F1 `0.5221 -> 0.5287`
- Bedroom:
  - baseline final macro-F1 `0.1816`
  - mixed final macro-F1 `0.2723`
  - low-confidence rate `0.2117 -> 0.0087`
- LivingRoom:
  - baseline final macro-F1 `0.2882`
  - mixed final macro-F1 `0.3856`
  - low-confidence rate `0.0071 -> 0.0000`
- Kitchen:
  - unchanged from baseline in this mixed candidate
- Entrance:
  - unchanged from baseline in this mixed candidate

## Blockers Identified

### Bedroom blocker

The Dec 17-added Bedroom recalibration is not promotable as-is.

Evidence:

- Combined-pack Bedroom recalibration produced:
  - final macro-F1 `0.1536`
  - low-confidence rate `0.4674`
- The `sleep` threshold moved to `0.40899087884201635`
- Raw top-1 `sleep` acceptance scores on Dec 17 sat around `0.3734`
- Result: almost every raw `sleep` row fell below threshold and rewrote to `low_confidence`

Interpretation:

- This is no longer the old raw-softmax/global-unknown bug.
- The blocker is now Bedroom score geometry after adding Dec 17 into calibration.
- Bedroom needs model-side work or a revised calibration-selection regime before a Dec 17-added unified retrain is promotable.

### LivingRoom blocker

The Dec 17-added LivingRoom direct retrain is not promotable as-is.

Evidence from direct retrain candidate `v29`:

- final validation accuracy `0.4796`
- gate rejected for:
  - `no_regress_failed:livingroom:drop=0.248>max_drop=0.050`
  - `lane_b_gate_failed:livingroom:collapse_livingroom_active`
- sampled class prior drift after unoccupied downsampling was about `19.64` percentage points

Interpretation:

- Existing LivingRoom weights are usable under recalibrated confidence.
- The blocker is the weight retraining recipe on the corrected pack, not the runtime confidence architecture.
- Direct retraining still collapses `livingroom_active` recall.

### Bathroom integration blocker that had to be fixed

The first mixed-candidate run hid a branch-integration problem rather than a Bathroom model problem.

Evidence:

- The mixed namespace had `Bathroom_v35_two_stage_*` versioned files, but the active aliases were not being protected/synced correctly in this branch.
- `legacy/registry.py` in this branch was also discovering `Bathroom_two_stage_stage_a_model.keras` and `Bathroom_two_stage_stage_b_model.keras` as if they were room aliases.
- Once the missing local Bathroom parity/runtime changes were merged into this branch, `load_models_for_elder()` loaded `platform.two_stage_core_models['Bathroom']` correctly again and the Dec 17 Bathroom metrics returned to the expected `v35` frontier.

Interpretation:

- Bathroom `v35` itself was not the blocker here.
- The blocker was branch drift: this worktree had the Bedroom/LivingRoom confidence architecture work but not the local Bathroom parity/runtime fixes.
- After that merge, the room-wise mixed candidate became a valid integrated benchmark.

## Honest Recommendation

1. If you want the strongest Dec 17 behavior now, use the mixed candidate shape.
2. Do not promote the Dec 17-added full retrain or the Dec 17-added Bedroom recalibration.
3. Keep Bathroom on the local `v35` recall-biased anchor when evaluating the mixed pack. Do not compare against the stale pre-merge mixed benchmark.
4. Treat Bedroom and LivingRoom retraining blockers as separate forensic/modeling tasks:
   - Bedroom: why combined-pack calibration pushes `sleep` above its score band
   - LivingRoom: why corrected-pack retraining collapses active recall under the current training recipe
