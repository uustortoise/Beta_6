# Beta6 Jessica LivingRoom Seed Forensic

## Scope

- Keep live `HK0011_jessica` frozen at Bathroom `v35`, Bedroom `v38`, Entrance `v26`, Kitchen `v27`, LivingRoom `v40`.
- Explain why `LivingRoom_v40` succeeds while `LivingRoom_v39`, `v41`, `v42`, and `v43` fail under the same no-downsample recipe.
- Use only saved artifacts from the synced live namespace in this worktree.

## Reproducible inputs

- Forensic helper: `backend/scripts/livingroom_seed_forensic.py`
- Regression test: `backend/tests/test_livingroom_seed_forensic.py`
- Generated artifact:
  - `tmp/jessica_livingroom_seed_forensic_20260310T233753Z.json`
- Source traces:
  - `backend/models/HK0011_jessica/LivingRoom_versions.json`
  - `backend/models/HK0011_jessica/LivingRoom_v39_decision_trace.json`
  - `backend/models/HK0011_jessica/LivingRoom_v40_decision_trace.json`
  - `backend/models/HK0011_jessica/LivingRoom_v41_decision_trace.json`
  - `backend/models/HK0011_jessica/LivingRoom_v42_decision_trace.json`
  - `backend/models/HK0011_jessica/LivingRoom_v43_decision_trace.json`
  - `backend/models/HK0011_jessica/LivingRoom_v39_activity_confidence_calibrator.json`
  - `backend/models/HK0011_jessica/LivingRoom_v40_activity_confidence_calibrator.json`
  - `backend/models/HK0011_jessica/LivingRoom_v41_activity_confidence_calibrator.json`
  - `backend/models/HK0011_jessica/LivingRoom_v42_activity_confidence_calibrator.json`
  - `backend/models/HK0011_jessica/LivingRoom_v43_activity_confidence_calibrator.json`

## Findings

### 1. The compared recipes are identical except for `random_seed`

The forensic artifact reports:

- `winner_version = 40`
- `winner_seed = 42`
- `policies_match_except_random_seed = true`

Across the five compared versions, the saved policy matches after removing `policy.reproducibility.random_seed`. The seeds line up as:

- `v39 -> seed 41`
- `v40 -> seed 42`
- `v41 -> seed 43`
- `v42 -> seed 44`
- `v43 -> seed 45`

Interpretation:

- this panel is a seed-only split under the same saved LivingRoom recipe
- there is no evidence here of a hidden policy change between winner and losers

### 2. Sampling drift is held constant across the full panel

From `tmp/jessica_livingroom_seed_forensic_20260310T233753Z.json`:

- `post_sampling_prior_drift_pp_range.min = 3.9661816836198627`
- `post_sampling_prior_drift_pp_range.max = 3.9661816836198627`
- `post_sampling_prior_drift_pp_range.spread = 0.0`

Each version also retained the same tiny post-downsample drift:

- `post_downsample_prior_drift_pp = 0.01108646391684287`

Interpretation:

- the prior no-downsample fix remains intact
- the residual failure is not renewed sampling drift

### 3. The winner is the only seed that reaches a recoverable validation geometry

`v40` / seed `42`:

- `macro_f1 = 0.6589739603000081`
- `selection_mode = no_regress_floor`
- `reaches_no_regress_floor = true`
- best-epoch distribution:
  - `livingroom_normal_use = 200`
  - `unoccupied = 1564`
- thresholds:
  - `0 = 0.0`
  - `1 = 0.5245264121945039`
- activity-confidence intercept:
  - `-5.233748830412337`

The neighboring failed seeds never reach the floor:

- `v39` / seed `41`
  - `macro_f1 = 0.0815475350114022`
  - `selection_mode = no_regress_macro_f1_fallback`
  - best-epoch distribution `1673 / 91`
  - threshold `1 = 0.966763888530709`
  - intercept `3.2821553844705624`
- `v41` / seed `43`
  - `macro_f1 = 0.08035615790717832`
  - `selection_mode = no_regress_macro_f1_fallback`
  - best-epoch distribution `1675 / 89`
  - threshold `1 = 0.9525418334363592`
  - intercept `1.2516648602551956`
- `v42` / seed `44`
  - `macro_f1 = 0.0815475350114022`
  - `selection_mode = no_regress_macro_f1_fallback`
  - best-epoch distribution `1673 / 91`
  - threshold `1 = 0.9644214354448793`
  - intercept `3.730197810720776`
- `v43` / seed `45`
  - `macro_f1 = 0.07255520504731862`
  - `selection_mode = no_regress_macro_f1_fallback`
  - `collapsed_best_epoch = true`
  - best-epoch distribution `1764 / 0`
  - threshold `1 = 0.001183768967166543`
  - intercept `-1.8134694540313643`

Interpretation:

- the decisive split happens before runtime replay
- `v40` is the only seed whose selected checkpoint still preserves a strong unoccupied-majority geometry
- the failed seeds are already active-heavy at checkpoint-selection time

### 4. Threshold and confidence artifacts are downstream reflections of the checkpoint split

Winner-relative deltas from the forensic artifact:

- `v39`
  - unoccupied-threshold delta: `+0.44223747633620514`
  - activity-confidence intercept delta: `+8.5159042148829`
- `v41`
  - unoccupied-threshold delta: `+0.4280154212418553`
  - activity-confidence intercept delta: `+6.485413690667533`
- `v42`
  - unoccupied-threshold delta: `+0.43989502325037544`
  - activity-confidence intercept delta: `+8.963946641133113`
- `v43`
  - unoccupied-threshold delta: `-0.5233426432273374`
  - activity-confidence intercept delta: `+3.4202793763809733`

Interpretation:

- `v39`, `v41`, and `v42` all react to an already active-heavy checkpoint by pushing the unoccupied threshold toward `0.95`
- `v43` is worse: the selected checkpoint is collapsed enough that the threshold profile no longer resembles the winner at all
- these threshold and calibrator differences explain the saved artifacts, but they do not look like the first cause of failure

## Conclusion

The LivingRoom seed panel now has a narrower, more honest explanation:

- the compared `v39..v43` runs differ only by random seed
- the same no-downsample recipe stays in place for all five versions
- only seed `42` (`v40`) reaches a checkpoint with enough unoccupied preservation to clear the no-regress floor
- seeds `41`, `43`, and `44` stay active-heavy even at their selected best epoch
- seed `45` collapses fully into all-active prediction

The next thread should therefore stay narrow:

- keep live `HK0011_jessica` unchanged
- do not reopen confidence/runtime work
- do not reopen Bedroom support-gating work
- focus on LivingRoom optimizer / initialization / checkpoint-selection stability under the existing no-downsample recipe

The main question for the next thread is no longer "did downsampling drift again?" It is:

- why does seed `42` reach a recoverable validation geometry while the neighboring seeds cannot under the same recipe?
