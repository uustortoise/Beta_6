# Beta6 LivingRoom v52 Robustness Validation

## Scope

- Keep promotion plumbing closed unless model-side evidence forces a reopen.
- Determine whether `LivingRoom_v52` is already promotion-grade beyond the corrected-pack Dec 17 confirmation.
- Use the saved fresh-retrain artifacts and replay summaries as the primary evidence source.

## Evidence reviewed

- Fresh retrain panel:
  - `tmp/jessica_livingroom_fresh_20260311T023304Z/train_metrics.json`
- Fresh candidate traces:
  - `backend/models/HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z/LivingRoom_v49_decision_trace.json`
  - `backend/models/HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z/LivingRoom_v50_decision_trace.json`
  - `backend/models/HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z/LivingRoom_v51_decision_trace.json`
  - `backend/models/HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z/LivingRoom_v52_decision_trace.json`
- Current live comparator trace:
  - `backend/models/HK0011_jessica/LivingRoom_v46_decision_trace.json`
- Confirmed Dec 17 replay summaries:
  - `tmp/jessica_17dec_eval_candidate_livingroom_fresh_20260311T023304Z/final_v52/comparison/summary.json`
  - `tmp/jessica_17dec_eval_live_recheck_20260311T023304Z/final/comparison/summary.json`

## Findings

### 1. `v52` is a real model-side improvement over the old `v46` / `v49` safe shape

Saved checkpoint evidence:

- `v46`
  - holdout macro-F1 `0.6729`
  - gate-aligned score `1.2915`
  - predicted distribution `1969 / 6148`
  - thresholds:
    - `livingroom_normal_use=0.5220`
    - `unoccupied=0.9141`
- `v52`
  - holdout macro-F1 `0.7230`
  - gate-aligned score `1.3740`
  - predicted distribution `1704 / 6413`
  - thresholds:
    - `livingroom_normal_use=0.4316`
    - `unoccupied=0.7362`

Interpretation:

- `v52` is not just a metadata or promotion artifact; it has a materially stronger saved checkpoint than the older `v46` geometry.
- The gain is not coming from looser sampling drift. Both versions report the same post-sampling prior drift (`2.1215` percentage points).

### 2. Dec 17 confirms the gain, but only for one replay date

Dec 17 replay delta vs live `v46`:

- overall accuracy `+0.0004`
- overall macro-F1 `+0.0142`
- LivingRoom accuracy `+0.0019`
- LivingRoom macro-F1 `+0.0475`
- Bedroom macro-F1 `+0.0000`

Interpretation:

- The verified runtime win is real and isolated to `LivingRoom`.
- This is still only one corrected-pack replay date.

### 3. The fresh panel does not prove unique stability for `v52`

Fresh panel versions retained in the candidate namespace:

- `v49` / seed `42`
  - holdout macro-F1 `0.6729`
  - passes no-regress floor
- `v50` / seed `43`
  - holdout macro-F1 `0.6964`
  - passes no-regress floor
- `v51` / seed `44`
  - holdout macro-F1 `0.6944`
  - passes no-regress floor
- `v52` / seed `45`
  - holdout macro-F1 `0.7230`
  - passes no-regress floor

Interpretation:

- The fresh retrain did not collapse into a single surviving seed the way the earlier `v39..v43` panel did.
- `v52` is the best saved checkpoint in the current panel, but it is not the only no-regress-safe checkpoint.
- That means the panel supports "best current winner" more strongly than "uniquely robust seed."

### 4. Geometry across the passing seeds is still meaningfully unstable

Passing neighbors differ sharply from each other:

- `v49` keeps a very conservative `unoccupied` threshold (`0.9141`) and looks like the old `v46` safe anchor.
- `v50` keeps a much lower `unoccupied` threshold (`0.7363`) but predicts a more unoccupied-heavy distribution than `v52` (`6863` vs `6413`).
- `v51` flips `livingroom_normal_use` into fallback-threshold behavior with a very low active threshold (`0.1961`), even though it still clears the floor.
- `v52` improves the holdout score while also concentrating a very high near-threshold share on `livingroom_normal_use` (`0.8485`), which does not read as obviously stable.

Interpretation:

- The panel is healthier than the earlier brittle split, but it still shows multiple materially different passing geometries.
- That is not enough evidence to claim the selected `v52` shape is stable across dates or conditions.

## Conclusion

Current verdict:

- `LivingRoom_v52` is the leading candidate.
- `LivingRoom_v52` is better than the old `v46` / `v49` anchor on saved holdout evidence and on the verified Dec 17 replay.
- `LivingRoom_v52` is not yet proven promotion-grade beyond Dec 17.

Why the verdict stays conservative:

- only one replay date has been validated end to end
- nearby seeds `v50` and `v51` also clear the no-regress floor
- the passing seed geometries remain meaningfully different rather than converging on one clearly stable shape

Recommended next step:

- keep promotion closed
- treat `v52` as the best current candidate, not as a fully proven robust winner
- finish the cross-date replay sweep on corrected-pack dates `4, 5, 6, 7, 8, 9, 10 Dec 2025`
- only open a new brittleness forensic if those additional replays show date-specific reversals or large metric spread between `v49..v52`
