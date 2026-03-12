# Beta6 LivingRoom Brittleness Forensic

## Scope

- Explain why the fresh LivingRoom seeds (`v50..v52`) can beat `v46` on Dec 17 while losing to the old anchor on most corrected-pack dates.
- Compare `v46`, `v50`, and `v52` on identical replay timestamps using the already-generated corrected-pack sweep.
- Use `v51` as a catastrophic control to test whether the fresh panel still contains unstable occupied-routing geometries.
- Decide whether the next move should be a narrow calibration/gate experiment or a deeper model-shape forensic.

## Inputs

Replay artifacts:

- `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v46/*/comparison/LivingRoom_merged.parquet`
- `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v50/*/comparison/LivingRoom_merged.parquet`
- `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v51/*/comparison/LivingRoom_merged.parquet`
- `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v52/*/comparison/LivingRoom_merged.parquet`

Model metadata:

- `backend/models/HK0011_jessica/LivingRoom_v46_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_v50_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_v51_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_v52_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_v46_two_stage_meta.json`
- `backend/models/HK0011_jessica/LivingRoom_v50_two_stage_meta.json`
- `backend/models/HK0011_jessica/LivingRoom_v51_two_stage_meta.json`
- `backend/models/HK0011_jessica/LivingRoom_v52_two_stage_meta.json`

Machine-readable output:

- `tmp/livingroom_brittleness_forensic_20260311T093026Z/summary.json`
- `tmp/livingroom_brittleness_forensic_20260311T093026Z/date_version_summary.csv`
- `tmp/livingroom_brittleness_forensic_20260311T093026Z/hourly_summary.csv`
- `tmp/livingroom_brittleness_forensic_20260311T093026Z/pair_segment_summary.csv`
- `tmp/livingroom_brittleness_forensic_20260311T093026Z/changed_rows.parquet`

## Finding 1: the fresh family is a different occupied-routing shape, not just a noisier seed

Per-date occupied-rate summary from `date_version_summary.csv`:

- `v46` stays close to or below truth occupied share on most dates.
- `v50` is almost always more occupied than `v46`.
- `v52` is also usually more occupied than `v46`, but not in a uniform way.
- `v51` effectively collapses into "almost always occupied":
  - 2025-12-04 predicted occupied rate `0.9996`
  - 2025-12-07 predicted occupied rate `0.9559`
  - 2025-12-10 predicted occupied rate `0.9996`

Interpretation:

- `v50` and `v52` are not independent winners that happen to fluctuate.
- they belong to the same broader fresh-family geometry
- `v51` shows that the family can tip into a catastrophic occupied collapse

## Finding 2: the Dec 17 win comes from specific occupied blocks that `v46` misses almost completely

Key hourly rows on 2025-12-17 from `hourly_summary.csv`:

- hour `14`
  - truth occupied rate `0.5028`
  - `v46` predicted occupied rate `0.0111`
  - `v52` predicted occupied rate `0.2222`
- hour `15`
  - truth occupied rate `0.2972`
  - `v46` predicted occupied rate `0.0500`
  - `v52` predicted occupied rate `0.1750`
- hour `20`
  - truth occupied rate `0.3833`
  - `v46` predicted occupied rate `0.1222`
  - `v52` predicted occupied rate `0.2722`

Largest `v46 -> v52` changed segments on 2025-12-17 from `pair_segment_summary.csv`:

- `2025-12-17 14:36:50 -> 14:45:10`, `51` rows
  - truth `livingroom_normal_use`
  - `v46=unoccupied`
  - `v52=livingroom_normal_use`
- `2025-12-17 15:47:50 -> 15:52:40`, `30` rows
  - truth `livingroom_normal_use`
  - `v46=unoccupied`
  - `v52=livingroom_normal_use`

Interpretation:

- `v52` wins Dec 17 because it recovers long occupied spans that `v46` routes to `unoccupied`.
- this is a real model behavior change, not a tiny metric wobble.

## Finding 3: the losing dates are not fixed by simply "being less occupied" or "being more occupied"

2025-12-04 is the clearest counterexample.

On the same day, `v52` is too occupied in some blocks and too unoccupied in others:

- hour `05`
  - truth occupied rate `0.0583`
  - `v46` predicted occupied rate `0.1056`
  - `v52` predicted occupied rate `0.2250`
- hour `07`
  - truth occupied rate `0.1833`
  - `v46` predicted occupied rate `0.2972`
  - `v52` predicted occupied rate `0.4833`
- hour `19`
  - truth occupied rate `0.1417`
  - `v46` predicted occupied rate `0.2472`
  - `v52` predicted occupied rate `0.2917`

But on the same date:

- hour `00`
  - truth occupied rate `0.5889`
  - `v46` predicted occupied rate `0.4111`
  - `v52` predicted occupied rate `0.2407`

Largest `v46 -> v52` changed segments on 2025-12-04:

- `2025-12-04 05:22:40 -> 05:29:00`, `39` rows
  - truth `unoccupied`
  - `v46=unoccupied`
  - `v52=livingroom_normal_use`
- `2025-12-04 07:20:30 -> 07:25:00`, `28` rows
  - truth `unoccupied`
  - `v46=unoccupied`
  - `v52=livingroom_normal_use`
- `2025-12-04 00:38:50 -> 00:41:50`, `19` rows
  - truth `livingroom_normal_use`
  - `v46=livingroom_normal_use`
  - `v52=unoccupied`

Interpretation:

- the bad-date behavior is bidirectional within the same day
- `v52` does not just need to be globally "more occupied" or globally "less occupied"
- that weakens the case for a single monotonic threshold fix

## Finding 4: this is upstream routing/calibration behavior, not low-confidence runtime rewriting

Across all pairwise changed rows:

- maximum low-confidence rate on changed rows: `0.0`
- raw top-1 probability on the dominant error rows: `1.0`

Representative error scores:

- `v52` false positives (`unoccupied -> livingroom_normal_use`)
  - acceptance score mean `0.49616157058390087`
  - low-confidence rate `0.0`
- `v52` false negatives (`livingroom_normal_use -> unoccupied`)
  - acceptance score mean `0.9743865695629647`
  - low-confidence rate `0.0`

Interpretation:

- the disagreements are confident binary decisions
- the failure happens before any late abstention / low-confidence rewrite
- this is a routing / calibrated decision-boundary problem upstream of runtime abstention

## Finding 5: the saved metadata already warned that the fresh family was unstable

Saved model-side calibration / routing evidence from `summary.json`:

- `v46`
  - holdout macro-F1 `0.6729`
  - `livingroom_normal_use` threshold `0.5220`
  - `livingroom_normal_use` near-threshold share `0.1998`
  - stage-A occupied threshold `0.8537`
  - stage-A status `target_met+pred_occ_floor`
- `v50`
  - holdout macro-F1 `0.6964`
  - `livingroom_normal_use` threshold `0.5796`
  - `livingroom_normal_use` near-threshold share `0.3905`
  - stage-A occupied threshold `0.8126`
  - stage-A status `target_met+pred_occ_floor`
- `v52`
  - holdout macro-F1 `0.7230`
  - `livingroom_normal_use` threshold `0.4316`
  - `livingroom_normal_use` near-threshold share `0.8485`
  - stage-A occupied threshold `0.95`
  - stage-A status `target_met`
- `v51`
  - holdout macro-F1 `0.6944`
  - `livingroom_normal_use` threshold `0.1961`
  - stage-A occupied threshold `0.00157`
  - stage-A status `fallback_recall_floor`

Interpretation:

- the fresh panel did not converge on one stable occupied-routing geometry
- `v52` already showed a highly unstable `livingroom_normal_use` calibration shape on saved evidence
- `v51` confirms that a passing holdout seed can still be catastrophically unsafe cross-date

## Root-cause conclusion

The dominant mechanism is:

- the fresh LivingRoom family (`v50..v52`) learns a different occupied-routing geometry from the old `v46` anchor
- that geometry helps on Dec 17 because `v46` misses long midday/evening occupied segments
- the same geometry harms Dec 4-9 because it overcalls occupied in some low-occupancy regimes and undercalls occupied in other high-occupancy regimes on the same day
- the disagreement is confident and contiguous, so this is not runtime low-confidence policy noise

This is stronger evidence for:

- regime-sensitive model / routing brittleness

than for:

- a single global threshold issue
- promotion plumbing
- generic seed randomness

## Recommended next step

Do not start with a LivingRoom threshold sweep.

Why:

- the Bedroom case was mostly monotonic occupancy-gate overshoot
- LivingRoom is not monotonic: the bad dates require both "more occupied" and "less occupied" corrections in different time blocks of the same day
- a single threshold or gate knob is unlikely to fix both 2025-12-04 hour `00` and hour `07`

Operational recommendation:

- keep `v46` / `v49` as the LivingRoom safety anchor
- do not promote `v50..v52`
- start a deeper model-shape forensic on representative segments:
  - 2025-12-04 `00:35-00:42`
  - 2025-12-04 `05:22-05:29`
  - 2025-12-04 `07:20-07:25`
  - 2025-12-17 `14:36-14:45`
  - 2025-12-17 `15:47-15:52`

The next forensic should inspect raw sensor traces / derived features / stage-A occupied scores on those exact segments to explain why the fresh family flips entire blocks in opposite directions across dates.
