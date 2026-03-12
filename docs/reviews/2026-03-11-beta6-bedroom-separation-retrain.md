# Beta6 Bedroom Separation Retrain Review

## Scope

- Finish the in-flight Bedroom-only oversampling-rollback retrain.
- Activate the saved Bedroom version inside the candidate namespace only.
- Replay the corrected Dec 17 Jessica workbook on the same branch/runtime harness used by the canonical Bedroom benchmark.

## Retrain result

Run artifacts:

- `tmp/jessica_bedroom_sepfix_20260311T041856Z/status.json`
- `tmp/jessica_bedroom_sepfix_20260311T041856Z/train_metrics.json`
- `tmp/jessica_bedroom_sepfix_20260311T041856Z/exit_code_combined.txt`

Candidate namespace:

- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z`

Final run state:

- status: `ok`
- exit code: `0`
- saved Bedroom version: `v40`
- gate pass: `true`
- holdout macro-F1: `0.6394460946743338`

This confirms the combined-pack Bedroom-only retrain completed successfully and produced a new saved candidate Bedroom version.

## Candidate activation

Activation artifact:

- `tmp/jessica_17dec_eval_candidate_bedroom_sepfix_20260311T041155Z/load_sanity.json`

What was activated:

- candidate namespace: `HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z`
- room: `Bedroom`
- version before activation: `38`
- version after activation: `40`

Verified loaded room versions after activation:

- `Bathroom=35`
- `Bedroom=40`
- `Entrance=26`
- `Kitchen=27`
- `LivingRoom=52`

This means the replay below is evaluating the new Bedroom model while leaving the rest of that candidate namespace untouched.

## Dec 17 replay

Replay artifacts:

- `tmp/jessica_17dec_eval_candidate_bedroom_sepfix_20260311T041155Z/final/raw_predictions/summary.json`
- `tmp/jessica_17dec_eval_candidate_bedroom_sepfix_20260311T041155Z/final/comparison/summary.json`
- `tmp/jessica_17dec_eval_candidate_bedroom_sepfix_20260311T041155Z/final/comparison/Bedroom_merged.parquet`

Canonical Bedroom comparator:

- `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/summary.json`
- `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/Bedroom_merged.parquet`

### Bedroom result vs canonical `Bedroom_v38`

Candidate `Bedroom_v40` on Dec 17:

- final accuracy: `0.6995271867612293`
- final macro-F1: `0.3931771489209137`
- raw top-1 macro-F1: `0.3931771489209137`
- raw export macro-F1: `0.3931771489209137`
- low-confidence rewrite count from raw top-1: `0`

Canonical Bedroom reference:

- final accuracy: `0.6345153664302601`
- final macro-F1: `0.3511152121985108`
- raw top-1 macro-F1: `0.43865005859309375`
- raw export macro-F1: `0.35107096454067005`
- low-confidence rewrite count from raw top-1: `16`

Bedroom deltas:

- final accuracy: `+0.0650118203309692`
- final macro-F1: `+0.04206193672240294`

Interpretation:

- the retrain improved the benchmark Bedroom result beyond the `0.3511` canonical reference
- the new Bedroom output no longer loses performance to runtime low-confidence rewriting
- the improvement is model-side, not runtime-policy-side, because final/raw-top1/raw-export all collapse to the same value on `v40`

## Error-shift analysis

Canonical dominant Bedroom error:

- true `unoccupied` -> predicted `bedroom_normal_use`: `2849`

Candidate dominant Bedroom errors:

- true `sleep` -> predicted `unoccupied`: `1468`
- true `bedroom_normal_use` -> predicted `unoccupied`: `548`
- true `unoccupied` -> predicted `bedroom_normal_use`: `470`

Key targeted error delta:

- true `unoccupied` -> predicted `bedroom_normal_use`: `2849 -> 470`
- absolute change: `-2379`

Prediction distribution shift:

- canonical predictions: `bedroom_normal_use=3558`, `sleep=3383`, `unoccupied=1508`, `low_confidence=11`
- candidate predictions: `bedroom_normal_use=527`, `sleep=1969`, `unoccupied=5964`

Interpretation:

- the oversampling rollback directly addressed the original failure mode
- the old `unoccupied -> bedroom_normal_use` confusion dropped sharply
- the retrain over-corrected toward `unoccupied`, creating a new large `sleep -> unoccupied` failure band

## Recommendation

This run is a real forensic gain, but not a clean promotion candidate yet.

Why:

- it beats the canonical Bedroom benchmark on Dec 17
- it removes the dominant `unoccupied -> bedroom_normal_use` error that motivated the experiment
- it replaces that failure with a different large model-side error (`sleep -> unoccupied`)
- runtime policy is no longer the limiting factor on this replay; the remaining problem is still model-side class separation

Recommended next step:

- do not reopen LivingRoom work
- do not spend time on runtime threshold tuning first
- keep this evidence as proof that Bedroom class-0 oversampling pressure was materially harming occupancy separation
- move to the next Bedroom-only model/data experiment aimed at reducing the new `sleep -> unoccupied` overcorrection while preserving the big drop in `unoccupied -> bedroom_normal_use`
