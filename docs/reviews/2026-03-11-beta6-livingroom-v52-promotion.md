# Beta6 LivingRoom v52 Promotion Review

## Scope

- Promote `LivingRoom_v52` from `HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z` into the forensic `HK0011_jessica` target namespace.
- Verify that the target namespace preserves rollback history and reproduces the authoritative candidate Dec 17 replay.

## Promotion summary

Promotion artifact:

- `tmp/jessica_live_livingroom_v52_promotion_20260311.json`

Executed promotion path:

- helper: `backend/scripts/promote_room_versions_from_namespace.py`
- source namespace: `backend/models/HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z`
- target namespace: `backend/models/HK0011_jessica`
- promoted room/version: `LivingRoom_v52`

Result:

- target `LivingRoom_versions.json.current_version=52`
- imported candidate-era versions `49, 50, 51, 52`
- reused existing `v46`
- preserved older live rollback history:
  - `46, 45, 44, 43, 42, 41, 40, 39, 28, 27, 26, 25, 24`

Latest alias materialization confirmed:

- `LivingRoom_model.keras`
- `LivingRoom_label_encoder.pkl`
- `LivingRoom_scaler.pkl`
- `LivingRoom_thresholds.json`
- `LivingRoom_decision_trace.json`
- `LivingRoom_two_stage_meta.json`
- `LivingRoom_two_stage_stage_a_model.keras`

## Fresh-load sanity

Load sanity artifact:

- `tmp/jessica_live_livingroom_v52_load_sanity_20260311.json`

Verified runtime state:

- loaded rooms:
  - `Bathroom`
  - `Bedroom`
  - `Entrance`
  - `Kitchen`
  - `LivingRoom`
- room versions:
  - `Bathroom=35`
  - `Bedroom=38`
  - `Entrance=26`
  - `Kitchen=27`
  - `LivingRoom=52`
- two-stage core models:
  - `Bathroom`
  - `Bedroom`
  - `LivingRoom`

## Dec 17 replay confirmation

Promoted target replay artifact:

- `tmp/jessica_17dec_eval_live_livingroom_v52_20260311/final/comparison/summary.json`

Authoritative candidate comparator:

- `tmp/jessica_17dec_eval_candidate_livingroom_fresh_20260311T023304Z/final_v52/comparison/summary.json`

Baseline live comparator:

- `tmp/jessica_17dec_eval_live_recheck_20260311T023304Z/final/comparison/summary.json`

### Promoted target result

- overall accuracy `0.8410592933373812`
- overall macro-F1 `0.47093447750938316`
- LivingRoom accuracy `0.9235899836180669`
- LivingRoom macro-F1 `0.5004234019159393`
- Bedroom macro-F1 `0.3900200536147847`

### Comparison to candidate `final_v52`

- all per-room replay metrics matched exactly
- overall accuracy matched exactly
- overall macro-F1 differed only at floating-point noise:
  - candidate `0.47093447750938305`
  - promoted target `0.47093447750938316`

Interpretation:

- the room-wise promotion reproduced the authoritative candidate result
- the `LivingRoom_v52` improvement survives promotion into the target namespace

### Comparison to pre-promotion live `v46`

- overall accuracy `+0.00037364844352072435`
- overall macro-F1 `+0.01419238760676117`
- LivingRoom accuracy `+0.0018722209220687347`
- LivingRoom macro-F1 `+0.047463337094571456`
- Bedroom macro-F1 `+0.0`

Interpretation:

- the gain remains isolated to `LivingRoom`
- `Bedroom` stayed unchanged on the same replay harness

## Residual caveat

The promoted JSON artifacts retain the candidate namespace in their embedded metadata:

- `backend/models/HK0011_jessica/LivingRoom_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_two_stage_meta.json`
- `backend/models/HK0011_jessica/LivingRoom_v52_decision_trace.json`

Each still reports:

- `elder_id=HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z`

Interpretation:

- runtime loading and replay behavior were unaffected
- this is still worth tracking as metadata drift if downstream tooling expects embedded `elder_id` to match the installation namespace exactly

## Conclusion

The controlled room-wise promotion succeeded locally:

- `HK0011_jessica/LivingRoom` is now on `v52`
- rollback history was preserved
- the promoted target replay reproduced the authoritative candidate `v52` result

Promotion readiness assessment:

- strong enough for evidence-preserving handoff
- metadata rewrite is the only residual caveat surfaced during local validation
