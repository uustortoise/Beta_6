# Beta6 Jessica Pre-Final Sign-Off

## Decision

Pre-final status for Jessica is `GO`.

This is a controlled `GO`, not a claim that every room is fully solved.

Candidate under sign-off:

- `HK0011_jessica_candidate_supportfix_20260310T2312Z`

Fresh replay artifact:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_17dec_eval_candidate_supportfix_prefinal_20260312T0935Z/final/comparison/summary.json`

Fresh load sanity artifacts:

- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_prefinal_candidate_supportfix_20260312T_load_sanity/load_sanity.json`
- `/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/jessica-pre-final-testing/tmp/jessica_prefinal_candidate_supportfix_20260312T_load_sanity_recheck/load_sanity.json`

## Why This Is A GO

No room is `block`.

Room status:

- `Bathroom`: `pass`
- `Bedroom`: `conditional`
- `Kitchen`: `pass`
- `LivingRoom`: `pass`

Fresh overall replay result:

- accuracy `0.8417598841689825`
- macro-F1 `0.4663352546942085`

Fresh room-level results:

- `Bathroom`: accuracy `0.8159544821179749`, macro-F1 `0.42003535353043925`
- `Bedroom`: accuracy `0.8216312056737589`, macro-F1 `0.4449128824184004`
- `Kitchen`: accuracy `0.8856643356643357`, macro-F1 `0.40799361200550766`
- `LivingRoom`: accuracy `0.8747952258366487`, macro-F1 `0.43397967577731184`

Interpretation:

- `Bathroom`, `Kitchen`, and `LivingRoom` matched the accepted frontier and showed no new regression.
- `Bedroom` did not show a new runtime collapse on the fresh replay.
- The integrated candidate remains suitable for pre-final progression under the current release constraints.

## Bedroom Condition

`Bedroom` remains `conditional`, not `pass`.

Why:

- the structural `unoccupied <-> bedroom_normal_use` instability family is still large
- Bedroom is still fail-closed rather than fully robust across valid alternate regimes
- Bedroom runtime is intentionally single-stage fallback in the accepted candidate path

Fresh Bedroom evidence:

- final macro-F1 `0.4449128824184004`
- low-confidence rate `0.01773049645390071`
- `unoccupied -> bedroom_normal_use = 1017`
- `sleep -> unoccupied = 124`
- `bedroom_normal_use -> unoccupied = 119`
- rewrites `171`, mostly `bedroom_normal_use -> low_confidence = 144`

This is acceptable for pre-final sign-off because it did not introduce a new collapse and the existing safety posture remains intact.

It is not evidence that Bedroom is fully solved.

## Runtime Contract Note

The first fresh load self-healed alias drift, and the recheck remained stable.

Current accepted runtime contract:

- `platform.two_stage_core_models` contained `Bathroom` and `LivingRoom`
- `Bedroom` was intentionally absent because `Bedroom_v38_two_stage_meta.json` sets `runtime_enabled=false`

That absence should be treated as expected repaired behavior, not as a pre-final failure.

## Required Release Notes

Carry these points forward into any final release summary:

1. `Bedroom_v38` stays frozen as the operational Bedroom choice for this release path.
2. No new Bedroom retrains should be mixed into this release line.
3. Bedroom remains fail-closed and conditionally accepted, not fully robust.
4. The no-historical-corrections caveat remains because local PostgreSQL validation was unavailable during the Bedroom root-fix work.
5. The first-load alias self-heal should be recorded as a runtime note and considered for a small cleanup follow-up.

## Bottom Line

Jessica pre-final status is `GO` with one explicit condition:

> ship only with Bedroom clearly marked as a bounded-risk, fail-closed room under the current `v38` operational anchor.

That is good enough to continue the release path.
