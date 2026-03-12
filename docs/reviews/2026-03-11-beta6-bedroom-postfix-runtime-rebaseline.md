# Beta6 Bedroom Post-Fix Runtime Rebaseline

## Scope

- Rebaseline the canonical live Bedroom champion and the Bedroom `v40` candidate after the permanent two-stage runtime metadata fix.
- Compare both on the corrected Dec 17 Jessica workbook using the repaired runtime semantics.
- Decide whether any Bedroom candidate path remains worth carrying forward.

## Inputs

Replay input:

- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`

Namespaces:

- live champion: `backend/models/HK0011_jessica`
- repaired candidate: `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z`

Post-fix artifacts:

- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/comparison.json`
- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/live_v38_postfix/summary.json`
- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/candidate_v40_postfix/summary.json`
- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/live_v38_load_check.json`
- `tmp/bedroom_postfix_rebaseline_20260311T175000Z/candidate_v40_load_check.json`

## Runtime state

Both namespaces now resolve Bedroom to single-stage fallback under the repaired runtime rules:

- live `v38`: `bedroom_two_stage_loaded = false`
- candidate `v40`: `bedroom_two_stage_loaded = false`
- both versioned Bedroom two-stage metadata files now say:
  - `runtime_enabled = false`
  - `runtime_gate_source = single_stage_fallback_no_regress`

Interpretation:

- the previous Bedroom candidate replay was evaluated on an invalid runtime path
- after the root fix, neither the live Bedroom champion nor the candidate is allowed to activate two-stage runtime

## Dec 17 result

### Live `Bedroom_v38` after the root fix

- final accuracy: `0.8885342789598109`
- final macro-F1: `0.4929066675409425`
- raw top-1 accuracy: `0.8905437352245863`
- raw top-1 macro-F1: `0.6054206328955019`
- predicted occupied rate: `0.4951536643026005`
- true `sleep -> unoccupied`: `117`
- true `unoccupied -> bedroom_normal_use`: `435`
- true `bedroom_normal_use -> unoccupied`: `132`

### Candidate `Bedroom_v40` after the root fix

- final accuracy: `0.6024822695035461`
- final macro-F1: `0.33052062399815013`
- raw top-1 accuracy: `0.6224586288416075`
- raw top-1 macro-F1: `0.4201960949332568`
- predicted occupied rate: `0.5239952718676123`
- true `sleep -> unoccupied`: `832`
- true `unoccupied -> bedroom_normal_use`: `1734`
- true `bedroom_normal_use -> unoccupied`: `233`

### Candidate minus live delta

- final macro-F1: `-0.16238604354279235`
- final accuracy: `-0.28605200945626474`
- raw top-1 macro-F1: `-0.18522453796224503`
- true `sleep -> unoccupied`: `+715`
- true `unoccupied -> bedroom_normal_use`: `+1299`
- true `bedroom_normal_use -> unoccupied`: `+101`

## Conclusion

Post-fix, the Bedroom `v40` candidate is not just non-promotable; it is materially worse than the live champion.

The corrected runtime conclusion is:

- live `Bedroom_v38` is the current valid Dec 17 winner
- the previous `v40` replay and threshold-sweep conclusions cannot be used as promotion evidence
- the old Bedroom candidate branch should be treated as a contaminated forensic path created before the runtime metadata bug was fixed

## Recommendation

Push the permanent runtime metadata fix to GitHub.

Do **not** push any Bedroom candidate promotion decision derived from the old `v40` replay.

Next Bedroom move:

- start a fresh Bedroom investigation from the repaired runtime baseline
- use live `v38` post-fix behavior as the canonical benchmark
- discard the old `v40` branch as a promotion candidate
- if Bedroom work continues, launch a new model/data experiment from scratch rather than iterating on the contaminated `v40` artifacts
