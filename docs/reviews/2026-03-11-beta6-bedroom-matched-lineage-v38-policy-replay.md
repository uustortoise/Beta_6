# Beta6 Bedroom Matched-Lineage Replay With v38-Style Sampling

## Scope

- Test the strongest remaining causal question from the upstream-lineage note.
- Hold the Bedroom sampling posture close to `v38`, but train from the full corrected `v40` source pack.
- Check whether the larger Dec 4-10 + Dec 17 source lineage alone is enough to leave the old `v38` success regime.

## Experiment

Candidate namespace:

- `HK0011_jessica_candidate_bedroom_lineagev38_20260311T113038Z`

Artifacts created for this replay:

- `tmp/jessica_bedroom_lineagev38_20260311T113038Z/build_combined_corrected_pack.py`
- `tmp/jessica_bedroom_lineagev38_20260311T113038Z/combined_corrected_pack.parquet`
- `tmp/jessica_bedroom_lineagev38_20260311T113038Z/run_bedroom_lineagev38_retrain.py`
- `tmp/jessica_bedroom_lineagev38_20260311T113038Z/activate_latest_bedroom.py`
- `tmp/jessica_bedroom_lineagev38_20260311T113038Z/train_metrics.json`
- `tmp/jessica_bedroom_lineagev38_20260311T113038Z/load_sanity.json`
- `tmp/jessica_17dec_eval_candidate_bedroom_lineagev38_20260311T113038Z/final/comparison/summary.json`

Historical controls used for comparison:

- live `Bedroom_v38`
- candidate `Bedroom_v40` from the support-fix separation rerun

## Constraint

This is not a bit-identical recreation of the historical `v38` global policy hash.

It is still a strong matched-lineage test because the Bedroom-specific knobs that matter for the prior hypothesis match the old `v38` posture and differ materially from `v40`:

| Run | Source lineage | transition-focus max multiplier | minority max multiplier |
| --- | --- | --- | --- |
| live `v38` | Dec 10 + Dec 17 | `3` | `6` |
| candidate `v40` | Dec 4-10 + Dec 17 | `1` / effectively disabled | `1` |
| matched-lineage replay `v39` | Dec 4-10 + Dec 17 | `3` | `6` |

Common Bedroom downsample settings remained:

- `min_share=0.6`
- `stride=2`
- `boundary_keep=6`
- `min_run_length=24`

## Finding 1: the full source pack still fails under the v38-style Bedroom posture

The replay trained and saved `Bedroom_v39`, but it did not recover the old holdout regime.

Key holdout result from `train_metrics.json` / `Bedroom_v39_decision_trace.json`:

- saved version: `39`
- policy hash: `8e24355aa6eb14636b14adea6fdb13dabcc8dfa801355a1c5d2bb33ed54a54e2`
- gate pass: `false`
- best holdout macro-F1: `0.5685497591261085`
- no-regress floor: `0.5757134795187449`
- critical recall failures:
  - `bedroom_normal_use`: `0.1053`
  - `sleep`: `0.6196`
  - `unoccupied`: `0.9598`

So even after restoring the old Bedroom transition-focus / minority expansion posture, the model still misses the holdout gate because `bedroom_normal_use` recall collapses.

## Finding 2: the replay preserves the full-pack class regime before and after sampling

Pre-sampling support for the replay exactly matches the full corrected eight-file pack:

| Run | `bedroom_normal_use` | `sleep` | `unoccupied` |
| --- | --- | --- | --- |
| live `v38` pre-sampling | `1288` | `5808` | `7159` |
| candidate `v40` pre-sampling | `8954` | `17437` | `28951` |
| matched-lineage replay `v39` pre-sampling | `8954` | `17437` | `28951` |

The restored `v38`-style Bedroom sampling changes the post-sampling balance, but not enough to recover the earlier regime:

| Stage | `bedroom_normal_use` | `sleep` | `unoccupied` |
| --- | --- | --- | --- |
| replay pre-sampling | `8954` | `17437` | `28951` |
| replay post transition-focus | `13708` | `17525` | `37059` |
| replay post minority sampling | `17073` | `17525` | `37059` |

This confirms the experiment is actually testing the intended question:

- same large source pack as `v40`
- old Bedroom sampling posture restored
- different outcome from `v40`, but still not the old `v38` behavior

## Finding 3: Dec 17 replay does not return to the old v38 success mode

Bedroom Dec 17 replay summary:

| Model | Bedroom final accuracy | Bedroom final macro-F1 | raw top-1 macro-F1 | rewrites |
| --- | --- | --- | --- | --- |
| live `v38` | `0.6345` | `0.3511` | `0.4387` | `16` |
| candidate `v40` | `0.6995` | `0.3932` | `0.3932` | `0` |
| matched-lineage replay `v39` | `0.6095` | `0.3205` | `0.4007` | `32` |

Predicted Bedroom class shares on Dec 17:

| Model | `bedroom_normal_use` | `sleep` | `unoccupied` |
| --- | --- | --- | --- |
| live `v38` | `42.06%` | `39.99%` | `17.83%` |
| candidate `v40` | `6.23%` | `23.27%` | `70.50%` |
| matched-lineage replay `v39` | `20.63%` | `22.17%` | `57.06%` |

The replay lands between `v38` and `v40`, but it clearly does **not** recover the old `v38` mode.

## Finding 4: the failure family still looks like the full-pack regime, not the old v38 regime

Most informative Bedroom confusion counts:

| Model | Dominant signal |
| --- | --- |
| live `v38` | `unoccupied -> bedroom_normal_use = 2849` and `sleep -> sleep = 3277` |
| candidate `v40` | `sleep -> unoccupied = 1468` and `bedroom_normal_use -> unoccupied = 548` |
| matched-lineage replay `v39` | `sleep -> unoccupied = 1496`, `unoccupied -> bedroom_normal_use = 1381`, `bedroom_normal_use -> unoccupied = 303` |

Interpretation:

- The replay no longer shows the extreme `v38` over-promotion of `unoccupied -> bedroom_normal_use`.
- But it still keeps the `sleep -> unoccupied` collapse family that appears in `v40`.
- Restoring the `v38` Bedroom sampling posture changes the trade-off shape, but it does not restore the old source-pair behavior.

## Conclusion

This replay is strong evidence that the **expanded corrected source lineage is sufficient to break the old `v38` Bedroom regime**.

The later `v40` Bedroom rollback policy still matters, but it now looks secondary:

- the large Dec 4-10 + Dec 17 pack appears to create the main regime shift
- the later `v40` rollback mainly changes how that failure is distributed between `bedroom_normal_use`, `sleep`, and `unoccupied`

In other words:

- source lineage is the main driver
- downstream policy is a shape modifier, not the earliest cause

## Recommended Next Step

If one more causal isolation run is needed, use the inverse control:

- apply the `v40` rollback posture to the small `v38` source pair (`10dec + 17dec`)

That would quantify how much damage the downstream rollback can do when the source lineage is held in the old regime.

At this point, broad additional archaeology is low value. The investigation is already narrow enough to shift from discovery to policy decision.
