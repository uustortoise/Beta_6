# Beta6 Bedroom Policy Ablation

## Scope

- Run the recommended Bedroom-only offline policy ablation on the canonical Dec 17 replay.
- Use saved `Bedroom_v38` replay outputs only.
- Decide whether runtime policy alone can recover meaningful Bedroom performance without a retrain.

## Canonical inputs

- canonical benchmark summary:
  - `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/summary.json`
- canonical Bedroom comparison parquet:
  - `tmp/jessica_17dec_eval_live_livingroom_rootfix_registryfix_20260311/final/comparison/Bedroom_merged.parquet`
- ablation output:
  - `tmp/bedroom_dec17_policy_ablation_20260311/summary.json`

Bedroom canonical reference:

- final benchmark macro-F1: `0.3511152121985108`
- final three-label macro-F1: `0.5851920203308513`

The second number is included because the benchmark macro-F1 counts `low_confidence` as a fourth predicted label when it appears.

## Variants tested

1. `raw_top1_direct`
- export `predicted_top1_label_raw` directly

2. `raw_export`
- export `predicted_activity_raw`

3. `final_export`
- export `predicted_activity`

4. `no_low_confidence`
- start from `final_export`
- replace `low_confidence` with the corresponding raw top-1 label

5. Bedroom-only threshold sweep
- vary only the saved `bedroom_normal_use` low-confidence threshold in the recorded acceptance-score space
- keep the saved weights and saved raw predictions fixed

## Results

### Main variants

| Variant | Rows changed vs final | Low-confidence rows | Benchmark macro-F1 | Three-label macro-F1 |
|---|---:|---:|---:|---:|
| `raw_top1_direct` | 16 | 0 | `0.43865005859309375` | `0.5848667447907917` |
| `raw_export` | 5 | 11 | `0.35107096454067005` | `0.5851182742344501` |
| `final_export` | 0 | 11 | `0.3511152121985108` | `0.5851920203308513` |
| `no_low_confidence` | 11 | 0 | `0.43870527762188205` | `0.5849403701625094` |

Interpretation:

- disabling `low_confidence` produces a large jump in benchmark macro-F1 (`0.3511 -> 0.4387`)
- but the substantive three-label Bedroom macro-F1 does not improve; it is slightly worse than the current final export
- therefore the benchmark gain is mostly accounting behavior from removing a fourth predicted label, not a real recovery of Bedroom activity separation

### Threshold sweep

Best sweep point for benchmark macro-F1:

- `bedroom_normal_use_threshold=0.0`
- benchmark macro-F1 `0.43868732182851156`
- three-label macro-F1 `0.5849164291046821`

Best sweep point for three-label macro-F1:

- `bedroom_normal_use_threshold=0.19`
- benchmark macro-F1 `0.35116942716513755`
- three-label macro-F1 `0.5852823786085626`

Interpretation:

- the sweep reproduces the same conclusion
- benchmark macro-F1 is maximized by effectively turning off low-confidence for Bedroom
- three-label activity quality changes only in the fourth decimal place
- threshold tuning in the saved acceptance-score space has negligible substantive leverage

## Error pattern after ablation

The dominant error survives all policy-only variants:

- true `unoccupied` predicted as `bedroom_normal_use`

Counts:

- `final_export`: `2849`
- `raw_top1_direct`: `2858`
- `no_low_confidence`: `2858`

Secondary errors remain much smaller:

- true `sleep` predicted as `bedroom_normal_use`
  - `111..114` rows depending on variant
- true `unoccupied` predicted as `sleep`
  - `101` rows in every major variant

Interpretation:

- policy-only changes do not meaningfully reduce the Bedroom over-active frontier
- the core failure remains occupied-vs-unoccupied separation, not low-confidence routing

## Recommendation

Do not spend more time on Bedroom runtime-policy tuning as the primary path.

Recommended next experiment:

- run one Bedroom-only retrain experiment focused on occupied-vs-unoccupied separation
- keep it narrow:
  - Bedroom only
  - corrected Jessica pack
  - no all-room rerun
  - compare against the current `Bedroom_v38` Dec 17 canonical replay on the same benchmark harness

The ablation result is sufficient to move the thread forward:

- runtime policy can change the benchmark artifact
- runtime policy does not fix the underlying Bedroom classification geometry
- the next signal now has to come from model-side changes
