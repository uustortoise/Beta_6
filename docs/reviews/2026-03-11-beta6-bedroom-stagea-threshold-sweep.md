# Beta6 Bedroom Stage-A Threshold Sweep

## Scope

- Test the `Bedroom_v40` root-cause hypothesis without another retrain.
- Hold Bedroom weights fixed and sweep only the Bedroom two-stage stage-A occupied threshold on the corrected Dec 17 Jessica replay.
- Select one threshold to carry forward as the next candidate calibration setting.

## Inputs

Source forensic candidate:

- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z`

Isolated sweep namespace:

- `backend/models/HK0011_jessica_candidate_bedroom_stagea_sweep_20260311T073724Z`

Sweep driver:

- `tmp/bedroom_stagea_threshold_sweep.py`

Replay input:

- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`

Machine-readable sweep outputs:

- `tmp/bedroom_stagea_sweep_20260311T073724Z_full/sweep_summary.json`
- `tmp/bedroom_stagea_sweep_20260311T073724Z_full/sweep_summary.csv`

Selected-threshold outputs:

- `tmp/bedroom_stagea_sweep_20260311T073724Z_selected/threshold_0p150000/summary.json`
- `tmp/bedroom_stagea_sweep_20260311T073724Z_selected/selected_final/comparison/summary.json`
- `tmp/bedroom_stagea_sweep_20260311T073724Z_selected/selected_final/comparison/Bedroom_merged.parquet`
- `tmp/bedroom_stagea_sweep_20260311T073724Z_selected/load_sanity.json`

## Sweep design

Thresholds tested:

- `0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01, 0.0027200691401958466`

Control points from the prior forensic:

- `v38` stage-A threshold: `0.0027200691401958466`
- `v40` stage-A threshold: `0.95`
- Dec 17 truth occupied rate: `0.47210401891252957`

All sweep runs kept Bedroom weights fixed at `v40` and only rewrote the stage-A threshold metadata before replay.

## Result

Top thresholds by final Bedroom macro-F1:

- `0.15`: macro-F1 `0.42453576648530544`, accuracy `0.6637115839243499`, predicted occupied rate `0.45614657210401893`
- `0.20`: macro-F1 `0.4234926990126331`, accuracy `0.6671394799054373`, predicted occupied rate `0.4380614657210402`
- `0.05`: macro-F1 `0.4230059498566658`, accuracy `0.6416075650118203`, predicted occupied rate `0.5281323877068558`
- `0.10`: macro-F1 `0.4225136912758893`, accuracy `0.6544917257683215`, predicted occupied rate `0.4795513002364066`
- `0.25`: macro-F1 `0.4208302673292515`, accuracy `0.6670212765957447`, predicted occupied rate `0.4222222222222222`

Selected threshold:

- `0.15`

Why `0.15` was selected:

- It is the best Dec 17 macro-F1 point in the full sweep.
- Its predicted occupied rate (`0.45614657210401893`) is closest to the Dec 17 truth occupied rate (`0.47210401891252957`) among the top-performing thresholds.
- It materially recovers the `sleep` and `bedroom_normal_use` false-unoccupied collapse introduced by `0.95` while staying much better than `v38` on the old `unoccupied -> bedroom_normal_use` failure.

## Error tradeoff at `0.15`

Compared with the current `v40` threshold `0.95`:

- final macro-F1: `0.3931771489209137 -> 0.42453576648530544` (`+0.03135861756439173`)
- final accuracy: `0.6995271867612293 -> 0.6637115839243499` (`-0.035815602836879346`)
- true `sleep -> unoccupied`: `1468 -> 1104` (`-364`)
- true `bedroom_normal_use -> unoccupied`: `548 -> 373` (`-175`)
- true `unoccupied -> bedroom_normal_use`: `470 -> 1291` (`+821`)
- predicted occupied rate: `0.2950354609929078 -> 0.45614657210401893`

Compared with the old `v38`-like permissive gate `0.0027200691401958466`:

- final macro-F1: `0.35701727312970505 -> 0.42453576648530544` (`+0.06751849335560039`)
- true `sleep -> unoccupied`: `99 -> 1104`
- true `unoccupied -> bedroom_normal_use`: `3453 -> 1291`

Three-label class view at `0.15`:

- `unoccupied` F1: `0.6892817896389326`
- `sleep` F1: `0.7948756388415672`
- `bedroom_normal_use` F1: `0.21518412590871015`

This keeps the core forensic conclusion intact: the dominant Bedroom failure is stage-A routing balance, not runtime low-confidence rewriting.

## Near-tie alternative

`0.20` remains the best conservative fallback:

- macro-F1 `0.4234926990126331`
- accuracy `0.6671394799054373`
- true `sleep -> unoccupied`: `1155`
- true `bedroom_normal_use -> unoccupied`: `387`
- true `unoccupied -> bedroom_normal_use`: `1204`

Interpretation:

- `0.20` gives up a small amount of macro-F1 versus `0.15`
- it is slightly more conservative on the original `unoccupied -> bedroom_normal_use` regression
- the difference is small enough that `0.20` is a valid fallback if later dates show `0.15` to be too aggressive

## Selected candidate state

The isolated sweep namespace is now pinned to the selected threshold:

- namespace: `backend/models/HK0011_jessica_candidate_bedroom_stagea_sweep_20260311T073724Z`
- `Bedroom current_version`: `40`
- `Bedroom stage_a_occupied_threshold`: `0.15`

Load sanity:

- loaded rooms: `Bathroom, Bedroom, Entrance, Kitchen, LivingRoom`
- versions: `Bathroom=35`, `Bedroom=40`, `Entrance=26`, `Kitchen=27`, `LivingRoom=52`

## Recommendation

Do not start another Bedroom retrain yet.

The sweep confirms that most of the `v40` regression is calibration/routing, not representation. The next move should be:

- keep this namespace at `0.15`
- replay it on the broader corrected-date panel before any promotion decision
- use `0.20` as the fallback threshold if cross-date validation shows `0.15` is too permissive

If the cross-date panel stays stable, this can be treated as a calibration fix candidate. If it breaks on nearby dates, the next retrain should constrain stage-A occupancy calibration around the `0.15-0.20` region instead of allowing a jump back to `0.95`.
