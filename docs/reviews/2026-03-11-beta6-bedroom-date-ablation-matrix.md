# Beta6 Bedroom Date-Ablation Matrix

## Scope

- Task 3 remains Bedroom-first.
- The tightened checkpoint now uses two artifact sets:
  - fresh rerun for the overlap variants needed to close the review gap:
    - `anchor`
    - `add_2025-12-04`
    - `add_2025-12-05`
    - `cumulative_through_2025-12-05`
  - the earlier single-day matrix for later-day context:
    - `add_2025-12-06`
    - `add_2025-12-07`
    - `add_2025-12-08`
    - `add_2025-12-09`
- The fresh tighten manifest is authoritative for the overlap variants because it uses the corrected machine-readable summary semantics and closes the cumulative-through-`2025-12-05` gap.

## Artifact Paths

- tightened manifest:
  - `tmp/bedroom_root_cause_matrix_tighten_20260311T153500Z/manifest.json`
- earlier single-day manifest:
  - `tmp/bedroom_root_cause_matrix_singles_20260311T135112Z/manifest.json`

## Reconciliation Notes

- The runner summary is now truthful:
  - top-level `status` reflects experiment outcome, not just execution success
  - top-level `execution_status` remains separate
  - `holdout_bedroom_normal_use_recall` now comes from the saved checkpoint-selection metrics
- PostgreSQL-backed historical corrections were unavailable in every replay here because local auth failed for `postgres`. No matrix cell received those corrections, so the relative comparisons remain aligned.
- The fresh rerun produced different absolute scores from the earlier single-day matrix for the overlap variants. That is now an explicit artifact fact, not something to hide. The important ordering held:
  - `add_2025-12-04` stayed beneficial over anchor
  - `add_2025-12-05` stayed harmful versus anchor
  - `cumulative_through_2025-12-05` was also harmful

## Fresh Tighten Results

Label order in the count columns is `bedroom_normal_use / sleep / unoccupied`.

| Variant | Holdout macro-F1 | Holdout BNU recall | Gate | Dec17 final macro-F1 | Sleep->Unocc | Unocc->BNU | BNU->Unocc | Runtime two-stage |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| anchor | 0.626 | 0.514 | PASS | 0.744 | 111 | 1013 | 119 | False |
| add_2025-12-04 | 0.661 | 0.582 | PASS | 0.849 | 293 | 398 | 51 | False |
| add_2025-12-05 | 0.670 | 0.499 | PASS | 0.723 | 292 | 826 | 246 | True |
| cumulative_through_2025-12-05 | 0.636 | 0.951 | PASS | 0.643 | 418 | 1320 | 9 | False |

## Later Single-Day Context

The earlier single-day matrix still matters for the later dates that were not rerun in the tighten pass:

- `add_2025-12-06` remained harmful and failed the Bedroom gate.
- `add_2025-12-07` remained harmful and failed the Bedroom gate.
- `add_2025-12-08` remained a recovery counterexample above the old anchor.
- `add_2025-12-09` remained harmful and was the only earlier single-day variant that flipped to two-stage runtime use.

Those later-day cells are still useful context, but the checkpoint conclusion below is anchored on the fresh tighten artifact for the earliest harmful target.

## Tightened Task 3 Conclusion

- The earliest harmful single-day add-back is **`2025-12-05`**.
- `add_2025-12-05` is still worse than the fresh anchor on Dec 17 replay:
  - `0.723 < 0.744`
- `cumulative_through_2025-12-05` is also harmful on Dec 17 replay:
  - `0.643 < 0.744`

What the executed evidence now proves:

- harm is already present when `2025-12-05` is added by itself
- a cumulative interaction is therefore **not required** to make the break visible
- adding `2025-12-04` back on top of `2025-12-05` does not rescue the regime; the cumulative set through `2025-12-05` is still harmful

What the executed evidence does **not** prove:

- it does not prove that later cumulative additions are irrelevant
- it does not prove that the harmful mechanism is purely one isolated slice with no interaction effects

Decision-grade checkpoint statement:

> `2025-12-05` remains the earliest validated harmful Bedroom target. The minimum cumulative control through `2025-12-05` is also harmful, so Task 4 should audit `2025-12-05` first, while treating cumulative amplification as a residual possibility rather than the primary target.

## Task 4 Carry-Forward

- harmful target:
  - `2025-12-05`
- bounded good references:
  - `2025-12-04`
  - `2025-12-10`
- useful later control:
  - `2025-12-08` from the earlier single-day matrix
