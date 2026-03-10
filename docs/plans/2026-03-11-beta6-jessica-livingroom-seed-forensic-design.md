# Beta6 Jessica LivingRoom Seed Forensic Design

**Date:** 2026-03-11

**Goal:** Explain, with reproducible evidence, why `HK0011_jessica` `LivingRoom_v40` is the only no-downsample seed in `v39..v43` that reaches the no-regress checkpoint floor, while the neighboring seeds collapse into active-heavy fallback selection.

## Current state

- Live `HK0011_jessica` is already promoted and must remain frozen:
  - Bathroom `v35`
  - Bedroom `v38`
  - Entrance `v26`
  - Kitchen `v27`
  - LivingRoom `v40`
- The prior promotion review already established:
  - Dec 17 live replay matches the validated support-fix candidate exactly at overall final macro-F1 `0.4486`
  - LivingRoom `v39..v43` all share essentially the same no-downsample prior drift of about `3.97` percentage points
  - only `v40` reaches `selection_mode = no_regress_floor`

## Problem

The remaining risk is no longer runtime confidence wiring, Bedroom support gating, or unoccupied downsample drift. The unresolved question is narrower:

- under the same LivingRoom no-downsample recipe, why does seed `42` (`v40`) produce a recoverable validation geometry while seeds `41`, `43`, `44`, and `45` collapse into `no_regress_macro_f1_fallback`?

The answer needs to be mergeable and reproducible, not just a one-off terminal readout.

## Approaches considered

### 1. Manual doc-only comparison

Pros:

- fastest
- no code changes

Cons:

- successor cannot reproduce the comparison mechanically
- easy to omit fields or make transcription mistakes
- weak merge value

Decision: reject.

### 2. One-off notebook or ad hoc shell parsing

Pros:

- quick to iterate
- can inspect more fields than a handwritten note

Cons:

- output is not naturally regression-tested
- logic stays hidden in shell history
- awkward to reuse for future seed panels

Decision: reject.

### 3. Small forensic helper plus regression test and review note

Pros:

- reproducible comparison from saved artifacts
- unit-testable on synthetic fixtures
- produces a structured JSON artifact that the review note can cite directly
- mergeable for future LivingRoom or other room seed forensics

Cons:

- slightly more upfront work than a manual note

Decision: recommend.

## Design

### Scope

- Stay LivingRoom-only.
- Read existing saved artifacts only; do not retrain.
- Compare exactly `LivingRoom_v39..v43`.
- Keep the live namespace unchanged.

### Forensic helper

Add a small script under `backend/scripts/` that:

- loads `backend/models/HK0011_jessica/LivingRoom_versions.json`
- loads each `LivingRoom_v39..v43_decision_trace.json`
- loads each `LivingRoom_v39..v43_activity_confidence_calibrator.json`
- derives the effective seed from `policy.reproducibility.random_seed`
- emits a JSON summary containing:
  - version and seed
  - macro-F1 and checkpoint-selection mode
  - no-regress target floor and whether it was reached
  - best-epoch and last-epoch validation class distributions
  - class thresholds
  - train / validation / holdout support
  - post-sampling prior drift
  - activity-confidence calibrator geometry
  - comparison to the selected winner `v40`

### Interpretation rules

The helper should classify each seed path using artifact-backed facts:

- `reaches_no_regress_floor`
- `fallback_selected`
- `collapsed_best_epoch`
- `active_heavy_best_epoch`
- `unoccupied_preserving_best_epoch`

The review note should interpret those outputs conservatively:

- identical post-sampling prior drift means the failure is not renewed downsample distortion
- the split appears before runtime thresholding, because the validation geometry already diverges at checkpoint-selection time
- calibration artifacts mirror the split but are downstream reflections, not the first cause

### Documentation

Record the design, implementation plan, JSON artifact path, and the final review note in the branch so the successor can merge the forensic smoothly later.

## Success criteria

- A reusable helper exists and is covered by a focused regression test.
- The helper emits a LivingRoom seed forensic JSON artifact from the saved Jessica models.
- A review note exists that explains why `v40` succeeds and why the neighboring seeds fail.
- The note explicitly states that live remains frozen and that the next thread should target LivingRoom optimizer / checkpoint stability, not confidence/runtime or Bedroom work.
