# Beta6 Jessica Promotion And Deep Dive Design

**Date:** 2026-03-11

**Goal:** Promote the validated `HK0011_jessica_candidate_supportfix_20260310T2312Z` Bedroom/LivingRoom improvements into live `HK0011_jessica` first, then use the now-promoted baseline to run one narrow model-risk deep dive.

## Current state

- Live namespace `HK0011_jessica` still points to:
  - Bathroom `v35`
  - Bedroom `v28`
  - Entrance `v26`
  - Kitchen `v27`
  - LivingRoom `v27`
- Promotion-grade candidate namespace `HK0011_jessica_candidate_supportfix_20260310T2312Z` currently points to:
  - Bathroom `v35`
  - Bedroom `v38`
  - Entrance `v26`
  - Kitchen `v27`
  - LivingRoom `v40`
- The candidate replay already matched the prior benchmark leader on the corrected Dec 17 workbook:
  - overall final macro-F1 `0.4486`
  - Bedroom final macro-F1 `0.3511`
  - LivingRoom final macro-F1 `0.4340`

## Approaches considered

### 1. Overwrite the entire live namespace with the candidate namespace

Pros:

- operationally simple
- easy to explain

Cons:

- live `Bedroom` and `LivingRoom` rollback history would be replaced by the candidate namespace metadata
- the candidate namespace does not carry the older live `Bedroom_v28` / `LivingRoom_v27` chain in its `*_versions.json`
- too risky for a promotion that may need deterministic rollback

Decision: reject.

### 2. Perform one-off manual JSON/file edits directly in the live namespace

Pros:

- no code changes
- fastest path if done perfectly

Cons:

- brittle and hard to audit
- easy to miss optional artifacts such as thresholds, confidence calibrators, or two-stage files
- no regression protection for future room-wise promotions

Decision: reject.

### 3. Add a small scripted room-wise merge helper and use the existing registry alias-sync path

Pros:

- preserves live rollback history while importing only the desired candidate room versions
- lets the existing registry logic materialize latest aliases, thresholds, confidence artifacts, and two-stage files
- testable before touching the live namespace

Cons:

- slightly more upfront work than a one-off shell sequence

Decision: recommend.

## Promotion design

- Add a script that copies selected room versioned artifacts from a source namespace into a target namespace.
- Merge `*_versions.json` entries by version number, with hard failure on conflicting duplicate versions.
- Preserve the target room's existing history and rollback points.
- Set the target room's current version by calling the existing registry rollback helper after the merge.
- Promote only the changed rooms:
  - Bedroom -> `v38`
  - LivingRoom -> `v40`
- Leave Bathroom, Entrance, and Kitchen untouched because the candidate matches live there already.

## Verification design

- Before promotion:
  - regression-test the merge helper in isolation
- After promotion:
  - load the live `HK0011_jessica` namespace through `UnifiedPipeline`
  - confirm current versions are Bedroom `38` and LivingRoom `40`
  - replay the corrected Dec 17 workbook through live `HK0011_jessica`
  - confirm the live replay matches the promoted candidate metrics closely enough to treat the promotion as faithful

## Deep dive design

- Do not reopen Bedroom support-gating work unless live promotion exposes a regression.
- Focus the deep dive on LivingRoom seed instability around `v40`, because that is the highest remaining model-risk after promotion.
- Use existing saved candidate metrics and decision traces to compare:
  - selected winner `v40`
  - neighboring candidate-only seeds/versions
- The deep dive deliverable is a short review note explaining whether the instability appears to come from sampling, calibration geometry, or checkpoint selection.

## Success criteria

- Live `HK0011_jessica` loads cleanly with Bedroom `v38` and LivingRoom `v40`.
- The corrected Dec 17 replay on live `HK0011_jessica` remains aligned with the support-fix candidate benchmark.
- Rollback history for the live rooms remains intact in `*_versions.json`.
- A post-promotion LivingRoom deep dive artifact exists and states the most likely residual failure mode.
