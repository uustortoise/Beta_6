# Beta6 Bedroom Separation Retrain Design

**Context**

- The March 11 canonical Bedroom benchmark remains `0.3511` final macro-F1 on Dec 17.
- The Bedroom policy-only ablation showed runtime policy is not the fix:
  - removing `low_confidence` changes benchmark accounting
  - it does not materially improve three-label Bedroom activity quality
- `Bedroom_v38` already restored the near-neutral Bedroom room-label weighting profile and Bedroom post-split shuffle.
- The remaining strong training-time anomaly is Bedroom class-0 inflation:
  - pre-sampling train share for `bedroom_normal_use`: about `9.0%`
  - post-sampling train share for `bedroom_normal_use`: about `23.6%`
  - Dec 17 dominant error: true `unoccupied -> bedroom_normal_use`

**Approaches considered**

1. Weight-profile-only rerun
- Reapply the older `v28`-style room-label weighting profile and Bedroom shuffle.
- Rejected as the primary next run because `Bedroom_v38` already uses that restored posture.

2. Bedroom class-0 inflation rollback
- Keep the current restored Bedroom weight profile and shuffle.
- Neutralize Bedroom-only transition-focus and minority-sampling expansion by setting both max multipliers to `1`.
- Recommended because it directly tests the strongest remaining causal signal with the smallest model-side change.

3. Threshold / two-stage tuning
- Change threshold or runtime policy without retraining.
- Rejected because the completed ablation showed negligible substantive leverage.

**Recommended design**

Run one Bedroom-only retrain on the corrected Jessica pack in a fresh candidate namespace cloned from the current `HK0011_jessica` model state.

Keep:

- current Bedroom room-label weighting posture
- current Bedroom post-split shuffle
- current runtime / threshold policy

Change only:

- `TRANSITION_FOCUS_MAX_MULTIPLIER_BY_ROOM=bedroom:1`
- `MINORITY_MAX_MULTIPLIER_BY_ROOM=bedroom:1`

This tests one narrow hypothesis:

- the remaining Dec 17 Bedroom weakness is driven by training-time inflation of `bedroom_normal_use`, which pushes the model to over-predict occupancy deep inside true `unoccupied` segments

**Success criteria**

- holdout stays non-collapsed
- Dec 17 canonical replay improves Bedroom final macro-F1 beyond the `0.3511` reference
- the dominant `unoccupied -> bedroom_normal_use` error count falls materially versus `Bedroom_v38`

**Failure interpretation**

- If Bedroom still over-predicts `bedroom_normal_use` at roughly the same rate, then the next highest-value Bedroom thread is no longer policy sampling; it is representation / feature separation between occupied and unoccupied Bedroom regimes.
