# Beta 6 Pretrain Corpus Manifest Contract (Beta 6.2)

- Date: 2026-02-26
- Status: Active
- Builder: `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/build_pretrain_corpus_manifest.py`
- Module: `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/data_manifest.py`

## 1. Purpose

Create a deterministic, replayable Beta 6.2 corpus contract that supports:
1. shadow cohort coverage tracking
2. unlabeled pretrain intake
3. labeled high-trust fine-tune/eval intake
4. resident/home context completeness checks
5. label-quality and review-trust accounting
6. auditable fingerprinting for cache-safe reuse

## 2. Manifest Schema (v1)

Top-level required fields:
1. `manifest_version`
2. `generated_at`
3. `policy`
4. `corpus_roots`
5. `entries`
6. `duplicates`
7. `violations`
8. `stats`
9. `fingerprint`
10. `corpus_views`
11. `context_summary`
12. `label_quality_summary`
13. `beta62_corpus_contract` when emitted by the CLI builder

Per-entry contract fields:
1. `resident_id`
2. `views`
3. `days_covered`
4. `context_completeness`
5. `label_quality`

## 3. Determinism Rules

1. Files are discovered in sorted deterministic order.
2. Duplicate files are removed by content hash (`sha256`).
3. Fingerprint is computed from normalized entry rows only (independent of generation timestamp).
4. Re-running on unchanged corpus must produce identical fingerprint.

## 4. Corpus Views

The builder tracks three canonical Beta 6.2 views:
1. `shadow_cohort`
2. `unlabeled_pretrain`
3. `labeled_high_trust_finetune_eval`

These are summarized under `corpus_views` with:
1. `entry_count`
2. `resident_count`
3. `resident_ids`
4. `min_days_covered`
5. `max_days_covered`

## 5. Context and Label Quality

Each manifest entry records:
1. `context_completeness.status`
2. `context_completeness.missing_fields`
3. `label_quality.trust_tier`
4. `label_quality.reviewed_fraction`
5. `label_quality.source`

The top-level manifest summarizes:
1. ready vs missing-context entries under `context_summary`
2. high-trust and reviewed-fraction totals under `label_quality_summary`

## 6. P0 Data Contract Violations

The builder emits P0 violations for:
1. Parse/read errors for candidate files.
2. `row_count < min_rows`.
3. `feature_count < min_features`.
4. `missing_ratio > max_missing_ratio`.

Any non-zero `stats.p0_violations` is fail-closed for Phase 2 pretraining starts.

Additional fail-closed rule:
1. `stats.records_kept` must be `> 0` (empty corpus is a hard failure).

## 7. Beta 6.2 Contract Evaluation

`evaluate_beta62_corpus_contract(...)` fails closed when:
1. `shadow_cohort.resident_count < required_residents`
2. `shadow_cohort.min_days_covered < required_days`
3. `labeled_high_trust_finetune_eval.min_days_covered < required_days`
4. any kept entries are missing required resident/home context

Current reason codes:
1. `shadow_resident_coverage_below_contract`
2. `shadow_days_coverage_below_contract`
3. `labeled_days_coverage_below_contract`
4. `resident_home_context_incomplete`

The CLI builder exits non-zero when:
1. any P0 manifest violations exist
2. no usable records are kept
3. the Beta 6.2 corpus contract does not pass
