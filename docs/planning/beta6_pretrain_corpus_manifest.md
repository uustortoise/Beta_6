# Beta 6 Pretrain Corpus Manifest Contract (Phase 2.1)

- Date: 2026-02-26
- Status: Active
- Builder: `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/build_pretrain_corpus_manifest.py`
- Module: `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/data_manifest.py`

## 1. Purpose

Create deterministic, replayable unlabeled-corpus inputs for self-supervised pretraining with auditable fingerprinting.

## 2. Manifest Schema (v2)

Top-level required fields:
1. `manifest_version`
2. `generated_at`
3. `policy`
4. `corpus_roots`
5. `entries`
6. `duplicates`
7. `quarantine`
8. `violations`
9. `summary`
10. `gate`
11. `stats`
12. `fingerprint`

## 3. Governed intake behavior

1. `entries` are the auto-approved sources that remain eligible for downstream pretraining.
2. `quarantine` contains red-flagged sources with explicit `quarantine_reasons`.
3. `summary.user_tags` and `summary.date_tags` provide grouped-by-user and grouped-by-date intake tags.
4. `summary.per_room_per_date_label_counts` provides audit-ready label density for tagged tabular training files.
5. `gate.approved=true` is allowed when at least one source is auto-approved, even if other sources are quarantined.
6. `gate.blocking_reasons` is non-empty only when no source survives intake.

## 4. Determinism Rules

1. Files are discovered in sorted deterministic order.
2. Duplicate files are removed by content hash (`sha256`).
3. Fingerprint is computed from normalized auto-approved entry rows only (independent of generation timestamp).
4. Re-running on unchanged corpus must produce identical fingerprint.

## 5. Red-flag quarantine rules

The builder quarantines sources for:
1. Parse/read errors for candidate files.
2. `row_count < min_rows`.
3. `feature_count < min_features`.
4. `missing_ratio > max_missing_ratio`.
5. missing user tags on labeled tabular sources.
6. missing date tags on labeled tabular sources.

## 6. Fail-closed pretraining contract

Phase 2 pretraining remains fail-closed when:
1. `gate.approved=false`
2. `stats.records_kept <= 0`
3. `stats.p0_violations > 0`

`stats.p0_violations` is now a manifest-level failure signal, not a count of quarantined sources. Mixed manifests with at least one auto-approved source can proceed while preserving the quarantine evidence for operator review.
