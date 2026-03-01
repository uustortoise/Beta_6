# Beta 6 Pretrain Corpus Manifest Contract (Phase 2.1)

- Date: 2026-02-26
- Status: Active
- Builder: `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/build_pretrain_corpus_manifest.py`
- Module: `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/data_manifest.py`

## 1. Purpose

Create deterministic, replayable unlabeled-corpus inputs for self-supervised pretraining with auditable fingerprinting.

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

## 3. Determinism Rules

1. Files are discovered in sorted deterministic order.
2. Duplicate files are removed by content hash (`sha256`).
3. Fingerprint is computed from normalized entry rows only (independent of generation timestamp).
4. Re-running on unchanged corpus must produce identical fingerprint.

## 4. P0 Data Contract Violations

The builder emits P0 violations for:
1. Parse/read errors for candidate files.
2. `row_count < min_rows`.
3. `feature_count < min_features`.
4. `missing_ratio > max_missing_ratio`.

Any non-zero `stats.p0_violations` is fail-closed for Phase 2 pretraining starts.

Additional fail-closed rule:
1. `stats.records_kept` must be `> 0` (empty corpus is a hard failure).
