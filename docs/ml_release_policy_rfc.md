# RFC: ML Release Gates and Training Operations (Beta 5.5)

- Status: Proposed for immediate adoption
- Date: February 13, 2026
- Scope: Resident activity recognition model lifecycle (training, evaluation, deployment)
- Source of truth config: `backend/config/release_gates.json`

## 1. Decision Summary

This RFC defines the minimum operational policy required to make model updates safe and measurable:

1. Phase 0.5 auto-aggregate training is mandatory before strict gating.
2. Every candidate model is evaluated as challenger vs champion.
3. Deployment happens only on gate pass; otherwise champion remains.
4. Correction updates must show local benefit and no unacceptable global harm.

## 2. Prerequisite: Phase 0.5 Auto-Aggregate

Daily training must use all archived + incoming training files for the resident, not only the newest file.

- Required behavior: `train_from_files(all_archived_plus_incoming)`
- Problem prevented: catastrophic forgetting from single-file replacement training

## 3. Hard Release Gates

### 3.1 Base thresholds (2-9 days of data)

| Scope | Metric | Threshold | Blocking |
|---|---|---:|---|
| Global | Macro F1 | >= 0.55 | Yes |
| Bedroom | F1 | >= 0.60 | Yes |
| Kitchen | F1 | >= 0.55 | Yes |
| LivingRoom | F1 | >= 0.50 | Yes |
| Entrance | F1 | >= 0.65 | Yes |
| Bathroom | F1 | >= 0.30 | No (under investigation) |

No-regress rule:

- Any non-exempt room F1 cannot drop by more than 0.05 from champion.
- Bathroom is exempt from blocking and exempt from no-regress until label audit closure.

### 3.2 Ratchet schedule

| Data volume milestone | Global Macro F1 | Bedroom F1 |
|---|---:|---:|
| 10+ days | >= 0.65 | >= 0.70 |
| 22+ days | >= 0.75 | >= 0.80 |

## 4. Corrected-Window Policy

For manual correction-triggered updates:

- Corrected region F1 gain must be >= 0.10 absolute to count as strong local improvement.
- Global Macro F1 drop must be <= 0.02 absolute.

Decision outcomes:

- PASS: local gain >= 0.10 and global drop <= 0.02
- PASS_WITH_FLAG: local gain < 0.10 and global drop <= 0.02
- FAIL: global drop > 0.02

## 5. Retrain Cadence

| Trigger | Action | Timing | Approval |
|---|---|---|---|
| New file uploaded | Auto-aggregate retrain (all archived files) | Immediate, automated | Not required |
| Manual correction applied | Fine-tune from champion checkpoint | Immediate, automated | Not required |
| Weekly maintenance | Full retrain from scratch on all data | Sunday night cron | Not required |
| Label audit / major schema change | Full retrain + walk-forward validation | Ad hoc | Manual required |

Operational note: weekly full retrain remains challenger-only and must pass the same gates before deployment.

## 6. Fine-Tuning Constraints

Correction fine-tuning must use replay mixing to avoid overfitting tiny edits:

- Replay ratio: ~10x uncorrected samples per corrected sample
- Example: 78 corrected rows + ~780 random stratified uncorrected rows

## 7. Retrospective Mode Policy

- If auto-aggregate is live: retrospective mode is diagnostic/backtesting only.
- If auto-aggregate is not live: retrospective mode remains the primary multi-day training trigger.

## 8. Baseline Check at 22 Days

At 22 days of data, run XGBoost once as baseline floor:

- Transformer must exceed baseline by >= 5% absolute Macro F1.
- If not, open architecture-value investigation.

## 9. Data Ingestion Protocol

- Before auto-aggregate: upload all files, then execute one retrain.
- After auto-aggregate: order is no longer functionally critical, but batched uploads are still recommended.

## 10. Ownership and Control Flow

| Stage | Owner |
|---|---|
| Data QA | Domain expert / care staff |
| Training | Automated pipeline |
| Evaluation | Automated gates |
| Deployment approval / override | Senior engineer |
| Rollback | Automated |

Control flow:

1. File or correction arrives
2. Auto train (aggregate or fine-tune)
3. Auto evaluate
4. Gate check
5. Pass -> deploy + log
6. Fail -> keep champion + alert senior engineer

## 11. Implementation Artifacts

- Policy config: `backend/config/release_gates.json`
- Auto-aggregate scheduler logic: `backend/run_daily_analysis.py`

## 12. Implementation Notes (Feb 13, 2026)

- Scheduled threshold resolver behavior is hardened: if data volume is below the first configured bracket, resolver returns the **earliest** bracket threshold.
- Room gate lookup is normalized consistently with policy key format to avoid accidental `no_room_policy:*` bypass.
- Small-dataset bootstrap path supports gated promotion without forcing failure due to intentionally omitted validation split.
- Version cleanup preserves `current_version` artifacts/metadata to maintain rollback safety and metadata coherence.
- Prediction integrity validation is enforced as a blocking contract before persistence.
