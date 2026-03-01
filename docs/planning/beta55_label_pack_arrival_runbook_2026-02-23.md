# Beta 5.5 Label Pack Arrival Runbook (2026-02-23)

## Purpose
Operational SOP for same-day intake -> smoke -> matrix -> signoff when corrected label pack arrives.

## Inputs
- Candidate pack directory
- Baseline pack directory
- Elder ID (`HK0011_jessica` default)
- Day window (`4-10` default)

## Step 1: Intake Validation
```bash
python3 backend/scripts/validate_label_pack.py \
  --pack-dir "<CANDIDATE_PACK_DIR>" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 10 \
  --output /tmp/label_pack_validation.json
```
Expected:
- JSON exists
- `status=pass`
- No blocking violations

## Step 2: Baseline-vs-Candidate Diff
```bash
python3 backend/scripts/diff_label_pack.py \
  --baseline-dir "<BASELINE_PACK_DIR>" \
  --candidate-dir "<CANDIDATE_PACK_DIR>" \
  --elder-id HK0011_jessica \
  --min-day 4 --max-day 10 \
  --json-output /tmp/label_pack_diff.json \
  --csv-output /tmp/label_pack_diff.csv
```
Expected:
- Diff artifacts exist
- `minutes_changed_total` is non-zero if corrections are present

## Step 3: Smoke Check (single seed/day)
```bash
python3 backend/scripts/run_event_first_smoke.py \
  --data-dir "<CANDIDATE_PACK_DIR>" \
  --elder-id HK0011_jessica \
  --day 7 --seed 11 \
  --expectation-config backend/config/event_first_go_no_go.yaml \
  --diff-report /tmp/label_pack_diff.json \
  --output /tmp/event_first_smoke.json
```
Expected:
- `status=pass`
- backtest report path exists
- no blocking reasons

## Step 4: Full Matrix Execution
```bash
python3 backend/scripts/run_event_first_matrix.py \
  --profiles-yaml backend/config/event_first_matrix_profiles.yaml \
  --profile pre_arrival_full \
  --data-dir "<CANDIDATE_PACK_DIR>" \
  --elder-id HK0011_jessica \
  --output-dir /tmp/event_first_matrix \
  --max-workers 3
```
Expected:
- Variant directories with seed reports + rolling/signoff
- Matrix manifest with no failed commands

## Step 5: Before/After Decision Pack
```bash
python3 backend/scripts/summarize_before_after.py \
  --before-rolling "<BEFORE_ROLLING_JSON>" \
  --before-signoff "<BEFORE_SIGNOFF_JSON>" \
  --after-rolling "<AFTER_ROLLING_JSON>" \
  --after-signoff "<AFTER_SIGNOFF_JSON>" \
  --markdown-output /tmp/before_after.md \
  --csv-output /tmp/before_after.csv
```
Expected:
- Markdown + CSV are generated
- Room-level accuracy/macro_f1/occupied_f1/occupied_recall/fragmentation deltas available

## Escalation Matrix
- Validation fail: stop and fix pack naming/schema/content.
- Smoke fail: stop and inspect smoke blocking reasons + diff evidence.
- Matrix partial fail: rerun failed seeds only after root-cause fix.
- Signoff fail: no promotion; archive artifacts and create failure forensics note.

## Operator Checklist
1. Validation pass recorded.
2. Diff evidence recorded.
3. Smoke pass recorded.
4. Matrix completed.
5. Before/after summary attached.
6. Go/no-go decision documented.
