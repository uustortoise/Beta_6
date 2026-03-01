# Beta 6 Label-Pack Intake Runbook (Phase 0)

- Status: Draft
- Date: 2026-02-25
- Purpose: Fail-closed intake gate before any Phase 1+ model work

## 1. Required Inputs

1. Candidate pack directory.
2. Baseline pack directory.
3. Elder ID.
4. Day window (`min_day`, `max_day`) and smoke day.

## 2. Single Command Intake Execution

```bash
python3 backend/scripts/run_beta6_label_pack_intake.py \
  --candidate-dir "<CANDIDATE_PACK_DIR>" \
  --baseline-dir "<BASELINE_PACK_DIR>" \
  --elder-id HK0011_jessica \
  --min-day 4 \
  --max-day 10 \
  --smoke-day 7 \
  --seed 11 \
  --expectation-config backend/config/event_first_go_no_go.yaml \
  --registry-path backend/config/adl_event_registry.v1.yaml \
  --output-dir /tmp/beta6_intake_20260225
```

Expected outputs:
1. `/tmp/beta6_intake_20260225/label_pack_validation.json`
2. `/tmp/beta6_intake_20260225/label_pack_diff.json`
3. `/tmp/beta6_intake_20260225/label_pack_diff.csv`
4. `/tmp/beta6_intake_20260225/label_pack_smoke.json`
5. `/tmp/beta6_intake_20260225/beta6_label_pack_intake_artifact.json`

Exit behavior:
1. Exit `0`: intake gate approved.
2. Exit `2`: intake gate rejected (fail-closed).

## 3. Mandatory Precheck Before Phase 1+ Jobs

```bash
python3 backend/scripts/check_beta6_intake_gate.py \
  --artifact /tmp/beta6_intake_20260225/beta6_label_pack_intake_artifact.json
```

Expected:
1. Command prints approved status.
2. Exit `0` only when artifact schema is valid, gate is approved, and report files exist.

Beta 6 training entrypoint (Phase 1+) is fail-closed on this artifact:

```bash
python3 backend/ml/beta6/beta6_trainer.py \
  --dataset backend/data/golden_samples \
  --intake-artifact /tmp/beta6_intake_20260225/beta6_label_pack_intake_artifact.json
```

Blocking behavior:
1. Missing artifact: `reason_code=intake_gate_missing_artifact`, exit `2`.
2. Invalid artifact/schema: `reason_code=intake_gate_invalid_artifact`, exit `2`.
3. Rejected artifact: `reason_code=intake_gate_not_approved`, exit `2`.

## 4. Gate Policy

1. Intake gate is approved only when all three steps pass:
   - `validate_label_pack`
   - `diff_label_pack`
   - `run_event_first_smoke`
2. Any step failure or exception produces a deterministic blocking reason.
3. Validation failure forces smoke step to `skipped` and keeps gate rejected.
4. Rejected intake artifacts must not be used for Phase 1+ model runs.

## 5. Operator Escalation

1. `validate_failed`: fix schema/content/room-sheet issues in candidate pack.
2. `diff_exception:*`: fix baseline/candidate path or parse issues.
3. `smoke_failed` or `smoke:*`: inspect smoke blocking reasons and re-run after fixes.
4. `smoke_skipped_due_validation_fail`: resolve validation errors first.
