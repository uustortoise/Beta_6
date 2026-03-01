# Beta 6 Label Adjudication Runbook

- Date: 2026-02-26
- Scope: Phase 3.3 reviewer workflow

## 1. Daily Flow

1. Generate triage queue (`run_active_learning_triage.py`).
2. Split queue by risk band (`green`, `amber`, `red`).
3. Assign reviewers and SLA target by band.
4. Record reviewer decisions with reason codes.
5. Resolve disagreements via adjudication policy.

## 2. Disagreement Handling

1. Round 1 disagreement:
   - Route to second reviewer.
2. Round 2 disagreement:
   - Escalate to `clinical_reviewer_oncall`.
3. Final adjudication:
   - Record final label + rationale.
   - Mark queue item closed with audit entry.

## 3. Quality Checks

1. Verify no missing audit fields.
2. Verify dual-review completion for amber/red.
3. Verify SLA breach count and unresolved queue count.

## 4. Incident Fallback

1. If adjudication service is unavailable:
   - Freeze new `red` promotions.
   - Continue `green` backlog only.
2. Reconcile deferred items once service recovers.
