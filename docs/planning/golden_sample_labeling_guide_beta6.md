# Golden Sample Labeling Guide (Beta 6)

- Date: 2026-02-26
- Policy source: `/Users/dicksonng/DT/Development/Beta_6/backend/config/label_review_policy.yaml`

## 1. Labeling Priority

1. `red` queue: clinically sensitive or high-disagreement windows.
2. `amber` queue: model uncertainty/disagreement windows.
3. `green` queue: low-risk consistency checks.

## 2. Reviewer Rules

1. Always record final label and reason code.
2. Do not overwrite source evidence; add adjudicated decision as new record.
3. Escalate `red` windows to clinical reviewer when decision remains uncertain.

## 3. Dual Review Requirements

1. `amber` and `red` require two independent reviewers.
2. If reviewers disagree after first pass:
   - Start adjudication round 2.
   - If still unresolved, escalate to clinical owner.

## 4. Output Requirements

1. Include `candidate_id`, `reviewer_id`, `decision`, `reason_code`, and `reviewed_at`.
2. Keep decision logs for audit retention window (`365` days).
