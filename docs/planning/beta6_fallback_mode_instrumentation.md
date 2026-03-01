# Beta 6 Operator-Safe Fallback Instrumentation (v1)

- Date: 2026-02-25
- Status: Active
- Registry implementation: `backend/ml/beta6/registry_v2.py`
- Policy file: `backend/config/beta6_fallback_mode_policy.yaml`

## 1. State Machine

1. `inactive` -> `active` via `activate_fallback_mode(...)`
2. `active` -> `inactive` via `clear_fallback_mode(...)`

State is persisted at room scope in `fallback_state.json`.

## 2. Deterministic Reason Codes

1. activation success: `fallback_activated`
2. clear success: `fallback_cleared`
3. activate while active: `fallback_already_active`
4. clear while inactive: `fallback_not_active`
5. activation without target pointer: `fallback_missing_target`

## 3. Auditability Requirements

Activation event payload includes:
1. trigger reason code
2. pointer switch indicator
3. pointer before/after
4. fallback flags
5. fallback state path

Clear event payload includes:
1. restore request flag
2. restore applied flag
3. pointer before/after
4. fallback state path

## 4. Recoverability

Fallback clear can restore pre-fallback champion pointer (`restore_previous_pointer=True`).
This is deterministic and traceable via:
1. `fallback_state.json`
2. room `events.jsonl`
3. room `champion_history.jsonl`
