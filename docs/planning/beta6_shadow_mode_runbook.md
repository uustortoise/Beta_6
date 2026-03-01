# Beta 6 Shadow Mode Runbook (Phase 6.1)

## Purpose
- Run Beta 6 in shadow against Beta 5.5 without changing active care authority.
- Surface room-level divergence with plain-language reasons first, technical trace second.

## Sources of truth
- Runtime policy artifact: `registry_v2/<elder>/_runtime/phase4_runtime_policy.json`
- Shadow compare artifact: `backend/tmp/beta6_gate_artifacts/<elder>/<run_id>/<run_id>_shadow_compare_report.json`
- Training metadata path: `training_history.metadata.beta6_gate.phase6_shadow_compare`

## Operator checks (daily)
1. Confirm active-system banner shows `Beta 5.5 currently active (Beta 6 in Shadow)` when rollout stage is `shadow` or `canary`.
2. Review `Model Insights -> Shadow Divergence Diagnostics`.
3. If status is `critical`, inspect top room badges and open technical trace for each affected room.
4. Verify unexplained divergence rate is at or below configured threshold (`BETA6_SHADOW_UNEXPLAINED_DIVERGENCE_RATE_MAX`, default `0.05`).

## Escalation rules
- `status=critical`: open incident and block rung progression.
- `status=watch`: continue shadow while tracking trend for 48 hours.
- `status=ok`: eligible for next rollout checkpoint, subject to all other gates.

## Required evidence for rung progression
- Latest signed shadow compare artifact path + signature.
- Divergence summary (`divergence_rate`, `unexplained_divergence_rate`).
- Room-level badge list with reason text and linked technical trace.
