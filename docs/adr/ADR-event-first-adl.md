# ADR: Event-First ADL Modeling for Real-World Home Patterns

## Status
Accepted (implementation in progress)

## Date
February 16, 2026

## Context
Current window-level multiclass ADL modeling underperforms on critical minority labels in realistic elder-home patterns dominated by `unoccupied` windows. Additional same-pattern data is unlikely to change this ceiling materially.

The product goal is care insight reliability (sleep routine, shower regularity, kitchen use, living room activity), not perfect per-window multiclass prediction.

## Decision
Adopt an event-first architecture and evaluation protocol while keeping labeling workflow simple.

1. Keep current labels close to reality (no complicated relabeling process).
2. Train occupancy-aware models (occupied vs unoccupied + occupied activity discrimination).
3. Decode model outputs into stable episodes/events with temporal hysteresis.
4. Promote based on event-level care KPIs first, window-level macro-F1 second.

## Event-Level Success Criteria
Primary gates:
- Sleep duration MAE (minutes/day)
- Shower-day precision/recall
- Kitchen-use detection precision/recall
- Living-room active-minutes MAE
- Collapse check for critical labels/events (no near-zero recall under sufficient support)
- Uncertainty-rate cap

Secondary diagnostics:
- Window-level macro-F1 and per-label recall

## Why This Decision
- Matches care objectives directly.
- Reduces sensitivity to inevitable short-term label noise and transition ambiguity.
- Better aligned to realistic home behavior where long unoccupied segments dominate.

## Non-Goals
- No complex new labeling ontology.
- No dependence on adding more same-pattern training files as primary fix.

## Implementation Steps (Initial)
1. Add event label compiler from existing labels.
2. Add temporal decoder with hysteresis.
3. Add event metric utilities and gating hooks.
4. Add event-model skeleton (two-stage occupancy-first).
5. Run rolling-day retrospective protocol and compare against current baseline.

## Risks and Mitigations
Risk: Event decoder can hide short true events.
Mitigation: Maintain tunable thresholds + minimum support checks and report uncertainty.

Risk: Better event KPI but lower window macro-F1.
Mitigation: Treat macro-F1 as secondary metric unless event KPI regresses.

## Rollout
- Shadow mode first.
- Controlled validation run before promotion.
- Config switch for immediate rollback.
