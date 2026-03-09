# Beta 6.1 / Beta 6.2 Reliable Activity Timeline Design

Promoted to the active `docs/plans/` tree on 2026-03-10 after recovery of the frozen `beta6.1/6.2` thread.

Planning baseline:
1. `codex/pilot-bootstrap-gates` at `94524af`

Validated current branch state:
1. local branch `codex/pilot-bootstrap-gates` at `6794c6c`
2. targeted Beta 6.1 / Beta 6.2 validation passed locally on the recovered working tree

Reviewed evidence branch:
1. `origin/codex/livingroom-fast-diagnosis` at `6c4128b` remains a narrow LivingRoom tuning candidate, not a replacement roadmap baseline

## Goal

Turn Beta 6 into a product-reliable activity timeline system by separating:
1. Beta 6.1 as the productionization track
2. Beta 6.2 as the correction-reduction and learning-efficiency track

## Why the March 8 plan is no longer sufficient

The March 8 LivingRoom plan was directionally useful, but the branch baseline moved materially:
1. `f565ce1` hardened authority/runtime handling
2. `a27cf50`, `cc09e0e`, `ace7a7a`, and `94524af` improved LivingRoom reliability and review tooling
3. `ffd23de` moved startup defaults toward a production-profile launch path
4. `6c4128b` showed that some rooms remain policy-sensitive and benefit from small replayable policy sweeps

The remaining product risks are no longer dominated by isolated room failures. They are:
1. inconsistent authority/runtime assumptions across env, scripts, and registry flows
2. insufficient end-to-end measurement of timeline reliability and correction load
3. a correction loop that exists operationally but is not consistently treated as a learning system
4. duplicated namespace/module surfaces that slow safe iteration

## Design Principles

### 1. Product timeline quality is the top-level objective

Optimize for:
1. event and timeline fidelity
2. low contradiction rate
3. stable duration and onset/offset behavior
4. low manual review load
5. safe deterministic authority behavior

### 2. Beta 6.1 is the productionization track

Beta 6.1 should focus on:
1. explicit production contracts
2. deterministic rollback and fallback
3. real-environment shadow certification
4. product-facing reliability instrumentation
5. replayable room-policy diagnostics for sensitive rooms

Beta 6.1 should not absorb new modeling risk unless it directly fixes a production blocker.

### 3. Beta 6.2 is the correction-reduction track

Beta 6.2 should be judged by whether it reduces human work while improving timeline quality:
1. maximize learning from each corrected minute
2. maximize learning from each retrain
3. reduce review volume over time
4. improve event-native reliability before escalating to humans

### 4. Human correction is front-loaded, then progressively reduced

Human correction remains:
1. a safety backstop
2. a drift monitor
3. an active-learning source

It should not become the default operating mode.

### 5. Explicit contracts beat hidden env and duplicated code paths

Beta 6 repeatedly showed that:
1. hidden env/config state creates false blockers and false confidence
2. duplicated module surfaces slow debugging and safe experimentation

Beta 6.2 should reduce ambiguity rather than add to it.

## Beta 6.1 Design

### Purpose

Make the current integrated Beta 6 stack production-credible.

### Main workstreams

1. Authority/runtime contract hardening
2. Product reliability visibility
3. Resident/home context contract
4. Real-environment certification
5. Room-policy sensitivity guardrails

### Beta 6.1 success definition

1. required authority artifacts are signed and written correctly
2. rollback/fallback is deterministic and tested
3. stable-day certification can be measured without hidden env assumptions
4. product/ops views show timeline reliability and correction burden clearly
5. room-level default changes are backed by replay evidence, not ad hoc retunes

## Beta 6.2 Design

### Purpose

Maximize model-first timeline performance and minimize manual correction.

### Main workstreams

1. shared `20x14` corpus program
2. correction-to-training loop
3. active-learning prioritization
4. timeline-native learning
5. context-conditioned modeling
6. learning-efficiency infrastructure
7. namespace and SSOT cleanup

### Beta 6.2 success definition

1. correction rate declines over time on the same cohort
2. event-level timeline quality improves against the Beta 6.1 baseline
3. training and research throughput improve materially
4. Beta 6.1 outputs remain unchanged when Beta 6.2 flags are off
5. room-level diagnosis can be done with small replay runs before full retrains

## Metrics That Matter

### Beta 6.1 production metrics

1. room pass rate
2. run-level authority pass rate
3. rollback drill success rate
4. signed artifact completeness
5. timeline contradiction rate
6. duration MAE / fragmentation / unknown-rate / abstain-rate
7. daily correction volume and review backlog

### Beta 6.2 learning metrics

1. review minutes required per resident-day
2. accepted corrections per 100 resident-days
3. event IoU / onset-offset tolerance / duration error
4. residual contradiction rate after decoder/arbitration
5. retrain wall-clock and cost per experiment
6. sample efficiency from pretraining / active-learning / context conditioning
7. time-to-diagnose a room-policy regression using replay sweeps

## Consistency With the Recovered Branch

This design is consistent with the recovered `codex/pilot-bootstrap-gates` findings.

Already present in the current local working tree and validated by targeted suites:
1. authority preflight hardening
2. deterministic fallback/rollback resolution
3. product-facing reliability and correction scorecards
4. replayable room-policy diagnostics
5. resident/home context contract
6. Beta 6.2 import-boundary / SSOT guardrails
7. shared `20x14` corpus contract
8. correction-derived learning signals
9. timeline-native targets/metrics/heads
10. learning-efficiency infrastructure

Still open or externally blocked:
1. real-environment certification rerun against PostgreSQL-backed authority flow
2. additional resident coverage beyond `HK001_jessica`
3. Step 4 shadow soak evidence window
4. final promotion artifacts and sign-off

## Planning Consequences

1. The March 8 roadmap remains useful background, but it is no longer the active Beta 6.1 / 6.2 plan.
2. Beta 6.1 must explicitly own product-facing reliability and correction observability.
3. Beta 6.2 must explicitly own correction reduction and learning efficiency.
4. Replayable room-policy diagnostics should be the path for sensitive-room investigation before default retunes.
