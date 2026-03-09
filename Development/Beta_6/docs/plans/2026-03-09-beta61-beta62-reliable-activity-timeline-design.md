# Beta 6.1 / Beta 6.2 Reliable Activity Timeline Design

**Baseline:** `codex/pilot-bootstrap-gates` at `94524af`

**Reviewed candidate branch:** `origin/codex/livingroom-fast-diagnosis` at `6c4128b` was reviewed as new evidence. It is treated here as a narrow LivingRoom tuning candidate, not as the new global planning baseline.

**Goal:** Turn Beta 6 into a product-reliable activity timeline system by separating the near-term Beta 6.1 productionization track from the Beta 6.2 model/data efficiency track, while explicitly reducing manual correction over time.

## Why the March 8 plan is no longer sufficient

The March 8 plan correctly shifted focus away from Bedroom/Entrance firefighting, but the baseline has moved:

1. `f565ce1` hardened parts of the authority/runtime path.
2. `a27cf50`, `cc09e0e`, `ace7a7a`, and `94524af` materially improved LivingRoom reliability and review tooling.
3. `ffd23de` moved startup defaults toward a production-profile launch path.
4. `6c4128b` showed that LivingRoom remains sensitive to room-specific downsample defaults, and that small replayable policy sweeps can materially change room stability.

The new lesson is that Beta 6 is no longer mainly blocked by isolated room-model failures. The remaining product risks are:

1. inconsistent authority/runtime assumptions across env, scripts, and registry flows
2. insufficient end-to-end measurement of timeline reliability and correction load
3. a correction loop that exists operationally, but is not yet treated as a first-class learning system
4. duplicated namespace/module surfaces that increase confusion and slow experimentation

## Design Principles

### 1. Product timeline quality is the top-level objective
Room accuracy and macro F1 still matter, but they are no longer enough. The system should optimize for:

1. event/timeline fidelity
2. low contradiction rate
3. stable duration and onset/offset behavior
4. low manual review load
5. safe and deterministic authority behavior

### 2. Beta 6.1 is the productionization track
Beta 6.1 should now focus on:

1. explicit production contracts
2. deterministic rollback/fallback
3. real-environment shadow certification
4. product-facing reliability instrumentation
5. replayable room-policy diagnostics for rooms that remain sensitive to small policy changes

Beta 6.1 should not absorb new modeling risk unless it directly fixes a production blocker.

### 3. Beta 6.2 is the correction-reduction track
Beta 6.2 should be judged by whether it reduces human work while improving timeline quality. That means:

1. maximizing learning from each corrected minute
2. maximizing learning from each retrain
3. reducing review volume over time
4. improving event-native reliability before escalating to humans

### 4. Human correction is front-loaded, then progressively reduced
The correction strategy should be:

1. heavy and systematic during `20 residents x 14 days`
2. ongoing afterward, but targeted and low-volume

Human correction remains part of the system, but only as:

1. safety backstop
2. drift monitor
3. active-learning source

It should not become the default operating mode.

### 5. Explicit contracts beat hidden env and duplicated code paths
Beta 6 taught two recurring lessons:

1. hidden env/config state creates false blockers and false confidence
2. duplicated module surfaces slow debugging and slow safe iteration

Beta 6.2 should not add new ambiguity. It should reduce it.

## Beta 6.1 Design

### Purpose
Make the current good integrated Beta 6 stack production-credible.

### Main workstreams

1. **Authority/runtime contract hardening**
   - signing key behavior
   - fallback target determinism
   - explicit evidence profile
   - clear preflight failures
   - non-contradictory artifact and event semantics

2. **Product reliability visibility**
   - daily timeline scorecards
   - contradiction and fragmentation trend visibility
   - review queue and correction volume visibility
   - operator-facing active-system clarity

3. **Resident/home context contract**
   - single vs multi-resident
   - helper presence
   - layout / adjacency / topology
   - context-aware routing, gating, arbitration, and cohorting

4. **Real-environment certification**
   - real PostgreSQL
   - signed artifacts
   - explicit production-profile run
   - Jessica + follow-on cohort shadow window

5. **Room-policy sensitivity guardrails**
   - replayable room-level policy sweep workflow before changing defaults
   - evidence pack for low-signal or policy-sensitive rooms such as LivingRoom
   - no silent default retunes without resident-level replay evidence

### Beta 6.1 success definition

1. all required authority artifacts are signed and written correctly
2. rollback/fallback is deterministic and tested
3. stable-day certification can be measured without hidden env assumptions
4. product/ops views show timeline reliability and correction burden clearly
5. room-level default changes are backed by replay evidence rather than ad hoc tuning

## Beta 6.2 Design

### Purpose
Maximize model-first timeline performance and minimize manual correction.

### Main workstreams

1. **Shared `20x14` corpus program**
   The corpus is a Beta 6 shared asset with three views:
   - authority shadow cohort
   - unlabeled pretrain corpus
   - labeled high-trust fine-tune/eval corpus

2. **Correction-to-training loop**
   Every accepted correction should become structured training signal:
   - corrected events
   - boundary targets
   - hard negatives
   - residual review packs

3. **Active-learning prioritization**
   Route humans to the highest-yield slices:
   - uncertainty spikes
   - model disagreement
   - timeline contradictions
   - transition boundaries
   - rare context profiles

4. **Timeline-native learning**
   Add event-native targets and heads:
   - onset
   - offset
   - duration
   - continuity

5. **Context-conditioned modeling**
   Use the highest-value metadata first:
   - layout / adjacency / topology
   - helper / household context
   - demographics only after explicit offline value and fairness review

6. **Learning-efficiency infrastructure**
   Improve how much learning we get from each run:
   - canonical cached feature/sequence store
   - manifest/policy fingerprinting
   - shared tensors across retries and seeds
   - early bad-candidate elimination
   - smaller probe experiments before full retrains
   - replayable room-policy sweep harnesses for fast diagnosis

7. **Namespace and SSOT cleanup**
   Pick authoritative module paths for Beta 6.2 and prevent duplicate shadow implementations from proliferating.

### Beta 6.2 success definition

1. correction rate declines over time on the same cohort
2. event-level timeline quality improves against Beta 6.1 baseline
3. training/research throughput improves materially
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

## Planning Consequences

1. The March 8 roadmap is still directionally useful, but it should no longer treat authority-only hardening as the whole Beta 6.1 story.
2. Beta 6.1 must now explicitly own product-facing reliability and correction observability.
3. Beta 6.2 must explicitly own correction reduction and learning efficiency, not just “offline experiments.”
4. The next execution plan should reflect current `94524af` state, not the older `3fb0257` baseline.
5. Small room-specific policy branches like `livingroom-fast-diagnosis` should be absorbed as replay-diagnostic lessons, not allowed to fragment the roadmap.
