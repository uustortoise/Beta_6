# Beta 6.1 Authority Hardening and Beta 6.2 Isolation Roadmap

> **Superseded:** Use [2026-03-09-beta61-beta62-reliable-activity-timeline-design.md](/Users/dickson/DT/DT_development/Development/Beta_6/docs/plans/2026-03-09-beta61-beta62-reliable-activity-timeline-design.md) for the current roadmap/design baseline based on `codex/pilot-bootstrap-gates` at `94524af`.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reprioritize Beta 6 work so Beta 6.1 focuses on authority/runtime hardening of the now-good Jessica model stack, while Beta 6.2 remains an isolated offline R&D track until Beta 6.1 passes real-environment shadow certification.

**Architecture:** Treat the integrated Jessica model stack as the current Beta 6 baseline and stop spending the main critical path on room-model forensics. Beta 6.1 becomes the release track for authority signing, rollback/fallback completeness, real-environment runtime readiness, and shadow certification. Beta 6.2 remains a separate namespace for timeline-native supervision and other training/objective upgrades, with no allowed impact on Beta 6.1 outputs while flags are off.

**Tech Stack:** Python 3.12, pytest, signed JSON artifacts, PostgreSQL-backed runtime flows, Beta 6 registry v2, rollout ladder, room decision traces, phase-6 stability artifacts.

---

## Executive Assessment

### What the integrated Jessica rerun proved
The integrated run `beta6_daily_HK0011_jessica_20260308T005915Z` on branch `codex/beta6-integration-rerun` passed all five room-level release gates:

| Room | Trace | Accuracy | Macro F1 | Metric Source | Release Gate |
| --- | --- | ---: | ---: | --- | --- |
| Bathroom | `backend/models/HK0011_jessica/Bathroom_v6_decision_trace.json` | 0.5554 | 0.6377 | `holdout_validation_two_stage_primary` | PASS |
| Bedroom | `backend/models/HK0011_jessica/Bedroom_v3_decision_trace.json` | 0.9416 | 0.6456 | `holdout_validation_single_stage_fallback` | PASS |
| Entrance | `backend/models/HK0011_jessica/Entrance_v17_decision_trace.json` | 0.9986 | 0.9990 | `holdout_validation` | PASS |
| Kitchen | `backend/models/HK0011_jessica/Kitchen_v2_decision_trace.json` | 0.7206 | 0.7180 | `holdout_validation_single_stage_fallback` | PASS |
| LivingRoom | `backend/models/HK0011_jessica/LivingRoom_v4_decision_trace.json` | 0.5448 | 0.7030 | `holdout_validation_single_stage_fallback` | PASS |

Lane-B event gates also passed for every room in those same traces, with zero unknown-rate failures and non-zero event-level recall on the room-specific event checks.

### What still failed
The same run failed at the authority/runtime layer, not the room-model layer. The latest `_run/events.jsonl` entry shows:
1. all five rooms passed; and
2. the run still failed because:
   - `BETA6_GATE_SIGNING_KEY` was missing for live authority signing
   - rollout auto-rollback could not find a fallback target for `livingroom`
   - PostgreSQL was unavailable in the current environment, which kept `pipeline_success_rate` at `0.0`
   - `RELEASE_GATE_EVIDENCE_PROFILE` was still defaulting to `pilot_stage_a`

### Market-readiness assessment
Beta 6 is not market-ready yet.

Reasons:
1. The room-model stack is now strong enough for continued shadow and limited canary preparation, but the authority path is still operationally incomplete.
2. The latest phase-6 artifact still says `active_system=beta5.5_authority`, `certification_ready=false`, and `stable_today=false`.
3. The timeline quality looks promising on this resident, but it has not yet been proven through signed live authority artifacts, real PostgreSQL-backed runs, or the required 7-14 consecutive stable-day certification window.
4. Several rooms are passing on `single_stage_fallback` metric sources rather than a uniform two-stage primary path. That is acceptable for shadow qualification, but it is not yet the kind of boring, production-proven consistency required for market claims.

### Reliability interpretation
The latest room metrics are good enough to justify continued Beta 6.1 shadow maturation:
1. `Entrance` looks production-strong on this batch.
2. `Kitchen` and `LivingRoom` look usable but are not premium-grade yet.
3. `Bathroom` and `Bedroom` are no longer blocked, but their macro F1 remains moderate rather than exceptional.

Interpretation:
1. Accuracy is not the primary quality signal because several rooms are class-imbalanced; macro F1 and lane-B event recall are more trustworthy.
2. The activity timeline is no longer obviously broken on Jessica, but it is not yet reliable enough to market as authority-grade because the end-to-end runtime/authority proof is still missing.

## Roadmap Changes

### Beta 6.1 becomes the primary release track
Change the Beta 6.1 objective from “runtime integration plus several remaining room-model fixes” to:
1. authority signing and artifact completeness
2. rollback/fallback completeness across all rooms
3. real-environment runtime success
4. signed shadow certification
5. laddered canary readiness

### Beta 6.2 becomes explicitly non-blocking
Beta 6.2 should continue only as an isolated R&D/program track. It should not be allowed to delay Beta 6.1 rollout hardening.

Allowed Beta 6.2 work:
1. timeline-native supervision experiments
2. household-state modeling
3. hard-negative training experiments
4. offline baselines, label audits, and architecture probes

Blocked from Beta 6.1 critical path:
1. any change to Beta 6.1 serving outputs
2. any change to authority signing/registry/rollout semantics
3. any training/objective change that mutates Beta 6.1 behavior while Beta 6.2 flags are off

### Resident/Home Metadata Strategy
Resident/home metadata is worth adding, but not all metadata belongs in the same layer.

High-value context for Beta 6:
1. household type (`single_resident` vs `multi_resident`)
2. helper/caregiver presence
3. home layout, room adjacency, and sensor topology

Lower-priority and higher-risk context:
1. age
2. sex

Policy:
1. Beta 6.1 may use resident/home context as typed policy/routing/gating input only.
2. Beta 6.2 may experiment with context-conditioned modeling, starting with layout/adjacency and household/helper context.
3. Age/sex must not become direct training inputs until offline ablations show clear value and fairness/privacy review is complete.

### Older roadmap items to de-prioritize
These are still useful, but no longer the primary gating path for Beta 6.1:
1. additional room-model forensic work on Jessica
2. generic “model stabilization” tasks without a concrete authority blocker
3. broad evaluation upgrades that do not unblock the current authority/runtime path

## Revised Execution Order

### Milestone 1: Beta 6.1 Authority Contract Hardening

**Purpose:** Remove the non-model blockers that currently make an all-room-pass run fail.

**Files:**
- Modify: `backend/run_daily_analysis.py`
- Modify: `backend/ml/beta6/registry/registry_v2.py`
- Modify: `backend/ml/t80_rollout_manager.py`
- Modify: `backend/ml/beta6/serving/serving_loader.py`
- Modify: `backend/.env.example`
- Test: `backend/tests/test_run_daily_analysis_beta6_authority.py`
- Test: `backend/tests/test_beta6_dynamic_gate_artifacts.py`
- Test: `backend/tests/test_beta6_registry_v2.py`
- Test: `backend/tests/test_t80_rollout_manager.py`

**Required outcomes:**
1. live Beta 6 runs fail early and clearly if signing prerequisites are not satisfied
2. fallback target resolution is deterministic for every room, including `livingroom`
3. rollout auto-rollback cannot crash just because a room lacks an already-materialized fallback pointer
4. authority logs and artifacts do not contradict themselves about pass/fail vs candidate promotion
5. `RELEASE_GATE_EVIDENCE_PROFILE` must be explicit in production-like runs

**Pass requirements:**
1. targeted tests pass
2. a local dry run produces explicit, deterministic artifact/signing/fallback diagnostics
3. no `No fallback target available for HK0011_jessica/livingroom` failure remains on the integration branch

### Milestone 2: Real-Environment Integration Proof

**Purpose:** Re-run the already-good model stack in a production-like environment.

**Required environment:**
1. `BETA6_GATE_SIGNING_KEY` set
2. PostgreSQL reachable
3. `RELEASE_GATE_EVIDENCE_PROFILE` explicitly set to the intended production/staging profile
4. no shadow-only environment shortcuts

**Run target:**
Use the exact integrated branch/worktree currently carrying the Jessica model fixes:
`codex/beta6-integration-rerun`

**Pass requirements:**
1. all five rooms pass again
2. authority passes
3. signed evaluation/rejection artifacts are written correctly
4. `phase6_stability_report.json` no longer shows `pipeline_success_rate=0.0`
5. no rollout fallback or signing errors appear in `_run/events.jsonl`

### Milestone 3: Beta 6.1 Shadow Certification

**Purpose:** Prove stability rather than one-off success.

**Requirements:**
1. Jessica must accumulate the required stable-day run count
2. at least one additional resident must also pass the same authority/runtime bar
3. no unexplained critical shadow divergence may remain open
4. rollback drills must work end-to-end

**Pass requirements:**
1. `active_system` may still remain `beta5.5_authority` during shadow
2. `certification_ready=true` only after the required consecutive-day window
3. rollout blockers are limited to policy decisions, not technical incompleteness

### Milestone 3A: Beta 6.1 Resident/Home Context Contract

**Purpose:** Introduce typed resident/home context as a system input before broader shadow/canary expansion, without perturbing the current Jessica model stack.

**Scope:**
1. household type
2. helper/caregiver presence
3. room adjacency / layout / sensor topology

**Rules:**
1. Context is used for policy selection, decoder constraints, arbitration, routing, and evaluation cohorting.
2. Context is not used as a raw training feature in Beta 6.1.
3. Missing context must fail clearly where required, rather than silently falling back to wrong assumptions.

**Candidate files:**
- Modify: `backend/db/schema.sql`
- Modify: `backend/processors/profile_processor.py`
- Modify: `backend/ml/household_analyzer.py`
- Modify: `backend/ml/home_empty_fusion.py`
- Modify: `backend/ml/beta6/sequence/transition_builder.py`
- Modify: `backend/export_dashboard.py`

**Pass requirements:**
1. resident/home context has a typed source of truth
2. single-resident vs multi-resident behavior is explicit and testable
3. layout/topology can be consumed by arbitration/decoder code without ad hoc env/config reads
4. the additional resident shadow cohort is not blocked by missing context contracts

### Milestone 4: Laddered Beta 6.1 Canaries

**Purpose:** Controlled authority transition.

**Stages:**
1. 10% canary
2. 50% ladder
3. full authority candidate

**Pass requirements:**
1. rollback contract proven
2. no unresolved authority-signing/runtime blockers
3. product/ops/clinical sign-off on the shadow evidence

### Milestone 5: Beta 6.2 Offline Upgrade Track

**Purpose:** Continue model architecture improvement without blocking Beta 6.1.

**Files / namespace guidance:**
1. keep Beta 6.2 in a distinct config/module/test namespace
2. do not alter Beta 6.1 default runtime behavior when Beta 6.2 flags are off

**Allowed deliverables before Beta 6.1 cutover:**
1. offline experiments and reports
2. benchmark comparisons
3. feature and objective prototypes

**Blocked deliverables before Beta 6.1 cutover:**
1. any default-on serving path
2. any change to the Beta 6.1 authority pipeline
3. any rollout dependency that says “Beta 6.1 cannot ship until Beta 6.2 lands”

**Recommended metadata use in Beta 6.2:**
1. first priority: layout/adjacency/sensor-topology context
2. second priority: household/helper context
3. third priority: age/sex only after explicit ablation, fairness review, and privacy review

**Recommended modeling patterns for Beta 6.2:**
1. context-conditioned decoder constraints
2. context-conditioned timeline heads
3. household-state latent modeling
4. routing/profiling paths that reduce unnecessary retries and impossible transitions

## Task-Level Plan

### Task 1: Write the authority-blocker regression tests

**Files:**
- Modify: `backend/tests/test_run_daily_analysis_beta6_authority.py`
- Modify: `backend/tests/test_beta6_registry_v2.py`
- Modify: `backend/tests/test_t80_rollout_manager.py`

**Step 1: Add tests for signing-key requirements**

Add tests that assert a live authority run with registry v2 present returns a deterministic failure report when the signing key is missing.

**Step 2: Add tests for fallback target resolution**

Add tests that assert `rollback_and_activate_fallback(...)` and rollout auto-protection do not throw `No fallback target available` when the room has a valid current or previous pointer.

**Step 3: Run the focused tests**

Run:

```bash
pytest backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_beta6_registry_v2.py backend/tests/test_t80_rollout_manager.py -q
```

Expected:
1. new tests fail before implementation

### Task 2: Implement authority-signing and fallback fixes

**Files:**
- Modify: `backend/run_daily_analysis.py`
- Modify: `backend/ml/beta6/registry/registry_v2.py`
- Modify: `backend/ml/t80_rollout_manager.py`
- Modify: `backend/.env.example`

**Step 1: Harden signing preconditions**

Make the signing-key path explicit and operationally obvious for live authority runs.

**Step 2: Harden fallback target lookup**

Make fallback selection deterministic from:
1. explicit fallback candidate
2. rollback pointer
3. current champion pointer
4. known safe legacy pointer if policy allows

**Step 3: Fix contradictory pass/fail semantics**

Ensure room candidate promotion/deferred-candidate logging is consistent with run-level authority failure semantics.

**Step 4: Run focused tests**

Run:

```bash
pytest backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_beta6_registry_v2.py backend/tests/test_t80_rollout_manager.py -q
```

Expected: PASS

### Task 3: Run the integration verification bundle

**Files:**
- Modify: none unless tests expose a real bug

**Step 1: Run compilation check**

```bash
python3 -m py_compile backend/run_daily_analysis.py backend/ml/beta6/registry/registry_v2.py backend/ml/t80_rollout_manager.py backend/ml/beta6/serving/serving_loader.py
```

**Step 2: Run the integrated Beta 6 regression slice**

```bash
pytest backend/tests/test_policy_config.py backend/tests/test_beta6_policy_config_schemas.py backend/tests/test_training.py backend/tests/test_run_daily_analysis_beta6_authority.py backend/tests/test_beta6_dynamic_gate_artifacts.py backend/tests/test_beta6_registry_v2.py backend/tests/test_t80_rollout_manager.py -q
```

Expected:
1. PASS

### Task 4: Execute the real-environment Jessica rerun

**Files:**
- Modify: none

**Step 1: Prepare the environment**

Required:
1. PostgreSQL up
2. `BETA6_GATE_SIGNING_KEY` set
3. explicit `RELEASE_GATE_EVIDENCE_PROFILE`

**Step 2: Run watcher-level retrain from the integration branch**

**Step 3: Verify artifacts**

Check:
1. all room traces
2. run `_run/events.jsonl`
3. signed authority artifacts in `backend/tmp/beta6_gate_artifacts/...`
4. `phase6_stability_report.json`

**Step 4: Record the result**

Write a review note with:
1. room outcomes
2. authority outcome
3. remaining blockers, if any

## Decision Summary

### Is Beta 6 “market” yet?
No.

Why:
1. room-model quality on Jessica is now respectable and often strong
2. but authority/runtime readiness is still incomplete
3. certification is still false
4. stable-day shadow evidence is still zero
5. the activity timeline is promising, not yet market-proven

### Is Beta 6.1 worth continuing now?
Yes. Strongly yes. It is now the shortest path to a real shippable system.

### Should Beta 6.2 continue?
Yes, but as isolated R&D only. It should not be allowed to hold Beta 6.1 hostage.
