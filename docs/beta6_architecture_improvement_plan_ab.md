# Beta 6 Architecture Improvement Plan (Driven by Jessica A/B)

## Scope
Resident: `HK001_jessica`  
Window: day `7-10`  
Seed: `22`

## A/B Evidence Snapshot
Compared variants:
- Baseline anchor: `/tmp/beta6_sim_baseline_d7_10_seed22.json`
- Full promotion bundle: `/tmp/beta6_sim_promoted_d7_10_seed22.json`
- Extra semantic variant: `/tmp/beta6_sim_lr_semantic_align_v3_d7_10_seed22.json`
- Extra cross-room variant: `/tmp/beta6_sim_lr_cross_room_presence_v2_d7_10_seed22.json`

Observed:
- Baseline is best overall on Jessica.
- `LivingRoom MAE` worsens materially in all promoted variants:
  - baseline `88.39`
  - full bundle `242.95`
  - cross-room v2 `187.54`
  - semantic v3 `1007.77`
- Hard-gate pass also regresses (`4/5` -> `3/5`).

Conclusion from A/B:
- Do not promote decoder/fusion stack as primary yet.
- Fix base probability quality and gating semantics first.

---

## Architecture Decision
Apply a **base-signal-first** roadmap:
1. Fix activity-head gradient contamination (unoccupied dominance).
2. Make low-support gates non-blocking by default in short windows.
3. Promote episode-level metrics into Beta6 room/run decisions.
4. Only then enable HMM/CRF + cross-room runtime in controlled rollout.

---

## Phase Plan

## Phase 0 (Immediate Safety Freeze)
### Goal
Prevent current regressions from becoming primary runtime behavior.

### Actions
- Keep `ENABLE_TIMELINE_MULTITASK=false` in production.
- Keep `ENABLE_BETA6_HMM_RUNTIME=false` and `ENABLE_BETA6_CRF_RUNTIME=false` in production.
- Keep current event-first promotion gated by existing hard gates.

### Exit Criteria
- No production runtime path changes until Phase 1+2 pass on A/B.

---

## Phase 1 (Core Model Fix: Activity-Head Loss Mask)
### Goal
Stop activity head from learning “mostly unoccupied” shortcut.

### Primary Code Change
File: `backend/ml/training.py` around `3959-3969`.

Current behavior:
- `activity_logits` sample weights apply to all windows.

Target behavior:
- Mask activity loss on `unoccupied` (and `unknown`) windows, with optional floor.

Implementation shape:
```python
# derive label ids from encoder classes
unoccupied_id = ...
unknown_id = ...

mask = (activity_labels_train != unoccupied_id).astype(np.float32)
if unknown_id is not None:
    mask *= (activity_labels_train != unknown_id).astype(np.float32)

floor = float(os.getenv("TIMELINE_ACTIVITY_UNOCCUPIED_WEIGHT_FLOOR", "0.0"))
floor = float(np.clip(floor, 0.0, 1.0))

effective_mask = mask + (1.0 - mask) * floor
activity_sample_weight = base_weight_by_sample * timeline_native_weights * effective_mask
sample_weight_fit["activity_logits"] = activity_sample_weight
```

### Instrumentation Additions
Persist into room metrics:
- `activity_loss_mask_coverage`
- `activity_loss_mask_floor`
- `activity_effective_weight_mean_unoccupied`
- `activity_effective_weight_mean_occupied`

### A/B
- A: current code
- B1: mask with floor `0.0`
- B2: mask with floor `0.05`

### Success Criteria
- LivingRoom macro-F1 and occupied-F1 non-regression vs A.
- Bedroom sleep MAE non-regression.
- No occupancy-head collapse (occupied recall drop <= 0.02 absolute).

---

## Phase 2 (Gate Semantics Alignment)
### Goal
Stop low-support scarcity from acting like hard model failure.

### Actions
1. Release evidence profile for short windows:
- Use `pilot_stage_b` for <=14-day windows (`min_validation_class_support=8`, `min_recall_support=8`).
- Keep production profile for mature windows.

2. Lane-B event gates:
- Set tier gate minimum support to `10` for short-window runs.
- Keep low-support as `NOT_EVALUATED` (already implemented), avoid blocking on scarcity.

3. Critical label scope:
- Keep block-critical labels minimal (`sleep`, `unoccupied`, room normal-use).
- Keep infrequent ADLs on watchlist-only behavior in short-window stages.

### Success Criteria
- Gate failures correspond to real quality regressions, not purely support scarcity.
- Reduced false blocking on sparse events with unchanged safety-critical checks.

---

## Phase 3 (Episode-Level Evaluation in Beta6 Gate Path)
### Goal
Evaluate timeline quality where it matters: episodes/boundaries, not only windows.

### Actions
Wire episode metrics into Beta6 evaluation/gating path:
- Compute episodes from labels (`labels_to_episodes`).
- Compute timeline metrics (`compute_timeline_metrics`).
- Add boundary precision/recall/F1.
- Feed `duration_mae_minutes` + `fragmentation_rate` to `gate_engine` timeline hard gates.

Target files:
- `backend/ml/beta6/evaluation/evaluation_engine.py`
- `backend/ml/beta6/registry/gate_engine.py`
- (reuse) `backend/ml/timeline_metrics.py`

### Success Criteria
- Gate decisions include episode-level evidence.
- Fragmented but window-accurate predictions are correctly penalized.

---

## Phase 4 (Decoder/Fusion Re-Enablement, Shadow-First)
### Goal
Enable sequence/runtime decoders only after base logits and episode quality are healthy.

### Steps
1. Enable HMM runtime in shadow cohort only.
2. Compare pre/post on:
- duration MAE
- fragmentation
- LR occupied precision/recall balance
3. If non-regressive, canary enable by resident cohort.
4. Enable CRF only behind explicit canary flag.
5. Reintroduce cross-room fusion with tuned thresholds from successful canary slices.

### Success Criteria
- No hard-gate regression.
- No repeat of LR over-activation seen in prior A/B.

---

## Execution Order (Strict)
1. Phase 1 code + A/B
2. Phase 2 gate profile alignment + A/B
3. Phase 3 episode metrics into Beta6 gate engine + A/B
4. Phase 4 runtime decoder rollout

Any phase failing criteria blocks the next phase.

---

## Risks and Mitigations
- Risk: backbone underfits with hard mask.
  - Mitigation: fallback floor `0.05` and compare against floor `0.0`.

- Risk: reduced strictness allows weak models through.
  - Mitigation: keep safety-critical collapse checks and timeline hard gates blocking.

- Risk: decoder rollout reintroduces LR over-triggering.
  - Mitigation: shadow-first + canary gating + per-room rollback switch.

---

## Deliverables for Next Iteration
- [ ] Phase 1 patch in `training.py`
- [ ] Unit tests for masked weighting behavior
- [ ] A/B report (`A`, `B1`, `B2`) on Jessica + at least one additional resident
- [ ] Gate profile update for short-window stages
- [ ] Episode metrics integrated into Beta6 report payload
