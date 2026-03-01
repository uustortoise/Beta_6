# Beta 5.5 Labeling Guide

## 1. Purpose
This guide defines how to label activities so training data matches real resident behavior and produces a reliable timeline.

## 2. Core Labeling Principle
Label the resident's true state, not raw sensor spikes.

Practical meaning:
- Entry motion can start an occupancy episode.
- Low motion while seated/resting does not automatically mean `unoccupied`.
- Use explicit exit evidence before ending an occupancy episode.

## 3. Priority Rule
When conflicts exist, trust this order:
1. Human correction (golden sample)
2. Validated train file label
3. Model prediction

## 4. Standard Label Semantics
Use room-appropriate labels from the current registry/dropdown. In day-to-day corrections, these are the critical distinctions:

- `unoccupied`: resident not in that room.
- `[room]_normal_use`: resident in room without a more specific activity.
- `sleep`: bedroom sleep episode.
- `shower`: active water-use interval in bathroom.
- `inactive`: resident present but no distinct task.
- `unknown` (or `low_confidence` where configured): state cannot be determined reliably.

## 5. Start/End Rules

### Start a room episode when
- first clear presence signal appears in that room (motion/context evidence), or
- a transition from another room indicates entry.

### End a room episode when
- clear exit evidence appears (strong motion in another room, explicit entrance out signal, or consistent transition evidence), or
- confidence becomes insufficient and the interval should be marked `unknown`.

Do not end episodes only because motion drops for a short passive period.

## 6. LivingRoom Passive Occupancy Policy (Critical)
This is the main mismatch zone in Beta 5.5.

Example:
- 10:00 enters LivingRoom (high motion)
- 10:01-10:30 watching TV (low motion)

Correct labeling:
- entire 10:00-10:30 should remain `livingroom_normal_use` unless there is evidence of exit.

Incorrect labeling:
- switching to `unoccupied` immediately after motion drops.

## 7. Night Policy (22:00-06:00)
At night, apply a high evidence bar before labeling LivingRoom occupancy.

Recommended rule:
- if bedroom sleep continuity is strong and no credible exit is observed, do not force LivingRoom occupancy.
- if evidence is ambiguous, use `unknown` rather than forcing a wrong room label.

## 8. Unsensored-Room Reality
If resident is likely in an unsensored area, avoid forced room assignment.

Use `unknown` when:
- all sensed rooms are quiet,
- no reliable cross-room transition exists,
- and any room assignment would be speculative.

Runtime note:
- current runtime default fallback remains `low_confidence`.
- optional scoped runtime policy can emit `unknown` in targeted ambiguity windows when `RUNTIME_UNKNOWN_*` flags are enabled.

## 9. Room-Specific Practical Guidance
- Bedroom: preserve sleep continuity; split only on real wake/exit signals.
- LivingRoom: preserve passive occupancy after entry until real exit evidence.
- Kitchen/Bathroom: shorter episodes are normal; transitions can be faster.
- Entrance: transient by design; avoid long hold labels.

## 10. Label Quality Checklist Before Save
1. Is the start timestamp aligned to first credible evidence?
2. Is the end timestamp aligned to credible exit evidence?
3. Did I avoid using sensor state alone (for example light left on) as occupancy proof?
4. Did I use `unknown` where evidence is insufficient?
5. Does this correction improve timeline coherence across adjacent rooms?

## 11. Operational Workflow
1. Correct in Streamlit Correction Studio.
2. Save corrections (golden samples).
3. Re-run training/evaluation flow.
4. Validate with smoke + matrix before promotion decisions.

## 12. Related Docs
- E2E technical flow: `ml_adl_e2e_technical_flow.md`
- Ops manual: `operation_manual.md`
- Golden sample SOP: `golden_sample_harvesting.md`
