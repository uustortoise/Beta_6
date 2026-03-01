# Beta 5.5 Algorithmic Logic (Conceptual)

## 1. Scope
This document explains the model and decision logic at a conceptual level.

Authoritative implementation details are in:
- `ml_adl_e2e_technical_flow.md`
- `backend/ml/*`
- `backend/scripts/run_event_first_*.py`

## 2. Core Modeling Structure
Beta 5.5 uses a room-aware sequence modeling pipeline with:
1. standardized 10-second sensor timeline,
2. sequence feature construction,
3. CNN+Transformer-based activity modeling (current backbone path),
4. calibration and gating policies for release safety.

## 3. Why Sequence Modeling
Room activity is temporal. Single-window classification misses context.

Sequence modeling helps with:
- transition boundaries,
- routine continuity (sleep/rest patterns),
- short interruptions within longer episodes.

## 4. Timeline Construction Logic
Window-level outputs are converted into timeline segments by:
1. consolidating adjacent compatible labels,
2. applying room label validation,
3. preserving key continuity rules (notably sleep),
4. writing final segments for UI/API consumption.

## 5. Evaluation Logic
Release and tuning decisions use event-first evaluation:
1. validate incoming label pack,
2. diff label changes,
3. smoke test on targeted day,
4. full matrix across seeds/splits,
5. go/no-go by policy config.

## 6. Known Constraint
LivingRoom passive occupancy remains separability-limited in low-motion periods. Current release policy reflects this reality while preserving non-regression protections for timeline quality.

## 7. References
- E2E technical flow: `ml_adl_e2e_technical_flow.md`
- Labeling policy: `labeling_guide.md`
- Ops manual: `operation_manual.md`
