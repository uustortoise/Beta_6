# Livingroom Reliability End-to-End Design

**Date:** 2026-03-08

**Goal:** Fix Livingroom model promotion so collapsed or misleading fallback models do not ship, and extend the Streamlit review surfaces so operators can see truthful model quality, compare training truth against prediction output, and review uncertain blocks efficiently.

## Problem

Livingroom is currently suffering from two separate but connected failures:

1. The training path can still select a bad model even when usable signal exists in the data.
2. The UI does not make it easy to see where predictions diverge from truth or whether displayed F1 / accuracy numbers represent walk-forward quality, runtime uncertainty, or raw training logs.

The prior investigation showed that the low Livingroom F1 is not best explained by missing signal in the training file. The stronger explanation is:

- collapse in the selected model path
- fallback selection favoring the wrong runtime path
- weak seed-panel ranking that can still prefer a gate-failing run

The end-to-end fix must therefore do two things:

- fail honestly when the model is unreliable
- expose uncertainty and correction opportunities in the UI

## Design Principles

1. Do not promote collapsed Livingroom models.
2. Prefer reliable path selection over flattering score deltas.
3. Keep holdout and walk-forward evaluation honest.
4. Use `unknown` / review-needed as an operational safety valve, not as fake success.
5. Reuse the existing Correction Studio and dashboard flows instead of inventing a parallel product surface.

## Scope

### In Scope

- tighten two-stage final path selection so collapse and no-regress outrank score-only fallback
- tighten multi-seed selection so non-collapsed, gate-passing candidates outrank gate-failing candidates
- reuse the evaluated winning seed result instead of retraining it from scratch
- expose failure reasons in debug metadata so bad Livingroom runs are diagnosable
- add training-vs-prediction compare timelines to Correction Studio
- mark unknown / low-confidence / mismatch / corrected spans in the prediction timeline
- make Streamlit F1 / accuracy / data-readiness displays source-aware and aligned to the 14-day target

### Out of Scope

- redesigning the overall model architecture
- relabeling historical holdout data
- rewriting the Streamlit dashboards from scratch

## Model-Side Design

### 1. Two-Stage Final Path Selection Must Fail Closed

The active logic in [backend/ml/training.py](/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/livingroom-reliability-e2e/backend/ml/training.py) can still land on `single_stage_fallback_score` when the single-stage path wins only on gate-aligned score. That is the wrong priority for Livingroom reliability.

New rule:

- prefer non-collapsed over collapsed
- prefer gate/no-regress passing over failing
- then compare gate-aligned score
- then compare macro F1
- if neither candidate is acceptable, record a fail-closed source and keep runtime on the safer path instead of silently presenting a flattering fallback

The selection metadata should explicitly include whether the chosen path passed the hard reliability checks.

### 2. Multi-Seed Panel Must Rank Reliability Before Score

The multi-seed panel currently soft-penalizes gate failure and retrains the selected seed from scratch. Both are reliability risks.

New rule:

- candidates are ranked by:
  - non-collapsed
  - gate-pass
  - no-regress pass
  - gate-aligned score
  - macro F1
  - seed tie-break
- the selected candidate should carry its already-evaluated metrics through to the final return path instead of launching a second fresh training run for the same seed

If every candidate fails reliability checks, the result should still record the best available candidate, but the debug payload must make the failure visible so promotion can be blocked upstream.

### 3. Debugging Metadata Must Explain the Failure

The seed panel and two-stage selection payloads should tell the truth:

- selected source
- selected seed
- whether selected candidate passed gate/no-regress
- whether it collapsed
- whether selection failed closed
- why fallback happened

This is necessary for both operator trust and future regression debugging.

## UI / Ops Design

### Correction Studio

Extend [backend/app/pages/1_correction_studio.py](/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/livingroom-reliability-e2e/backend/app/pages/1_correction_studio.py) with a compare mode:

- top timeline: training / corrected truth
- bottom timeline: prediction / runtime output
- prediction overlays:
  - unknown
  - low confidence
  - mismatch against truth
  - corrected span

Selected spans continue using the current correction queue and sensor context flow.

### Ops Surfaces

Update the dashboard surfaces so every displayed metric clearly communicates its source:

- Walk-Forward Quality
- Runtime Reliability
- Raw Training Run Stats
- Data Readiness

The sample collection target should be configurable and default to 14.

## Testing Strategy

Use TDD for both tracks:

- training selection tests in [backend/tests/test_training.py](/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/livingroom-reliability-e2e/backend/tests/test_training.py)
- service/UI tests in [backend/tests/test_ui_services.py](/Users/dickson/DT/DT_development/Development/Beta_6/.worktrees/livingroom-reliability-e2e/backend/tests/test_ui_services.py)

Add tests for:

- two-stage selection preferring non-collapsed, no-regress-safe candidates
- explicit fail-closed metadata when both paths are unreliable
- multi-seed ranking preferring gate-passing candidates
- multi-seed path reusing evaluated winning metrics
- training timeline retrieval
- compare annotations
- source-aware dashboard payloads
- 14-day target behavior

## Success Criteria

- Livingroom can no longer silently ship a collapsed score-favored fallback path.
- Multi-seed selection reflects reliability first, not raw score only.
- Correction Studio shows stacked training-vs-prediction timelines with uncertainty overlays.
- Streamlit F1 / accuracy / readiness surfaces are tied to their real sources.
- The targeted training and UI test suites pass after the change.
