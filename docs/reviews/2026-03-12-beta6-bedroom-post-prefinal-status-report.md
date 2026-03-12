# Beta6 Bedroom Post-Pre-Final Status Report

## Purpose

This note is the handoff-quality Bedroom summary to use **after** pre-final testing.

It is not a replacement for the full forensic trail. It is the shortest complete explanation of:

- what Bedroom root cause was
- what was fixed
- what still is not solved
- what future Bedroom work should focus on

## Current Bedroom Status

Bedroom is now **fail-closed**, but not fully robust.

That means:

- unsafe Bedroom candidates are now visible and blocked before promotion
- production is better protected than before
- but the model still has not proven it can learn all valid Bedroom day patterns stably in one recipe

Operationally, the safer anchor remains the promoted live Bedroom line rather than any high-risk mixed-regime candidate.

## Root Cause In Plain Terms

The real Bedroom problem was **not** obviously bad data.

The earliest harmful cause was adding `2025-12-05` into the Bedroom source pack. That day contains a real alternate Bedroom behavior pattern, especially long coherent `bedroom_normal_use` periods at times where the older “good” reference days were mostly `unoccupied`.

So the root cause was:

- a valid new Bedroom pattern entered the training data
- the Bedroom training / promotion workflow was not robust to that pattern mix
- the old workflow could let a risky Bedroom candidate look acceptable on average even when it was unstable on important date slices

## What Was Disproven

The investigation ruled out these simpler explanations:

- “the problem was mainly later sampling or threshold tuning”
- “the harmful day was obviously broken / sparse / corrupted”
- “the model only breaks after a larger cumulative date interaction”

The evidence showed `2025-12-05` alone was already harmful, and its harmful slices were dense, coherent, and unflagged.

## What Is Fixed Now

### 1. Promotion safety

Bedroom candidates now persist:

- grouped-by-date stability evidence
- promotion-time drift evidence
- Bedroom-specific risk flags

High-risk Bedroom instability is now a **blocking** promotion condition, not just a warning.

### 2. Review visibility

The machine-readable review surface now exposes the same Bedroom instability evidence that used to be buried inside traces.

This means a candidate that looks decent on pooled metrics can no longer quietly pass review if it is unstable on important date slices.

### 3. Provenance visibility

The system now records:

- exact source manifests
- source fingerprints
- per-date pre-sampling label counts
- compact lineage summaries

That removes the old ambiguity about what data actually trained a saved version.

## What Is Still Open

Bedroom is **not** fully solved because the current project fixed the **control-plane** problem, not the **model-quality** problem.

What remains unsolved:

- a training recipe that can safely absorb both the old anchor regime and the valid alternate `2025-12-05` regime
- a Bedroom model that stays stable across date slices while reducing the structural `bedroom_normal_use` / `unoccupied` confusion family

So the remaining work is now clearly a model-improvement problem, not a root-cause mystery.

## Why Bedroom Is Structurally Hard

Bedroom has at least two valid behavior regimes that do not look the same.

That makes it structurally hard because:

- average metrics can hide day-level instability
- one decent replay can be misleading
- the core decision boundary between `bedroom_normal_use` and `unoccupied` is fragile when valid day patterns differ

In other words, Bedroom is not just noisy. It is heterogeneous.

## Recommendation After Pre-Final Testing

Do **not** reopen the old root-cause investigation.

Start a new, explicitly scoped Bedroom model-robustness project with this goal:

> find a Bedroom training recipe that can learn both valid regimes safely, while preserving the current fail-closed promotion safety behavior

Recommended focus areas:

1. group-aware Bedroom training and model selection by date slice, not just pooled score
2. richer temporal / contextual features for `bedroom_normal_use` vs `unoccupied`
3. stability-first Bedroom acceptance criteria
4. if needed, regime-aware Bedroom modeling rather than forcing one pooled boundary

## Do Not Regress

Any future Bedroom work should preserve these safeguards:

- keep Bedroom instability visible in saved metadata
- keep high-risk unstable Bedroom candidates blocked from promotion
- keep lightweight lineage observability available across rooms

## Bottom Line

Bedroom is now **safer**, but not yet **fully learned**.

The root-cause program succeeded because it identified the real cause and made unsafe candidates non-promotable.

The next Bedroom project should be about **robust learning across valid regimes**, not about rediscovering what went wrong.
