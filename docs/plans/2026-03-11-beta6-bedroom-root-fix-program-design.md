# Beta6 Bedroom Root-Fix Program Design

## Goal

Prove whether the harmful added Bedroom dates (`2025-12-04` to `2025-12-09`) are:

1. bad / inconsistent data that should be corrected or excluded
2. a valid Bedroom behavior regime that the current training and promotion pipeline cannot absorb safely

Then fix the actual cause without widening scope prematurely.

## User-Approved Scope

- Go deep on `Bedroom` first.
- Do not start with a broad all-room forensic investigation.
- Add only lightweight cross-room instrumentation while the Bedroom root cause is being proven.
- Generalize the fix only if the Bedroom finding demonstrates a pipeline-level weakness that can plausibly affect other rooms.

## Evidence Baseline

The current investigation has already established these decision-grade facts:

- live `Bedroom_v38` used only `2025-12-10` + `2025-12-17`
- candidate `Bedroom_v40` used `2025-12-04..2025-12-10` + `2025-12-17`
- the first strong divergence appears at raw source-workbook selection
- restoring the old `v38` Bedroom sampling posture on the full `v40` source pack still fails to recover the old regime

That means:

- the main driver is upstream source lineage
- later Bedroom sampling / threshold policy changes matter, but they are not the first or dominant cause

## Approaches Considered

### Option A: containment only

Keep Bedroom on the known-good small manifest and stop.

Pros:

- fastest operationally
- protects current live behavior

Cons:

- does not explain whether the excluded dates are bad data or valid market behavior
- leaves a real risk that the same regime-shift failure will reappear later

### Option B: Bedroom-first root-cause program

Run a deep Bedroom-only forensic and ablation program, then generalize only if the finding proves systemic.

Pros:

- directly answers the real root-cause question
- keeps scope bounded
- supports a defensible market-readiness decision

Cons:

- slower than pure containment
- requires more disciplined experiment structure

### Option C: immediate all-room deep-dive

Treat this as a cross-room problem from the start.

Pros:

- maximizes systemic coverage

Cons:

- scope explosion
- delays the actual Bedroom root fix
- weakens causal discipline because the first proven failure is Bedroom

## Recommendation

Use **Option B**.

The work should be split into:

1. `Bedroom` root-cause investigation and root fix
2. lightweight all-room observability additions that make future generalization possible without prematurely changing other rooms

## Design

### 1. Reproducible Bedroom source-lineage observability

Before running more ablations, the pipeline must record exact Bedroom source manifests and per-date label topology in machine-readable outputs.

Required additions:

- exact workbook / parquet source list in training metadata
- per-source and per-date class counts before sampling
- a stable fingerprint for the Bedroom source pack used in each run

Purpose:

- remove future ambiguity about which Bedroom data actually trained a saved version
- make date-ablation comparisons deterministic and auditable

### 2. Bedroom day-ablation matrix

The core investigation should start from the known-good anchor and add the extra dates back one at a time.

Anchor:

- `2025-12-10` + `2025-12-17`

Ablation ladder:

- anchor + `2025-12-04`
- anchor + `2025-12-05`
- anchor + `2025-12-06`
- anchor + `2025-12-07`
- anchor + `2025-12-08`
- anchor + `2025-12-09`
- then cumulative adds in date order if needed

For every run, collect:

- pre-sampling label counts and shares
- post-sampling label counts and shares
- holdout macro-F1
- holdout `bedroom_normal_use` recall
- Dec 17 final macro-F1
- Dec 17 predicted class share
- Dec 17 confusion signatures:
  - `sleep -> unoccupied`
  - `unoccupied -> bedroom_normal_use`
  - `bedroom_normal_use -> unoccupied`

Purpose:

- isolate the first harmful date
- determine whether the failure comes from one specific day or from cumulative distribution drift across several days

### 3. Harmful-day segment audit

Once the first harmful day is identified, split it into smaller time blocks and inspect the blocks that move Bedroom into the bad regime.

For each block, compare against the good anchor days:

- label mix
- transition topology
- occupancy/run-length structure
- any obvious sensor sparsity or missingness anomalies
- disagreement with surrounding date patterns

Each block should be classified as:

- likely bad / inconsistent data
- likely valid alternate regime
- uncertain

Purpose:

- stop arguing abstractly about “bad days”
- localize the exact evidence that explains the regime shift

### 4. Root-fix decision gate

The investigation should not jump directly from ablation to a training tweak. It needs an explicit decision gate.

#### Branch A: harmful blocks are bad / inconsistent data

Root fix:

- correct labels if the error is recoverable
- otherwise exclude those blocks from the Bedroom pack
- document the exclusion rule explicitly

Validation:

- rerun Bedroom with the corrected / filtered pack
- confirm the failure family disappears without introducing a new one

#### Branch B: harmful blocks are valid alternate behavior

Root fix:

- treat this as a training / promotion robustness failure
- keep the valid data
- harden the pipeline to detect and reject unstable regime shifts before promotion

Validation:

- require grouped-by-date stability checks and date-slice replay checks
- confirm the new training recipe can retain acceptable Bedroom performance across both the old and new regime slices

### 5. Lightweight all-room instrumentation

While the deep investigation stays Bedroom-only, add low-risk observability that is useful across rooms:

- persist room-level source manifests in saved metadata
- persist per-date class summaries before sampling
- emit compact drift summaries for promotion review

Do **not** change training behavior for non-Bedroom rooms yet.

Purpose:

- if Bedroom exposes a systemic regime-shift weakness, the necessary observability is already present everywhere
- if Bedroom turns out to be a Bedroom-only data problem, the broader system remains stable

## Data Flow

1. corrected Jessica source workbooks / combined pack inputs
2. Bedroom source-manifest builder
3. Bedroom date-ablation runner
4. Bedroom Dec 17 replay benchmark
5. harmful-day / segment audit
6. branch decision:
   - data correction / exclusion
   - or pipeline robustness hardening
7. final validation on Bedroom

## Deliverables

### Investigation deliverables

- one date-ablation manifest and summary table
- one harmful-day segment audit report
- one final decision note stating:
  - bad data
  - valid regime shift
  - or unresolved with bounded next unknown

### Code deliverables

- reproducible Bedroom ablation runner
- reproducible harmful-day segment audit tooling
- persisted room-level source manifests in saved training metadata
- compact per-date drift summaries in promotion artifacts

## Risks And Mitigations

### Risk: the “first harmful date” is actually a multi-day interaction

Mitigation:

- run both single-day add-back and cumulative add-back ladders

### Risk: the harmful day is valid but rare

Mitigation:

- do not exclude it automatically
- require the segment audit to distinguish between invalid data and real alternate behavior

### Risk: the investigation turns into global pipeline redesign too early

Mitigation:

- keep behavior changes Bedroom-only until the Bedroom evidence proves broader applicability

### Risk: future forensic threads repeat the same provenance ambiguity

Mitigation:

- persist source manifests and date-level summaries as first-class training metadata now

## Success Criteria

This program is successful only if it produces all of the following:

1. a defensible statement about whether the harmful Bedroom dates are bad data or valid regime shift
2. a root fix that matches that diagnosis
3. Bedroom validation evidence showing the failure family is actually removed
4. enough metadata and drift reporting that the same failure mode cannot silently recur without leaving clear evidence

## Out Of Scope For This Thread

- deep forensic analysis of all other rooms
- broad retraining-policy redesign before the Bedroom diagnosis is complete
- promoting any new Bedroom model solely because it beats current macro-F1 on one replay without resolving the source-lineage question
