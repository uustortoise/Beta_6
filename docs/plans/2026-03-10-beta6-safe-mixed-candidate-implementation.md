# Beta6 Safe Mixed Candidate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Assemble and verify a Jessica mixed candidate that keeps `Bedroom_v36`, adopts `LivingRoom_v40`, and is ready for integration if the final replay and fresh-load checks pass.

**Architecture:** Clone the leading no-downsample candidate namespace, use the registry rollback path to set Bedroom back to `v36`, then replay the confirmed corrected Dec 17 workbook through the candidate using the existing Jessica benchmark harness. Finish with a clean model-load sanity check that exercises the same registry/runtime path production uses.

**Tech Stack:** Python 3, `UnifiedPipeline`, legacy model registry helpers, parquet/Excel evaluation artifacts, local benchmark harness in `tmp/`

---

### Task 1: Record the approved safe-path plan

**Files:**
- Create: `docs/plans/2026-03-10-beta6-safe-mixed-candidate-design.md`
- Create: `docs/plans/2026-03-10-beta6-safe-mixed-candidate-implementation.md`

**Step 1: Save the design record**

Create `docs/plans/2026-03-10-beta6-safe-mixed-candidate-design.md` with the approved safe-path rationale, the assembly base (`HK0011_jessica_candidate_nodownsample_20260310T132301Z`), and the verification/integration gates.

**Step 2: Save the execution plan**

Create `docs/plans/2026-03-10-beta6-safe-mixed-candidate-implementation.md` with the exact namespace, benchmark source workbook, and verification steps below.

**Step 3: Note the dirty-worktree constraint**

Do not commit these docs in this session because the worktree already contains unrelated in-progress changes.

### Task 2: Assemble the safe candidate namespace

**Files:**
- Read: `backend/models/HK0011_jessica_candidate_nodownsample_20260310T132301Z/*`
- Read: `backend/ml/legacy/registry.py`
- Create: `backend/models/HK0011_jessica_candidate_safev36_20260310T141724Z/*`

**Step 1: Clone the no-downsample candidate namespace**

Run:

```bash
rsync -a backend/models/HK0011_jessica_candidate_nodownsample_20260310T132301Z/ \
  backend/models/HK0011_jessica_candidate_safev36_20260310T141724Z/
```

Expected: the new namespace contains `Bathroom_v35`, `Bedroom_v37`, `Entrance_v26`, `Kitchen_v27`, and `LivingRoom_v40` plus their latest aliases.

**Step 2: Roll Bedroom back to `v36` using the registry helper**

Run a Python entrypoint that instantiates the registry and calls:

```python
registry.rollback_to_version(
    elder_id="HK0011_jessica_candidate_safev36_20260310T141724Z",
    room_name="Bedroom",
    version=36,
)
```

Expected: `Bedroom_versions.json` reports `current_version=36` and the unversioned Bedroom aliases match `Bedroom_v36_*`.

**Step 3: Verify all room current versions**

Check `*_versions.json` in the new namespace and confirm:
- Bathroom `35`
- Bedroom `36`
- Entrance `26`
- Kitchen `27`
- LivingRoom `40`

### Task 3: Run the final Dec 17 replay benchmark

**Files:**
- Read: `tmp/jessica_activity_confidence_benchmark.py`
- Read: `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`
- Create: `tmp/jessica_17dec_eval_candidate_safev36_20260310T141724Z/final/*`

**Step 1: Run the benchmark harness**

Run:

```bash
python3 tmp/jessica_activity_confidence_benchmark.py \
  --elder-id HK0011_jessica_candidate_safev36_20260310T141724Z \
  --source-file "/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx" \
  --output-dir tmp/jessica_17dec_eval_candidate_safev36_20260310T141724Z/final
```

Expected: fresh `comparison/summary.json`, per-room merged parquet files, and raw prediction parquet files.

**Step 2: Compare against the validated mixed baseline**

Compare the new `comparison/summary.json` against:

```text
tmp/jessica_17dec_eval_candidate_mixed_20260310T090755Z/final_v3/comparison/summary.json
```

Expected: Bedroom stays near the prior safe anchor while LivingRoom improves over `0.3856`.

### Task 4: Run the fresh-load/runtime sanity check

**Files:**
- Read: `backend/ml/legacy/registry.py`
- Read: `backend/ml/pipeline.py`
- Create: `tmp/jessica_17dec_eval_candidate_safev36_20260310T141724Z/load_sanity.json`

**Step 1: Instantiate a clean pipeline and load the candidate namespace**

Run a Python entrypoint that:
- creates `UnifiedPipeline()`
- calls `pipeline.registry.load_models_for_elder("HK0011_jessica_candidate_safev36_20260310T141724Z", pipeline.platform)`
- records the loaded rooms
- records `current_version` for the five target rooms
- records whether Bathroom is present in `platform.two_stage_core_models`

Expected: all five rooms load, Bathroom has a two-stage runtime object, and the current versions match the safe candidate definition.

**Step 2: Save the sanity payload**

Write the load summary to `tmp/jessica_17dec_eval_candidate_safev36_20260310T141724Z/load_sanity.json`.

### Task 5: Document the result and integrate only if clean

**Files:**
- Modify: `docs/reviews/2026-03-10-beta6-bedroom-livingroom-retrain-blockers.md`
- Modify: `/Users/dickson/DT/DT_development/Development/Beta_6/dev_history.log`

**Step 1: Update the blocker review**

Add a short execution update describing the safe mixed candidate namespace, the benchmark result, and the fresh-load outcome.

**Step 2: Update the development log**

Append a UTC-stamped entry with:
- the namespace created
- the benchmark source workbook
- the benchmark metrics
- the fresh-load sanity result
- the integration decision

**Step 3: Integrate only if both checks are clean**

If the replay and load sanity both pass, promote/integrate the candidate using the minimal room-wise operation consistent with the verified versions. If either check fails, stop short of integration and document the blocker precisely.
