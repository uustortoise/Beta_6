# Beta6.2 Grouped-Date Reliability Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the Beta6.2 grouped-date training surface so discontinuous day packs can be trained and evaluated reproducibly without fallback to legacy flattened training behavior.

**Architecture:** Keep the grouped-date work isolated to the Beta6.2 grouped-date modules and CLIs. First harden the report/manifest and artifact-resolution contracts, then make grouped-date preprocessing and sequence creation segment-aware inside each split so train/eval no longer fabricate continuity across date boundaries.

**Tech Stack:** Python, pandas, parquet, existing Beta6 training/evaluation internals, pytest

---

### Task 1: Harden The Grouped-Date Report Contract

**Files:**
- Modify: `backend/ml/beta6/grouped_date_supervised.py`
- Modify: `backend/tests/test_beta62_grouped_date_supervised.py`
- Test: `backend/tests/test_beta62_grouped_date_fit_eval.py`

**Step 1: Write the failing test**

Add a test that `run_grouped_date_supervised(...)` emits enough manifest metadata for downstream fit/eval:

```python
def test_grouped_date_supervised_report_embeds_full_manifest_contract(tmp_path: Path):
    report = run_grouped_date_supervised(manifest, artifact_dir=tmp_path / "artifacts")
    assert report["manifest"]["resident_id"] == "HK0011_jessica"
    assert report["manifest"]["target_rooms"] == ["Bathroom", "Bedroom"]
    assert report["manifest"]["sequence_length_by_room"]["bathroom"] == 30
```

**Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_beta62_grouped_date_supervised.py::test_grouped_date_supervised_report_embeds_full_manifest_contract -q`

Expected: FAIL because the current report only embeds `schema_version`, `segments`, and `notes`.

**Step 3: Write minimal implementation**

Update the grouped-date supervised report payload so `manifest` contains the normalized manifest contract needed downstream:

- `schema_version`
- `resident_id`
- `target_rooms`
- `sequence_length_by_room`
- `segments`
- `notes`

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest backend/tests/test_beta62_grouped_date_supervised.py::test_grouped_date_supervised_report_embeds_full_manifest_contract -q
pytest backend/tests/test_beta62_grouped_date_supervised.py backend/tests/test_beta62_grouped_date_fit_eval.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/beta6/grouped_date_supervised.py backend/tests/test_beta62_grouped_date_supervised.py backend/tests/test_beta62_grouped_date_fit_eval.py
git commit -m "fix: embed full grouped-date manifest contract in report"
```

### Task 2: Fix Deferred-Candidate Artifact Resolution

**Files:**
- Modify: `backend/ml/beta6/grouped_date_fit_eval.py`
- Modify: `backend/tests/test_beta62_grouped_date_fit_eval.py`

**Step 1: Write the failing tests**

Add tests for both root causes:

```python
def test_grouped_date_fit_eval_accepts_supervised_report_only_input(tmp_path: Path):
    report = run_grouped_date_fit_eval(
        supervised_report_path=report_path,
        artifact_dir=artifact_dir,
        candidate_namespace="candidate_ns",
        backend_dir=backend_dir,
    )
    assert report["manifest"]["resident_id"] == "HK0011_jessica"


def test_grouped_date_fit_eval_resolves_latest_saved_candidate_when_current_version_is_zero(tmp_path: Path):
    # versions.json contains current_version=0 but a latest saved version exists
    report = run_grouped_date_fit_eval(...)
    assert report["room_results"]["livingroom"]["fit_result"]["saved_version"] == 6
    assert report["room_results"]["livingroom"]["candidate_artifact_paths"]["model"].endswith("LivingRoom_v6_model.keras")
```

**Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q`

Expected: FAIL on report-only input and/or reused-namespace artifact resolution.

**Step 3: Write minimal implementation**

In `grouped_date_fit_eval.py`:

- normalize `resident_id`, `target_rooms`, and `sequence_length_by_room` from report-embedded manifest when `--supervised-report` is used
- stop trusting `metrics["saved_version"]` blindly after `train_room(...)`
- resolve the saved candidate version from `versions.json` using:
  - the reported `saved_version` if artifacts exist
  - otherwise the newest saved version greater than the room’s pre-fit max version
  - otherwise the latest saved version on disk as a fail-safe

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q
pytest backend/tests/test_beta62_grouped_date_fit_eval.py backend/tests/test_beta62_grouped_date_supervised.py backend/tests/test_training.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/beta6/grouped_date_fit_eval.py backend/tests/test_beta62_grouped_date_fit_eval.py
git commit -m "fix: harden grouped-date fit eval contract and artifact resolution"
```

### Task 3: Make Grouped-Date Preprocessing Segment-Aware

**Files:**
- Modify: `backend/ml/beta6/grouped_date_fit_eval.py`
- Modify: `backend/ml/beta6/grouped_date_supervised.py`
- Modify: `backend/tests/test_beta62_grouped_date_fit_eval.py`
- Modify: `backend/tests/test_beta62_grouped_date_supervised.py`

**Step 1: Write the failing tests**

Add tests that encode the real root cause:

```python
def test_grouped_date_fit_eval_preprocesses_train_segments_independently(tmp_path: Path):
    report = run_grouped_date_fit_eval(...)
    assert report["room_results"]["bathroom"]["lineage"]["train"]["dates"] == ["2025-12-04", "2026-03-02"]
    assert report["room_results"]["bathroom"]["fit_result"]["preprocess_mode"] == "segment_aware"


def test_grouped_date_fit_eval_does_not_create_cross_segment_sequences(tmp_path: Path):
    report = run_grouped_date_fit_eval(...)
    assert report["room_results"]["bathroom"]["fit_result"]["cross_segment_sequence_count"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q`

Expected: FAIL because current fit/eval preprocesses each split as one combined timeline.

**Step 3: Write minimal implementation**

Implement a grouped-date-only segmented preprocessing/sequence path:

- group each split by segment lineage columns:
  - `__segment_role`
  - `__segment_date`
  - `__segment_split`
- preprocess each segment independently
- fit scalers across the union of processed train segments
- create training/eval sequences inside each processed segment only
- concatenate segment-local sequences afterward, preserving per-sequence lineage
- keep legacy training paths unchanged

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest backend/tests/test_beta62_grouped_date_fit_eval.py -q
pytest backend/tests/test_beta62_grouped_date_fit_eval.py backend/tests/test_beta62_grouped_date_supervised.py backend/tests/test_training.py -q
```

Expected: PASS

**Step 5: Commit**

```bash
git add backend/ml/beta6/grouped_date_fit_eval.py backend/ml/beta6/grouped_date_supervised.py backend/tests/test_beta62_grouped_date_fit_eval.py backend/tests/test_beta62_grouped_date_supervised.py
git commit -m "fix: make grouped-date fit eval segment-aware"
```

### Task 4: Re-run The Jessica Safe-6 Experiment On The Hardened Path

**Files:**
- Modify: `docs/reviews/2026-03-14-beta62-jessica-grouped-safe6-exploratory-result.md`
- Modify: `dev_history.log`

**Step 1: Rebuild grouped-date artifacts**

Run:

```bash
PYTHONPATH='.:backend' python3 backend/scripts/run_beta62_grouped_date_supervised.py \
  --manifest <safe6_manifest.json> \
  --output <grouped_date_supervised_report.json> \
  --artifact-dir <prepared_splits_dir>
```

Expected: report and split parquets regenerated on the hardened path.

**Step 2: Run grouped-date fit/eval**

Run:

```bash
PYTHONPATH='.:backend' python3 backend/scripts/run_beta62_grouped_date_fit_eval.py \
  --supervised-report <grouped_date_supervised_report.json> \
  --artifact-dir <prepared_splits_dir> \
  --candidate-namespace <fresh_candidate_namespace> \
  --output <grouped_date_fit_eval_report.json>
```

Expected: candidate fit/eval completes without recovery workarounds.

**Step 3: Run replay/benchmark recovery only if the runner still does not emit it**

If Dec 17 replay is not integrated yet, run the existing saved-candidate benchmark harness against the fresh namespace and write:

- `dec17/final/comparison/summary.json`
- `holdout_summary.json`
- `exploratory_summary.json`

**Step 4: Compare against signed-off baseline**

Required comparisons:

- Dec 17 overall accuracy / macro-F1
- Bathroom, Bedroom, Kitchen, LivingRoom room-level Dec 17 metrics
- excluded-date March holdout summary

**Step 5: Commit**

```bash
git add docs/reviews/2026-03-14-beta62-jessica-grouped-safe6-exploratory-result.md
git commit -m "docs: record hardened grouped-date safe6 result"
```

### Task 5: Final Reliability Gate

**Files:**
- Modify: `docs/reviews/2026-03-14-beta62-grouped-date-fit-eval-runner.md`
- Modify: `docs/reviews/2026-03-14-beta62-grouped-date-supervised-path.md`
- Modify: `dev_history.log`

**Step 1: Verify the hardening slice**

Run:

```bash
pytest backend/tests/test_beta62_grouped_date_supervised.py \
       backend/tests/test_beta62_grouped_date_fit_eval.py \
       backend/tests/test_training.py -q
python3 -m py_compile \
  backend/ml/beta6/grouped_date_supervised.py \
  backend/ml/beta6/grouped_date_fit_eval.py \
  backend/scripts/run_beta62_grouped_date_supervised.py \
  backend/scripts/run_beta62_grouped_date_fit_eval.py
```

Expected: PASS

**Step 2: Write final reliability decision**

Document whether Beta6.2 grouped-date path is now:

- trusted for future grouped-date experiments
- still experimental but no longer broken
- or still blocked by deeper architectural issues

**Step 3: Commit**

```bash
git add docs/reviews/2026-03-14-beta62-grouped-date-fit-eval-runner.md docs/reviews/2026-03-14-beta62-grouped-date-supervised-path.md
git commit -m "docs: record beta62 grouped-date reliability status"
```
