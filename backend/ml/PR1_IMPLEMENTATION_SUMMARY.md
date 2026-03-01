# PR-1: Unify Runtime Training Path - Implementation Summary

## Overview
Successfully implemented a unified hardened training path that routes both `train_from_files()` (watcher) and `train_and_predict()` (manual) through the same gate pipeline.

## Files Created/Modified

### New Files
1. **`ml/unified_training.py`** - Core unified training pipeline
   - `UnifiedTrainingPipeline` class with hardened gate execution
   - `UnifiedTrainingResult` dataclass for standardized results
   - `GateResult` dataclass for individual gate evidence
   - Gate stack: CoverageContract → PostGapRetention → ClassCoverage → StatisticalValidity

2. **`ml/unified_training_spec.md`** - Design specification

3. **`tests/test_unified_training_path.py`** - Feature parity tests (15 tests)

### Modified Files
1. **`ml/pipeline.py`**
   - Added `UnifiedTrainingPipeline` import
   - Added `unified_training` attribute to `UnifiedPipeline.__init__`
   - Added `_train_room_unified()` method for hardened path
   - Updated `train_and_predict()` to use unified path with legacy fallback
   - Updated `train_from_files()` to use unified path with legacy fallback

## Key Design Decisions

### 1. Gate Pipeline Order
```
1. CoverageContractGate      - Data sufficiency check
2. PostGapRetentionGate      - Data continuity check  
3. ClassCoverageGate         - Split coverage check
4. StatisticalValidityGate   - Metric reliability check
5. Training Execution        - Only if gates pass
```

### 2. Backward Compatibility
- Gates are evaluated and recorded for evidence
- Legacy training path executes even if gates fail (for backward compatibility)
- PR-2 will implement blocking behavior based on gate results
- Gate stack markers are added to all training results

### 3. Evidence Artifacts
Every training result now includes:
- `gate_stack`: List of gate execution results with timestamps
- `gate_reasons`: List of any gate failures
- `gate_pass`: Boolean indicating if all gates passed
- `rejection_artifact`: Structured artifact when gates fail

## Test Coverage

### 15 New Tests in `test_unified_training_path.py`
- `test_unified_pipeline_initialization` - Pipeline setup
- `test_coverage_gate_execution` - Gate evaluation
- `test_successful_gate_stack_execution` - Full gate pipeline
- `test_rejection_artifact_on_failure` - Artifact generation
- `test_gate_result_serialization` - JSON serialization
- `test_unified_result_serialization` - Result serialization
- `test_both_paths_include_gate_stack_in_metadata` - Feature parity
- `test_both_paths_produce_identical_gate_order` - Consistency
- `test_both_paths_generate_rejection_artifacts_on_failure` - Both paths
- `test_gate_failure_stage_tracking` - Failure tracking
- `test_progress_callback_invocation` - Progress reporting
- `test_gate_stack_contains_timestamps` - Audit trail
- `test_gate_stack_contains_details` - Debug info
- `test_gate_stack_json_serialization` - Persistence
- `test_create_unified_training_result_convenience` - Helper function

### Total Test Suite: 203 tests passing

## Evidence Artifact Format

### Gate Stack Example
```json
{
  "gate_stack": [
    {
      "gate_name": "CoverageContractGate",
      "passed": true,
      "timestamp": "2026-02-16T10:00:00Z",
      "details": {"observed_days": 10, "required_days": 8}
    },
    {
      "gate_name": "PostGapRetentionGate",
      "passed": true,
      "timestamp": "2026-02-16T10:00:01Z",
      "details": {"retained_ratio": 0.85, "segments": 3}
    }
  ],
  "gate_pass": true,
  "gate_reasons": []
}
```

### Rejection Artifact Example
```json
{
  "run_id": "uuid",
  "elder_id": "test_elder",
  "overall_passed": false,
  "coverage_reasons": [...],
  "executive_summary": "...",
  "top_priority_fix": "..."
}
```

## Usage

Both entrypoints now automatically use the unified path:

```python
# Watcher path (run_daily_analysis.py)
results, metrics = pipeline.train_from_files(
    aggregate_files, 
    elder_id,
    defer_promotion=True
)

# Manual path (UI/corrections)
corrected_results, trained_rooms = pipeline.train_and_predict(
    file_path,
    elder_id,
    progress_callback=callback
)
```

Both produce identical gate stack markers in their metadata.

## Next Steps (PR-2)
- Wire hardening gates into live promotion flow
- Implement blocking behavior when gates fail
- Add why_rejected.json artifact generation
- Persist gate reason codes in training history metadata
