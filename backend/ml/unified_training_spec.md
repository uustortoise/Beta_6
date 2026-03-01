# PR-1: Unified Training Path Specification

## Goal
Route all training (watcher `train_from_files` and manual `train_and_predict`) through a single hardened gate pipeline.

## Beta 6 RunSpec v1 Contract Freeze
1. `RunSpec v1` is enforced with strict nested schema validation in:
   - `backend/ml/beta6/contracts/run_spec.py`
2. Hash policy:
   - `schema_hash`: hashes the published schema descriptor (field contract freeze)
   - `run_spec_hash`: hashes a canonicalized run spec payload while excluding volatile `run_id`
3. Hash policy version:
   - `run_spec_hash_policy_v1`
4. Published `RunSpec v1` schema hash:
   - `sha256:677b27b537d9fcff202a84659fc6031101597dcd3ae3b2bf1f21e088e33208f6`
5. Change management:
   - Any schema field addition/removal/rename must increment schema version and update associated tests.
   - Any hash policy change must increment hash policy version and publish migration notes.

## Design Principles
1. **Single Source of Truth**: One `_run_training_pipeline()` method used by both entrypoints
2. **Gate Stack Consistency**: Same gates executed in same order for both paths
3. **Evidence Artifacts**: Identical metadata structure with gate markers
4. **Fail-Closed**: Any gate failure blocks training with clear reason codes

## Gate Pipeline Order

```
1. CoverageContractGate
   └─ Fails if: insufficient observed days for walk-forward
   
2. PostGapRetentionGate  
   └─ Fails if: data too fragmented after gap handling
   
3. ClassCoverageGate
   └─ Fails if: critical classes missing from splits
   
4. StatisticalValidityGate
   └─ Fails if: low support + high F1 (unreliable metrics)
   
5. Training Execution
   └─ With strict sequence-label alignment
   
6. Rejection Artifact Generation
   └─ Captures all gate failures in unified artifact
```

## Interface

### UnifiedTrainingPipeline
```python
class UnifiedTrainingPipeline:
    def train_unified(
        self,
        room_name: str,
        df: pd.DataFrame,
        elder_id: str,
        seq_length: int,
        observed_days: set,
        progress_callback=None,
        training_mode="full_retrain",
        defer_promotion: bool = False,
    ) -> dict:
        """
        Single hardened training path for both watcher and manual entrypoints.
        
        Returns dict with:
        - room: str
        - gate_pass: bool
        - gate_reasons: List[str]
        - gate_stack: List[dict]  # Evidence artifact
        - metrics: dict  # Training metrics if passed
        - rejection_artifact: dict  # If failed
        """
```

## Evidence Artifact Format

```json
{
  "gate_stack": [
    {
      "gate_name": "CoverageContractGate",
      "passed": true,
      "timestamp": "2026-02-16T10:30:00Z",
      "details": {"observed_days": 10, "required_days": 8}
    },
    {
      "gate_name": "PostGapRetentionGate", 
      "passed": true,
      "timestamp": "2026-02-16T10:30:01Z",
      "details": {"retained_ratio": 0.85, "segments": 3}
    }
  ],
  "final_status": "passed|failed",
  "failure_stage": null|"coverage"|"retention"|"class_coverage"|"statistical_validity"
}
```

## Test Requirements

### Feature Parity Tests
1. `test_watcher_path_gates_match_manual_path` - Same gates executed
2. `test_gate_stack_in_metadata_both_paths` - Evidence artifacts identical
3. `test_rejection_artifact_both_paths` - Unified rejection handling
4. `test_sequence_label_alignment_both_paths` - Strict alignment enforced
5. `test_duplicate_resolution_both_paths` - Deterministic duplicate handling

## Implementation Steps

1. Create `UnifiedTrainingPipeline` class in `ml/unified_training.py`
2. Integrate into `pipeline.py` - both `train_and_predict` and `train_from_files`
3. Update `run_daily_analysis.py` to use unified path
4. Add comprehensive feature parity tests
