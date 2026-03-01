# Beta 5.5 ML Module: Critique & Greenfield Redesign

## The Critique: Where We Are vs. Where We Should Be

The current Beta 5.5 ML module is a **robust monolith**. It has evolved through necessity, adding layers of "hardening" (like the ones we just implemented) to patch over fundamental architectural rigidities. It effectively solves the problem today, but at a high cost of complexity and maintenance.

### 1. Coupled Training & Serving
*   **Current State:** The `Pipeline` class tries to do everything: data loading, preprocessing, training, evaluation, AND registry management.
*   **Problem:** This coupling makes it hard to test components in isolation. You can't just "test the preprocessor" without mocking the entire `Pipeline` or `TrainingPipeline`.
*   **Consequence:** Hardening requires "surgery" deep inside monolithic methods (like `train_and_predict`), creating risk of side effects.

### 2. State Mutation Heavy
*   **Current State:** Dataframes are passed around and mutated in place (e.g., `preprocess_data` modifies `df` directly). The Registry relies on mutable JSON files and "syncing" logic.
*   **Problem:** Mutation is the enemy of determinism. We spend 50% of our time verifying state consistency (Registry Repair, Data Viability Gates) because the state *can* become inconsistent.
*   **Consequence:** We need elaborate "guardrails" to keep the system on the tracks.

### 3. Procedural vs. Declarative
*   **Current State:** The pipeline is a sequence of imperative steps: "Load data, then if x do y, then train...".
*   **Problem:** It's hard to visualize the DAG (Directed Acyclic Graph) of operations. Changing the order or adding a step (like "fail-closed resampling") requires careful code tracing.
*   **Consequence:** Assessing the impact of a change is mentally taxing.

---

## The Redesign: An Event-Sourced, Functional Architecture

If I were rebuilding this from scratch (Greenfield), I would adopt a **Functional Data Engineering** approach. The core philosophy would be: **Immutability, Isolation, and Declarative Pipelines.**

### 1. Decoupled Architecture (The "Micro-Pipes" Pattern)

Instead of one giant `Pipeline` class, I would break the system into three distinct, decoupled subsystems that communicate via **immutable artifacts**:

#### A. The Feature Store (Data Engineering)
*   **Responsibility:** Ingest raw sensor data, apply transformations, and output **versioned, immutable Feature Sets** (e.g., Parquet/Arrow files).
*   **Key Change:** No training logic here. Just "Data In -> Feature Snapshot Out".
*   **Benefit:** We can validate data quality *before* we even think about models. The "Data Viability Gate" becomes a simple check on the output file.

#### B. The Training Fabric (Model Engineering)
*   **Responsibility:** Take a specific Feature Snapshot + a specific Policy Config -> Produce a **Model Artifact**.
*   **Key Change:** Pure function. `train(data_v1, policy_v1) -> model_v1`. No side effects. No registry updates.
*   **Benefit:** Determinism is guaranteed by design. If inputs are the same, output is the same. Testing is trivial.

#### C. The Model Registry (Lifecycle Management)
*   **Responsibility:** Promoting, serving, and retiring models.
*   **Key Change:** The Registry is an *observer*. It sees a new Model Artifact, evaluates it against a "Gate Policy", and decides whether to update the `current_champion` pointer.
*   **Benefit:** Promotion logic is separated from training logic. We can re-evaluate promotion rules without retraining.

### 2. Implementation Sketch

#### The Manifest (Declarative Config)
Everything starts with a YAML manifest. No more passing args around.

```yaml
# experiment_manifest.yaml
data:
  source: "s3://clean/beta5/v4"
  window_size: "60m"
  resample_rule: "1T"

training:
  policy: "production_strict"
  seed: 42
  architecture: "transformer_v2"

gating:
  min_f1: 0.85
  min_days: 7
```

#### The Functional Pipeline (Pseudo-Code)

```python
def run_pipeline(manifest):
    # Step 1: Materialize Features (Immutable)
    feature_set_path = feature_eng.materialize(manifest.data)
    # Output: /data/features/hash_123.parquet
    
    # Step 2: Train Candidate (Pure Function)
    candidate_model_path = trainer.train(feature_set_path, manifest.training)
    # Output: /models/candidates/hash_456.keras
    
    # Step 3: Evaluate & Gating (Stateless)
    metrics = evaluator.evaluate(candidate_model_path, feature_set_path)
    if gate.check(metrics, manifest.gating):
        # Step 4: Registry Promotion (Atomic)
        registry.promote(candidate_model_path, metrics)
```

### 3. What We Gain

1.  **Zero "Registry Repair"**: The registry creates new immutable pointers. It never "mutates" an existing state in a way that can drift. We append, we don't overwrite.
2.  **Instant Reproducibility**: To reproduce a run, you just pass the same Manifest. The system creates the exact same Feature Set (or reuses the cached one) and trains the same model.
3.  **Simpler Testing**:
    *   Test Feature Eng: Input CSV -> Output Parquet.
    *   Test Trainer: Input Parquet -> Output Keras Model.
    *   Test Registry: Input Model + Metrics -> Output Promotion Event.

## Conclusion

**Would I do it differently? Yes.**
I would move away from the **"Manager Class"** pattern (where one class manages state and logic) towards a **"Pipeline of Artifacts"** pattern.

**Can we get there?**
We don't need a total rewrite. We are already moving this way! 
*   Your "Data Viability Gate" is Step 1 (validating features).
*   Your "Registry Hardening" is moving towards atomic improvements.
*   The "Execution Plan" we just wrote is effectively decoupling these stages logically, even if they physically live in the same class for now.

**Recommendation:**
Let's finish the current Hardening Plan (it's critical for stability). Then, for **Beta 6**, we should consider refactoring the `Pipeline` into these three distinct `FeatureStore`, `Trainer`, and `Registry` components.
