# Beta 5.5 User Manual Index

This index is the authoritative entrypoint for operations, ML, and labeling documentation in Beta 5.5.

## Start Here
1. Operations and startup:
   - `operation_manual.md`
2. End-to-end ML and ADL flow (technical source-of-truth):
   - `ml_adl_e2e_technical_flow.md`
3. Labeling policy and correction rules:
   - `labeling_guide.md`
4. Golden sample collection SOP:
   - `golden_sample_harvesting.md`
5. Non-ML handbook for product/ops stakeholders:
   - `ml_module_handbook_non_ml.md`

## Supporting References
- Data flow summary:
  - `data_flow_logic.md`
- Data architecture:
  - `data_architecture.md`
- ICOPE mapping:
  - `icope_framework.md`
- Feature status audit:
  - `feature_audit.md`

## Source-of-Truth Policy
1. For implementation and debugging details, use `ml_adl_e2e_technical_flow.md` first.
2. For day-to-day operations, use `operation_manual.md` and `golden_sample_harvesting.md`.
3. For labeling decisions, use `labeling_guide.md`.
4. If any older document conflicts with these files, treat the newer files above as authoritative.

## Current Runtime Endpoints
- Web UI: `http://localhost:3002`
- Streamlit Studio: `http://localhost:8503`
