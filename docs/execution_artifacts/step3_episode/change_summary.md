# Step3 Episode Metrics Change Summary

1. Normalized timeline payloads in Beta6 evaluation reports to include:
- duration_mae_minutes
- fragmentation_rate
- boundary_precision / boundary_recall / boundary_f1
- episode_count_ratio
2. Added non-blocking watch semantics for episode_count_ratio in GateEngine details.
3. Added tests for normalized timeline metric inclusion and watch-only gate behavior.

Decision: PASS
