# Beta6 Safe Mixed Candidate Design

**Goal:** Build the safest promotion-grade Jessica mixed candidate by preserving the proven Bedroom `v36` anchor, adopting the benchmark-better LivingRoom `v40`, and verifying the resulting room pack end-to-end before any integration.

**Decision:** Use `HK0011_jessica_candidate_nodownsample_20260310T132301Z` as the assembly base because it already contains the desired `Bathroom_v35`, `Entrance_v26`, `Kitchen_v27`, and `LivingRoom_v40` artifacts. Create a new candidate namespace from that pack, then roll Bedroom back from `v37` to `v36` using the registry helper so latest aliases, thresholds, activity-confidence artifacts, and two-stage metadata remain consistent with runtime expectations.

**Why this path:** `LivingRoom_v40` is a clean model-quality win over `v30`, while `Bedroom_v37` requires an explicit support-gate exception. The safe path keeps the confidence/runtime and sampling fixes that are already validated, isolates the candidate in a new namespace, and makes the only behavior change the already-accepted Bedroom rollback to `v36`.

**Verification plan:**
- Replay the confirmed corrected Dec 17 workbook through the candidate namespace with the existing Jessica benchmark harness.
- Compare the safe candidate against the validated mixed baseline so the only intentional deltas are LivingRoom improvement and any secondary overall movement.
- Run a clean `UnifiedPipeline` load check against the candidate namespace to verify room discovery, Bathroom two-stage wiring, and runtime-ready latest aliases.

**Integration gate:** Only integrate if the replay benchmark is internally consistent and the fresh-load sanity check shows all five rooms loading cleanly with the expected current versions.
