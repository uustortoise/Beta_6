# Beta6 LivingRoom Timestamp-Level Label Forensic

## Scope

- Revisit the LivingRoom cross-date brittleness at timestamp level with a precision-first labeling lens.
- Separate true model-side reversals from candidate ground-truth boundary/island inconsistencies.
- Produce an ops-reviewable proposal pack that can be imported into the new Correction Studio proposal-review workflow.

## Inputs

- Corrected Jessica training files:
  - `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_{4,5,6,7,8,9,10,17}dec2025.xlsx`
- Cross-date replay artifacts:
  - `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v46/*/comparison/LivingRoom_merged.parquet`
  - `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v50/*/comparison/LivingRoom_merged.parquet`
  - `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v52/*/comparison/LivingRoom_merged.parquet`
- Prior forensic context:
  - `docs/reviews/2026-03-11-beta6-livingroom-brittleness-forensic.md`

## Method

1. Resample the corrected LivingRoom labels from the source workbooks to the replay timestamp grid.
2. Join source labels with `v46`, `v50`, and `v52` predictions on identical timestamps.
3. Scan for short truth runs sandwiched by the opposite label.
4. Keep only high-precision candidates where the disagreement is persistent and evidence-backed:
   - all available models agree against the truth for the span, or the block is a tight boundary oscillation with the same consensus pattern
   - the surrounding label context points in the same opposite direction
   - raw sensor continuity does not contradict the proposed relabel
5. Exclude the exemplar Dec 4 and Dec 17 blocks when the evidence shows a real model miss rather than a label problem.

## Main Finding

The large Dec 4 and Dec 17 reversals remain model-side. They should not be “fixed” by relabeling:

- 2025-12-04 `00:35-00:42`
  - `v46` and `v50` support the current occupied label; `v52` is the outlier undercall.
- 2025-12-04 `05:22-05:29`
  - `v46` and `v50` support the current unoccupied label; `v52` is the outlier overcall.
- 2025-12-04 `07:20-07:25`
  - same pattern as above: `v52` overcalls occupied.
- 2025-12-17 `14:36-14:45`
  - `v52` recovers a real occupied block that `v46` misses.
- 2025-12-17 `15:47-15:52`
  - same pattern: `v52` recovers a real occupied block.

The labeling opportunity is narrower: short boundary / island conflicts inside the corrected pack, not the headline cross-date reversal mechanism.

## Proposed Review Pack

High / medium confidence segment proposals:

1. `2025-12-05 16:13:00 -> 16:16:20`
   - current: `unoccupied`
   - proposed: `livingroom_normal_use`
   - confidence: `medium`
   - rationale: a 3m20s unoccupied island sits inside occupied labels; all three models keep it occupied and the sensor profile stays continuous with the surrounding occupied block.

2. `2025-12-07 21:18:50 -> 21:19:20`
   - current: `unoccupied`
   - proposed: `livingroom_normal_use`
   - confidence: `high`
   - rationale: the truth flips to unoccupied for 40 seconds and immediately flips back; all three models keep this short span occupied.

3. `2025-12-07 21:19:40 -> 21:22:00`
   - current: `livingroom_normal_use`
   - proposed: `unoccupied`
   - confidence: `high`
   - rationale: after the boundary spike, all three models move to unoccupied and the sensor profile aligns with the following long unoccupied run.

4. `2025-12-10 18:22:10 -> 18:25:50`
   - current: `livingroom_normal_use`
   - proposed: `unoccupied`
   - confidence: `medium`
   - rationale: a 3m40s occupied island sits inside long unoccupied runs; aside from the opening motion spike, the span looks like background and every model keeps it unoccupied.

Net proposal pack:

- segment proposals: `4`
- timestamp proposals: `63`

## Artifacts

- forensic script:
  - `tmp/livingroom_label_timestamp_forensic.py`
- summary:
  - `tmp/livingroom_label_timestamp_forensic_20260311T120957Z/summary.json`
- ops review table:
  - `tmp/livingroom_label_timestamp_forensic_20260311T120957Z/ops_review.csv`
- proposed corrected timestamp set:
  - `tmp/livingroom_label_timestamp_forensic_20260311T120957Z/proposed_timestamp_set.csv`
- importable proposal pack for Correction Studio:
  - `tmp/livingroom_label_timestamp_forensic_20260311T120957Z/proposal_pack.json`

## Validation

- `python3 -m py_compile tmp/livingroom_label_timestamp_forensic.py`
- `PYTHONPATH='.:backend' python3 tmp/livingroom_label_timestamp_forensic.py`
- `PYTHONPATH='.:backend' python3 -c "... _normalize_proposals_df(... proposal_pack.json ...) ..."`
  - normalized cleanly to `67` proposal rows (`4` segment, `63` timestamp), all with `review_status=proposed`

## Caveat

The proposal pack is schema-valid against `services.label_proposal_service._normalize_proposals_df(...)`, but I did not perform a live DB-backed import into Streamlit in this environment because local PostgreSQL access is blocked here. Treat the pack as import-ready, not DB-roundtrip-verified.

## Recommendation

- Keep the operational LivingRoom verdict unchanged:
  - the main Dec 4-17 instability is still a model-shape problem, not a label-pack fix.
- Use the proposal pack only for ops review of the four narrow candidate segments above.
- After ops approves or rejects them in Correction Studio, rerun a narrow LivingRoom retrain / replay only if those approved corrections are actually applied.
