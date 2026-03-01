# Golden Sample Collection Template (n=20)

## 1) Purpose
- Build a high-quality, auditable golden sample set (`n=20`) for ADL model validation and retraining.
- Ensure samples are clinically sensible, timestamp-valid, and label-consistent.

## 2) Scope
- Model/version: `[MODEL_NAME]` / `[COMMIT_OR_TAG]`
- Data window: `[START_DATE]` to `[END_DATE]`
- Resident(s): `[RESIDENT_IDS]`
- Rooms included: `[ROOM_LIST]`
- Activities included: `[ACTIVITY_LIST_CANONICAL]`

## 3) Canonical Label Rules
- Use canonical labels only: `[CANONICAL_LABELS]`
- Room-label constraints:
- `bathroom`: `toilet`, `shower`, `bathroom_normal_use`, `inactive`, `low_confidence`, `out`
- `bedroom`: `sleep`, `nap`, `bedroom_normal_use`, `inactive`, `low_confidence`, `out`
- `kitchen`: `cooking`, `kitchen_normal_use`, `inactive`, `low_confidence`, `out`
- `livingroom`: `livingroom_normal_use`, `watch_tv`, `inactive`, `low_confidence`, `out`
- `entrance`: `out`, `inactive`, `low_confidence`
- Timestamp format: `YYYY-MM-DD HH:MM:SS` (local timezone: `[TZ]`)

## 4) Inclusion / Exclusion
- Include:
- Clear, reviewable behavior windows with sufficient context (`>= [X]` seconds).
- Valid timestamps and room metadata.
- Exclude:
- Invalid/unparseable timestamps.
- Duplicate samples (same resident+room+timestamp window).
- Windows dominated by missing/corrupt sensor payloads.

## 5) Sampling Plan (Stratified, n=20)
- Goal: balanced coverage across room, activity, and time-of-day.
- Time bins: `night (00:00-06:00)`, `morning (06:00-12:00)`, `afternoon (12:00-18:00)`, `evening (18:00-24:00)`.

| Bucket ID | Room | Activity | Time Bin | Target Count | Owner | Status |
|---|---|---|---|---:|---|---|
| B1 | bathroom | toilet | night | 3 | `[NAME]` | `[ ]` |
| B2 | bedroom | sleep/nap | night | 4 | `[NAME]` | `[ ]` |
| B3 | kitchen | cooking/normal_use | morning | 3 | `[NAME]` | `[ ]` |
| B4 | livingroom | normal_use/watch_tv | evening | 3 | `[NAME]` | `[ ]` |
| B5 | entrance | out | afternoon | 2 | `[NAME]` | `[ ]` |
| B6 | mixed | low_confidence edge cases | all | 5 | `[NAME]` | `[ ]` |

## 6) Sample Annotation Sheet
Use one row per candidate sample.

| Sample ID | Resident ID | Room | Start TS | End TS | Proposed Label | Final Label | Reviewer 1 | Reviewer 2 | Agreement | Notes | Include (Y/N) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `[ID]` | `[RES]` | `[ROOM]` | `[TS]` | `[TS]` | `[LABEL]` | `[LABEL]` | `[NAME]` | `[NAME]` | `[Y/N]` | `[TEXT]` | `[Y/N]` |

## 7) Quality Control Checklist
- [ ] `n=20` achieved.
- [ ] All timestamps parse successfully.
- [ ] No duplicate sample windows.
- [ ] Label-room constraints satisfied.
- [ ] Inter-rater agreement completed for all samples.
- [ ] Clinical reviewer spot-check done (`[COUNT]` samples).
- [ ] Audit fields complete (who/when/why for each correction).

## 8) Acceptance Gates (Must Pass)
- Gate A: Data integrity
- Timestamp parse failures = `0`
- Duplicate samples = `0`
- Gate B: Label quality
- Inter-rater agreement >= `[THRESHOLD]%`
- Invalid label-room combinations = `0`
- Gate C: Coverage
- All required buckets meet target counts.

## 9) Handoff Package
- Dataset artifact: `[PATH_OR_TABLE_NAME]`
- Version tag: `golden_v1_[YYYYMMDD]`
- Summary stats:
- total samples: `[20]`
- per-room counts: `[JSON_OR_TABLE]`
- per-activity counts: `[JSON_OR_TABLE]`
- parse-drop count: `[0]`
- Sign-off:
- Data lead: `[NAME/DATE]`
- ML lead: `[NAME/DATE]`
- Clinical lead: `[NAME/DATE]`

## 10) Risks / Open Items
- Risk 1: `[TEXT]` | Owner: `[NAME]` | ETA: `[DATE]`
- Risk 2: `[TEXT]` | Owner: `[NAME]` | ETA: `[DATE]`
