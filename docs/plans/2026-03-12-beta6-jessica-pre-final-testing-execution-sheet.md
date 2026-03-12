# Beta6 Jessica Pre-Final Testing Execution Sheet

Use this sheet while executing the checklist in `docs/plans/2026-03-12-beta6-jessica-pre-final-testing-checklist.md`.

## Run Header

- Branch:
- Commit:
- Worktree:
- Tester:
- Date:
- Candidate namespace:
- Primary replay workbook:
- Optional auxiliary workbooks reviewed:

## Required File Set

Primary required replay file for pre-final testing:

- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_17dec2025.xlsx`

Optional auxiliary corrected-pack files for provenance or room-specific follow-up only:

- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_4dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_5dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_6dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_7dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_8dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_9dec2025.xlsx`
- `/Users/dickson/DT/DT_development/Development/New training files/Jessica (sleep and out fixed)/HK0011_jessica_train_10dec2025.xlsx`

Rule:

- Do not expand the pre-final gate into a new full-pack retrain unless a room-level blocker requires it.
- The default pre-final path is candidate validation on the corrected `2025-12-17` workbook plus artifact readback.

## Preflight

- [ ] branch confirmed as `codex/jessica-pre-final-testing`
- [ ] commit recorded
- [ ] candidate namespace confirmed as `HK0011_jessica_candidate_supportfix_20260310T2312Z`
- [ ] focused regression suite passed
- [ ] syntax sanity passed
- [ ] candidate load sanity completed
- [ ] fresh Dec 17 replay completed

Artifacts:

- Regression output:
- Load sanity output:
- Dec 17 replay summary:
- Additional comparison artifacts:

## Bathroom Sign-Off

- Expected version: `v35`
- Versions file checked:
- Load path checked:
- Dec 17 final macro-F1:
- Low-confidence / rewrite notes:
- Failure-family notes:
- Verdict: `pass` / `conditional` / `block`
- Blocking issue:
- Artifact paths:

## Bedroom Sign-Off

- Expected version: `v38`
- Versions file checked:
- Dec 17 final macro-F1:
- Low-confidence / rewrite notes:
- `unoccupied -> bedroom_normal_use`:
- `sleep -> unoccupied`:
- `bedroom_normal_use -> unoccupied`:
- Control-plane safety artifacts checked:
- Verdict: `pass` / `conditional` / `block`
- Blocking issue:
- Artifact paths:

## Kitchen Sign-Off

- Expected version: `v27`
- Versions file checked:
- Dec 17 final macro-F1:
- Drift notes:
- Verdict: `pass` / `conditional` / `block`
- Blocking issue:
- Artifact paths:

## LivingRoom Sign-Off

- Expected version: `v40`
- Versions file checked:
- Dec 17 final macro-F1:
- Low-confidence / rewrite notes:
- `livingroom_active` recall notes:
- Runtime artifact load notes:
- Verdict: `pass` / `conditional` / `block`
- Blocking issue:
- Artifact paths:

## Final Decision

- Overall verdict: `GO` / `NO-GO`
- Conditional rooms:
- Blocking rooms:
- Exact blocker summary:
- Recommended next step:

## Notes To Preserve

- Bedroom is still fail-closed, not fully robust across valid alternate regimes.
- All-room lightweight lineage observability is present, but only Bedroom has the added instability block.
- Local PostgreSQL historical-corrections access remained unavailable during the Bedroom root-fix program; carry that caveat if it affects the final risk call.
