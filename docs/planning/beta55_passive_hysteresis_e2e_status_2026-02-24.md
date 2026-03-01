# Beta 5.5 Passive Hysteresis E2E Status (2026-02-24)

## Runtime Coverage
- Anchor: 3/3 seeds complete.
- Passive hysteresis: 2/3 seeds complete (seed 33 missing due host runner instability).
- Tuned hysteresis: 1/3 seeds available (seed 11 from prior ablation artifact).

## Variant Summary
| variant | seeds | eligible | go/no-go | Bedroom occ_f1 | Bedroom occ_recall | LivingRoom occ_f1 | LivingRoom occ_recall | blockers |
|---|---:|---:|---|---:|---:|---:|---:|---|
| anchor_top2_frag_v3 | 3 | 48/60 | fail | 0.7196681325600661 | 0.6757569180612517 | 0.2884210029433841 | 0.47167780225150807 | livingroom_eligible_pass_count_min |
| anchor_top2_frag_v3_passive_hysteresis | 2 | 24/40 | fail | 0.6798517615036617 | 0.9979147422750354 | 0.26180773192116025 | 0.9812169440885444 | livingroom_eligible_pass_count_min;bedroom_max_regression_splits;day8_livingroom_fragmentation_min |
| anchor_top2_frag_v3_passive_hysteresis_tuned | 1 | 12/20 | fail | 0.6806510082753557 | 1.0 | 0.24765902104926296 | 1.0 | overall_eligible_pass_count_min;livingroom_eligible_pass_count_min;bedroom_max_regression_splits;day8_livingroom_fragmentation_min |

## Assessment
- Current passive hysteresis does not improve gates versus anchor on completed seeds.
- Bedroom regressions appear under passive hysteresis (bedroom_max_regression_splits blocker appears).
- LivingRoom blocker remains (eligible pass count and fragmentation still unresolved).
- Recommendation: keep passive hysteresis default OFF in promotion path; use only as controlled experiment after runner stabilization and full 3-seed completion.
