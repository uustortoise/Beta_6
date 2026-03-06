# Beta 6.1 Home-Empty Duration Sweep (3 Seeds)

Generated: 2026-03-06T10:40:11.552689Z
Resident: HK001_jessica | Variant: anchor_top2_frag_v3 | Window: day 7-10

| Duration (s) | FER mean (95% CI) | Precision mean (95% CI) | Recall mean (95% CI) | Pred-empty rate mean | Safety pass all seeds | Utility pass seeds |
|---:|---:|---:|---:|---:|:---:|:---:|
| 600 | 0.0286 [0.0137,0.0435] | 0.2734 [0.1608,0.3861] | 0.1027 [0.1019,0.1035] | 0.0355 | YES | 2/3 |
| 900 | 0.0173 [0.0012,0.0335] | 0.2214 [0.0278,0.4150] | 0.0522 [0.0522,0.0522] | 0.0201 | YES | 0/3 |
| 1200 | 0.0114 [0.0072,0.0156] | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.0107 | YES | 0/3 |

## Recommendation

- Recommended duration: **600s**
- Operational recommendation: **CANARY_ELIGIBLE** (safety_all_seeds_pass_and_utility_majority)
- Criterion: pass safety cap across all seeds, then maximize utility pass count (precision/recall/predicted-empty-rate).
