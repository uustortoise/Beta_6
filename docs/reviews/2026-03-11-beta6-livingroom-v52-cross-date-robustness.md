# Beta6 LivingRoom v52 Cross-Date Robustness

## Scope

- Run the corrected Jessica replay pack across Dec 4, 5, 6, 7, 8, 9, 10, and 17, 2025.
- Compare promoted `LivingRoom_v52` against the old live `LivingRoom_v46` first.
- Only if the primary sweep is mixed, compare `v52` against `v50` and `v51` on the problematic dates.
- Keep promotion plumbing closed unless replay evidence forces a new model-side forensic.

## Artifacts

- Primary matched sweep:
  - `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/comparison_primary.json`
- Targeted panel summary:
  - `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/comparison_panel.json`
- Per-date replay outputs:
  - `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v46/*`
  - `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v50/*`
  - `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v51/*`
  - `tmp/jessica_livingroom_v52_crossdate_20260311T071050Z/v52/*`

## Findings

### 1. `v52` does not hold the Dec 17 win across the corrected-pack dates

Matched `v52` vs `v46` LivingRoom replay results:

| Date | `v46` acc | `v46` macro-F1 | `v52` acc | `v52` macro-F1 | `v52 - v46` acc | `v52 - v46` macro-F1 | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-12-04 | 0.9298 | 0.5320 | 0.9150 | 0.5082 | -0.0149 | -0.0238 | `v46` better |
| 2025-12-05 | 0.9508 | 0.5324 | 0.9208 | 0.4782 | -0.0299 | -0.0541 | `v46` better |
| 2025-12-06 | 0.8513 | 0.5279 | 0.7814 | 0.4744 | -0.0698 | -0.0535 | `v46` better |
| 2025-12-07 | 0.9336 | 0.7691 | 0.8934 | 0.7031 | -0.0402 | -0.0660 | `v46` better |
| 2025-12-08 | 0.8887 | 0.5598 | 0.8359 | 0.5065 | -0.0527 | -0.0533 | `v46` better |
| 2025-12-09 | 0.9061 | 0.5425 | 0.8745 | 0.5098 | -0.0316 | -0.0327 | `v46` better |
| 2025-12-10 | 0.8591 | 0.4204 | 0.8497 | 0.4443 | -0.0094 | +0.0239 | split: `v52` macro-F1 only |
| 2025-12-17 | 0.9200 | 0.4468 | 0.9243 | 0.5024 | +0.0043 | +0.0556 | `v52` better |

Interpretation:

- `v52` is not robustly better than the old `v46` / `v49` safe anchor.
- The corrected-pack sweep shows a date-specific reversal pattern, not a broad cross-date confirmation.
- The only clean `v52` win in this sweep is Dec 17.

### 2. `v50` is the best fresh neighbor, but it still does not beat the old anchor

Targeted problematic-date replay summary:

| Date | Best macro-F1 version | Best accuracy version | Notes |
| --- | --- | --- | --- |
| 2025-12-04 | `v46` | `v46` | `v50` and `v52` cluster near `0.509` macro-F1 |
| 2025-12-05 | `v46` | `v46` | `v50` nearly matches `v46` macro-F1 but still trails |
| 2025-12-06 | `v46` | `v46` | `v50` is second-best fresh seed |
| 2025-12-07 | `v46` | `v46` | `v50` > `v52`, but both trail `v46` |
| 2025-12-08 | `v46` | `v46` | `v50` > `v52`, both below anchor |
| 2025-12-09 | `v46` | `v46` | `v50` > `v52`, both below anchor |
| 2025-12-10 | `v52` | `v46` | fresh seeds still split on what "best" means |
| 2025-12-17 | `v52` | `v52` | `v50` / `v51` intentionally skipped per protocol |

Interpretation:

- `v50` is generally the strongest fresh alternative on the problematic dates.
- `v50` does not reclaim the old `v46` anchor on any of those dates.
- This is not a simple "`v52` picked the wrong winner but the fresh family is robust" story.

### 3. `v51` is catastrophically date-brittle

`v51` LivingRoom replay results on the targeted problematic dates:

- 2025-12-04: accuracy `0.0878`, macro-F1 `0.0539`
- 2025-12-05: accuracy `0.1466`, macro-F1 `0.0976`
- 2025-12-06: accuracy `0.2940`, macro-F1 `0.1770`
- 2025-12-07: accuracy `0.1145`, macro-F1 `0.1139`
- 2025-12-08: accuracy `0.2323`, macro-F1 `0.1412`
- 2025-12-09: accuracy `0.2025`, macro-F1 `0.1325`
- 2025-12-10: accuracy `0.1239`, macro-F1 `0.0736`

Interpretation:

- The fresh panel still contains at least one severely unstable passing seed.
- The saved no-regress-floor evidence did not guarantee cross-date replay safety.
- This materially strengthens the case for a model-side brittleness forensic instead of more blind seed search.

### 4. The recovered-harness sweep is internally useful but should not replace the earlier Dec 17 canonical absolute numbers

Fresh same-session Dec 17 reruns:

- `v52`: accuracy `0.9242920664638428`, macro-F1 `0.5024060992969163`
- `v46`: accuracy `0.9199625555815586`, macro-F1 `0.4467619015155215`

Earlier saved Dec 17 references:

- promoted live `v52`: accuracy `0.9235899836180669`, macro-F1 `0.5004234019159393`
- old live `v46`: accuracy `0.9217177626959981`, macro-F1 `0.4529600648213678`

Interpretation:

- The recovered harness reproduced the ordering on Dec 17 (`v52` > `v46`) but not the exact earlier absolute values.
- The drift affects both versions, so the new sweep is still valid as matched comparator evidence inside the same session.
- Keep the earlier saved Dec 17 artifacts as the authoritative canonical numbers when citing absolute replay values outside this sweep.

## Conclusion

Current verdict:

- `LivingRoom_v52` is not promotion-grade robust across the corrected Jessica dates.
- The old `v46` / `v49` safe anchor remains the best cross-date performer on Dec 4-9.
- Dec 10 is split (`v52` macro-F1, `v46` accuracy), and Dec 17 still favors `v52`.
- `v50` is the strongest fresh neighbor but still fails to beat the old anchor on the problematic dates.
- `v51` is severely brittle and confirms the fresh panel still contains dangerous geometries.

Recommended next step:

- keep promotion plumbing closed
- open a new model-side brittleness forensic on the LivingRoom fresh panel
- focus the forensic on why `v50..v52` can win Dec 17 while losing Dec 4-9, with `v46` / `v49` as the stable anchor baseline
- do not collect more training data yet; first explain the cross-date reversal mechanism
