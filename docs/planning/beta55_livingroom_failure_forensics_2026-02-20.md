# Beta 5.5 LivingRoom Failure Forensics (2026-02-20)

- Scope: Eligible hard-gate LivingRoom failures under `min_train_days=3` strict WS-6 (`day4-8`)
- Goal: Isolate persistent blockers after parameter sweeps

## CurrentCode Anchor-Config (20/30)
- Gate summary: eligible `20/30`, full `38/60`
- Eligible split metrics (LivingRoom):
  - seed11 day7: pass=False occ_recall=0.280 occ_f1=0.141 frag=0.435 reasons=['occupied_f1_lt_0.580', 'occupied_recall_lt_0.500', 'fragmentation_score_lt_0.450']
  - seed11 day8: pass=False occ_recall=0.592 occ_f1=0.541 frag=1.000 reasons=['occupied_f1_lt_0.580']
  - seed22 day7: pass=False occ_recall=0.271 occ_f1=0.138 frag=0.500 reasons=['occupied_f1_lt_0.580', 'occupied_recall_lt_0.500']
  - seed22 day8: pass=False occ_recall=0.606 occ_f1=0.560 frag=0.895 reasons=['occupied_f1_lt_0.580']
  - seed33 day7: pass=False occ_recall=0.264 occ_f1=0.123 frag=0.435 reasons=['occupied_f1_lt_0.580', 'occupied_recall_lt_0.500', 'fragmentation_score_lt_0.450']
  - seed33 day8: pass=False occ_recall=0.615 occ_f1=0.570 frag=0.895 reasons=['occupied_f1_lt_0.580']
- Top fail reasons (eligible LivingRoom context):
  - `occupied_f1_lt_0.580`: 6
  - `occupied_recall_lt_0.500`: 3
  - `fragmentation_score_lt_0.450`: 2
- Worst eligible split: seed33 day7 (occ_f1=0.123, occ_recall=0.264)
- Top FN episodes:
  - 2025-12-07 07:43:58 to 2025-12-07 08:05:46 (22.0 min)
  - 2025-12-07 17:48:58 to 2025-12-07 17:56:29 (7.7 min)
  - 2025-12-07 08:53:18 to 2025-12-07 08:58:04 (4.9 min)

## Best Candidate D/E (21/30)
- Gate summary: eligible `21/30`, full `39/60`
- Eligible split metrics (LivingRoom):
  - seed11 day7: pass=False occ_recall=0.301 occ_f1=0.193 frag=0.556 reasons=['occupied_f1_lt_0.580', 'occupied_recall_lt_0.500']
  - seed11 day8: pass=False occ_recall=0.382 occ_f1=0.450 frag=0.708 reasons=['occupied_f1_lt_0.580', 'occupied_recall_lt_0.500', 'recall_livingroom_normal_use_lt_0.400']
  - seed22 day7: pass=False occ_recall=0.294 occ_f1=0.190 frag=0.526 reasons=['occupied_f1_lt_0.580', 'occupied_recall_lt_0.500']
  - seed22 day8: pass=False occ_recall=0.483 occ_f1=0.504 frag=0.548 reasons=['occupied_f1_lt_0.580', 'occupied_recall_lt_0.500']
  - seed33 day7: pass=False occ_recall=0.301 occ_f1=0.255 frag=0.714 reasons=['occupied_f1_lt_0.580', 'occupied_recall_lt_0.500']
  - seed33 day8: pass=False occ_recall=0.341 occ_f1=0.421 frag=0.586 reasons=['occupied_f1_lt_0.580', 'occupied_recall_lt_0.500', 'recall_livingroom_normal_use_lt_0.400']
- Top fail reasons (eligible LivingRoom context):
  - `occupied_f1_lt_0.580`: 6
  - `occupied_recall_lt_0.500`: 6
  - `recall_livingroom_normal_use_lt_0.400`: 2
- Worst eligible split: seed22 day7 (occ_f1=0.190, occ_recall=0.294)
- Top FN episodes:
  - 2025-12-07 07:43:58 to 2025-12-07 08:05:46 (22.0 min)
  - 2025-12-07 17:48:58 to 2025-12-07 17:56:29 (7.7 min)
  - 2025-12-07 08:53:18 to 2025-12-07 08:58:04 (4.9 min)

## Key Findings
1. LivingRoom eligible splits fail primarily on `occupied_recall_lt_0.500` and `occupied_f1_lt_0.580`.
2. Day 7 is consistently weakest for LivingRoom across seeds; day 8 is closer but still below F1 floor in most runs.
3. Segment-mode and sequence/hgb variants changed margins but did not produce any LivingRoom eligible hard-gate pass.
4. Next wins likely require data/label quality intervention on LivingRoom day-7/day-8 patterns, not additional threshold sweeps alone.

