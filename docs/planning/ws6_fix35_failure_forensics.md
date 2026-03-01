# WS6 Fix35 Failure Forensics

Source: `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_stage3_fix35_segment_room_specific`
Generated: 2026-02-19T05:38:43.463583Z

## Topline
- Hard-gate: 37/60 (decision=FAIL)
- Bedroom/LivingRoom hard-gate cells: 1/24

## Room Failure Counts
- Bedroom: 11
- LivingRoom: 12

## Top Hard-Gate Reasons
- 20: occupied_recall_lt_0.500
- 12: occupied_f1_lt_0.580
- 9: occupied_f1_lt_0.550
- 9: fragmentation_score_lt_0.450
- 3: recall_livingroom_normal_use_lt_0.400
- 2: recall_sleep_lt_0.400

## Worst Split-Room Cells
- 3: [4]->5 / Bedroom
- 3: [4]->5 / LivingRoom
- 3: [4, 5]->6 / Bedroom
- 3: [4, 5]->6 / LivingRoom
- 3: [4, 5, 6]->7 / Bedroom
- 3: [4, 5, 6]->7 / LivingRoom
- 3: [4, 5, 6, 7]->8 / LivingRoom
- 2: [4, 5, 6, 7]->8 / Bedroom