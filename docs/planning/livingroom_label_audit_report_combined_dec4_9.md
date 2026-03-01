# LivingRoom Label Audit Report: Dec 4-9 Contradiction Analysis

**Objective**: Identify mislabeled periods by cross-referencing LivingRoom sensor activity with occupancy labels across ALL rooms.
**Target Data**: Jessica HK0011 training files (Dec 4, 5, 6, 7, 8, 9).

---

## Executive Summary

The audit confirms **serious systematic mislabeling** in the LivingRoom across all six days analyzed. We found large "Phantom Gaps" where the resident is labeled as being "nowhere" (all rooms unoccupied) while the LivingRoom has high motion, lights on, and elevated CO2. 

In total, this represents **~24 hours** of potentially missed occupancy training data across these six days. Correcting these labels will drastically reduce the LivingRoom class imbalance.

### Overall Impact by Day

| Day | Suspicious windows (LR) | Phantom Gaps (Total Nowhere) | Verdict |
|---|---|---|---|
| **Dec 4** | 524 | 466 (~1.3 hours) | Moderate under-labeling |
| **Dec 5** | 2,067 | 1,909 (~5.3 hours) | Severe under-labeling |
| **Dec 6** | 1,519 | 1,386 (~3.8 hours) | Severe under-labeling |
| **Dec 7** | 2,261 | 2,047 (~5.7 hours) | Severe under-labeling |
| **Dec 8** | 3,006 | 2,658 (~7.4 hours) | Severe under-labeling |
| **Dec 9** | 2,822 | 2,665 (~7.4 hours) | Severe under-labeling |

*Note: No significant "dual-occupancy" (mislabeled as being in two places) or "sensor leakage" (sensors active while sleeping in another room) was found. The person was almost certainly in the LivingRoom during these gaps.*

---

## Detailed Chronological Findings

For each gap listed below:
*   **Current Label**: `unoccupied` (in ALL rooms)
*   **Contradiction**: Sensors show clear human presence in the LivingRoom while the labels claim the person is nowhere in the house.
*   **Recommended Label**: `livingroom_normal_use`

### DEC 4: Moderate Under-labeling
1.  **07:21:04 → 07:43:02 (7 min)** 
    *   **Evidence**: Motion spikes up to 187, lights are ON (967), CO2 is elevated.
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
2.  **08:07:04 → 08:26:55 (18 min)** 
    *   **Evidence**: Lights are ON (1015), CO2 is high (3264).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
3.  **19:22:07 → 19:55:42 (22 min)**
    *   **Evidence**: Motion spikes, lights are ON (961), CO2 is high (3214).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
4.  **20:56:14 → 21:24:27 (18 min)**
    *   **Evidence**: Motion spikes up to 222, lights are ON (953), sound is elevated (4.4).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`

### DEC 5: Severe Under-labeling (Evening Bias)
1.  **08:23:29 → 09:01:08 (24 min)**
    *   **Evidence**: Motion spikes up to 71, lights are ON (1195), CO2 is elevated (3199).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
2.  **10:07:25 → 10:49:09 (33 min)**
    *   **Evidence**: Ambient light present (592), CO2 is high (3250).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
3.  **11:58:40 → 12:35:04 (26 min)**
    *   **Evidence**: Ambient light present (730), CO2 is high (3255).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
4.  **16:28:03 → 18:01:34 (68 min)**
    *   **Evidence**: Sustained motion (averaging 2.3, max 211), lights are ON (991). **Over 1 hour completely missed.**
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
5.  **18:45:27 → 19:41:49 (35 min)**
    *   **Evidence**: Heavy motion (averaging 6.8, max 297), lights are ON (949).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`

### DEC 6: Severe Under-labeling (Afternoon/Evening)
1.  **12:20:21 → 13:10:20 (36 min)**
    *   **Evidence**: Sustained motion (averaging 3.3, max 236), lights are ON VERY HIGH (1509), sound is elevated (4.5).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
2.  **13:15:24 → 13:46:12 (14 min)**
    *   **Evidence**: Motion spikes (max 88), lights are ON VERY HIGH (1435).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
3.  **16:11:52 → 18:35:08 (85 min)**
    *   **Evidence**: Continuous motion (max 169), lights are ON (1005). **Almost 1.5 hours completely missed.**
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
4.  **18:46:00 → 19:42:51 (43 min)**
    *   **Evidence**: Lights are ON (945), sound is HIGH (5.0).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`

### DEC 7: Massive "Phantom Working" Day
1.  **08:05:56 → 08:53:08 (44 min)**
    *   **Evidence**: Motion is present (averaging 1.6, max 146), lights are ON (1117), sound is elevated (4.5).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
2.  **11:57:35 → 13:11:46 (34 min)**
    *   **Evidence**: Ambient light present (651), CO2 is elevated (3173).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
3.  **13:29:55 → 14:16:45 (27 min)**
    *   **Evidence**: Ambient light present (550), CO2 is elevated (3213).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
4.  **17:56:38 → 19:06:56 (56 min)**
    *   **Evidence**: Sustained motion (averaging 2.1, max 166), lights are ON (948), sound is VERY HIGH (5.4).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
5.  **19:26:20 → 20:39:45 (55 min)**
    *   **Evidence**: Sustained motion (averaging 2.2, max 196), lights are ON (952), sound is VERY HIGH (5.3).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`

### DEC 8: Massive Afternoon/Evening Gaps
1.  **07:16:38 → 08:16:36 (30 min)**
    *   **Evidence**: Motion spikes, lights are ON (1013), CO2 is elevated.
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
2.  **08:25:58 → 09:08:24 (40 min)**
    *   **Evidence**: Motion spikes (max 174), lights are ON (1149).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
3.  **13:09:59 → 14:06:43 (44 min)**
    *   **Evidence**: Heavy motion spikes (max 273), lights are ON (849), CO2 is HIGH (3217).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
4.  **17:02:11 → 17:59:25 (43 min)**
    *   **Evidence**: Motion spikes (max 144), lights are ON (945), CO2 is HIGH (3384).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
5.  **19:16:47 → 21:09:23 (69 min)**
    *   **Evidence**: Sustained motion (averaging 1.3, max 140), lights are ON (951).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
6.  **21:29:30 → 22:39:21 (47 min)**
    *   **Evidence**: Sustained motion (averaging 1.7, max 118), lights are ON (727).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`

### DEC 9: Extreme Single-Block Gaps
1.  **07:44:51 → 08:53:00 (46 min)**
    *   **Evidence**: Motion spikes, lights are ON (1049).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
2.  **10:48:00 → 13:14:52 (138 min)**
    *   **Evidence**: Sustained motion throughout (max 252), lights are ON (601), CO2 is HIGH (3381). **Over 2 hours missed!**
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
3.  **13:22:52 → 15:16:11 (89 min)**
    *   **Evidence**: Motion spikes (max 55), lights are ON (619), CO2 is HIGH (3339). **1.5 hours missed!**
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
4.  **17:39:07 → 18:32:58 (41 min)**
    *   **Evidence**: Sustained motion (averaging 2.0, max 136), lights are ON (941), CO2 is HIGH (3238).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`
5.  **20:05:44 → 22:05:14 (72 min)**
    *   **Evidence**: Sustained motion (averaging 1.5, max 175), lights are ON (944), sound is high (4.5), CO2 is extremely high (12769, indicative of a sensor spike but confirms general anomaly).
    *   **Action**: Change from `unoccupied` → `livingroom_normal_use`

---

## Action Items & Team Recommendations

1. **Systematic Evening Miss**: Notice the consistent gaps between **~16:00 and 20:00** across almost all days (Dec 5, 6, 7, 8). The labeler seems to have systematically skipped the late afternoon/early evening routine where the resident was clearly in the LivingRoom (evidenced by active motion + lights).
2. **"Bulk Correction" Opportunity**: Re-label the `Unoccupied` timestamps listed above as `livingroom_normal_use` focusing heavily on the `16:00-20:00` blocks.
3. **Retrain Candidate D/E**: Once the annotations are updated, retrain the `Candidate D/E` pipeline. With potentially +24 hours of LivingRoom occupied data integrated, the severe class imbalance will be drastically mitigated.
