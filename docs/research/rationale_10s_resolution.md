# ML Technical Rationale: 10-Second Temporal Resolution

## Executive Summary
This paper documents the technical rationale behind choosing a **10-second fixed interval** for the temporal discretization of sensor data in the Beta 5.5 Hybrid CNN-Transformer architecture. This decision optimizes for **sensor hardware constraints**, **human activity patterns**, and **computational scalability** for the 1000-resident POC.

---

## 1. Hardware & Physical Constraints
The primary input sources for the ADL (Activities of Daily Living) pipeline are passive sensors (PIR Motion, Door contacts, Environmental sensors).

- **Sensor Jitter & Debounce**: Passive Infrared (PIR) sensors typically have a hardware debounce period of 2s to 8s. Sampling at sub-2s intervals captures "noise" or "chatter" rather than meaningful motion.
- **Environmental Latency**: Temperature and humidity changes measured by low-power zigbee/ble sensors do not realistically fluctuate at a sub-10s resolution due to the physical properties of air mass and sensor thermal inertia.
- **Denoising Benefits**: A 10s window allows the **Hampel Filter** to aggregate spikes across multiple raw raw samples, producing a cleaner "steady-state" signal for the Transformer embedding.

---

## 2. Activity Contextualization
The Transformer architecture relies on learning patterns over time. The choice of 10s aligns with the characteristic durations of the activities we detect:

| Activity Class | Typ. Duration | Resolution Sensitivity | Rationale |
| :--- | :--- | :--- | :--- |
| **Sleep / Nap** | 30m - 8h | Very Low | Stillness is constant over long periods. |
| **Shower** | 10m - 25m | Low | Humidity spikes last minutes, not seconds. |
| **Cooking** | 15m - 60m | Low | Sustained activity patterns. |
| **Toileting** | 3m - 15m | Moderate | Entry/Exit timing is captured effectively at 10s. |
| **Transitions** | 5s - 20s | High | 10s resolution maps these to 1-2 frames. |

**Observation**: While room-to-room transitions happen quickly (e.g., 5 seconds), the system's focus is on **sustained state** (where the person IS) rather than the precise millisecond they crossed the threshold.

---

## 3. Computational Scalability (1000 POC Scale)
For a system designed to scale to 1,000 residents, data density directly impacts memory and compute costs.

- **Data Volume**:
  - **10s Interval**: 8,640 rows/day. Total memory footprint is manageable for on-device processing.
  - **1s Interval**: 86,400 rows/day. Increases processing overhead by **10x** with diminishing accuracy returns.
- **Transformer Complexity**: Global Self-Attention complexity is $O(L^2)$ where $L$ is sequence length. Halving the interval (to 5s) quadruples the computational cost of the attention maps.

---

## 4. Refinement Options (Future Work)
While 10s remains the standard, the following software-level improvements are identified for future refinement:

1.  **Temporal Rounding**: Switch from `dt.floor` to `dt.round('10s')` to reduce the maximum temporal error from 9.9s to ±5s.
2.  **Weighted Aggregation**: Instead of a simple `mean` for sensor aggregation, use a **recency-weighted average** to give priority to events happening closer to the bucket edge.
3.  **Label Priority**: Implement "Most Important Label Wins" (e.g., prioritize `fall` or `motion` over `unoccupied` if both exist in the same 10s bucket).

---

## Conclusion
The 10-second resolution provides the **optimal balance** for detecting long-duration ADLs while maintaining the high performance required for large-scale elder care deployments. The Transformer model effectively utilizes this granularity to learn robust global context fingerprints.
