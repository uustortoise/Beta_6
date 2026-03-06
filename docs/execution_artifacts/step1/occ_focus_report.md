# Beta 6.1 Step 1 Occupancy Focus Report

Date: 2026-03-06
Resident: HK001_jessica | Seed: 22 | Window: day 7-10

| Variant | LR MAE | Bedroom Sleep MAE | Home-empty FER | FER delta vs A0 | Home-empty Precision | Precision drop vs A0 | Hard Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0_anchor | 88.393 | 44.717 | 0.3145 | +0.0000 | 0.8910 | +0.0000 | 4/5 |
| A1_mask_floor0 | 88.393 | 44.717 | 0.3145 | +0.0000 | 0.8910 | +0.0000 | 4/5 |
| A2_mask_floor005 | 88.393 | 44.717 | 0.3145 | +0.0000 | 0.8910 | +0.0000 | 4/5 |
| lr_occ_focus_v1 | 108.283 | 87.990 | 0.3488 | +0.0344 | 0.8834 | +0.0076 | 4/5 |
| lr_occ_focus_v2 | 108.283 | 90.954 | 0.3455 | +0.0310 | 0.8841 | +0.0069 | 4/5 |
| lr_occ_focus_v3 | 108.642 | 60.016 | 0.3902 | +0.0757 | 0.8718 | +0.0192 | 4/5 |

## Outcome

- None of the occupancy-focused variants reduced false-empty rate; all remained far above the <=0.05 safety cap.
- `lr_occ_focus_v1/v2/v3` worsened false-empty rate and degraded key MAEs relative to anchor.
- Current recommendation remains **NO-GO** for Step-1 promotion on occupancy safety grounds.