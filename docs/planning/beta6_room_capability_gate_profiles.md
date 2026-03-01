# Beta 6 Room Capability Gate Profiles (v1)

- Date: 2026-02-25
- Status: Active
- Policy file: `backend/config/beta6_room_capability_gate_profiles.yaml`

## 1. Why Per-Room Profiles

Beta 6 uses room-specific gates because sensor separability differs by room. One global threshold would either over-block livingroom or under-protect bedroom quality.

## 2. Deterministic Selection Contract

1. Infer room type from room name (`bedroom`, `livingroom`, `bathroom`, otherwise `generic`).
2. Select profile by room type.
3. Emit `capability_profile_id` in room decision details.

Gate engine output fields:
1. `details.room_type`
2. `details.capability_profile_id`

## 3. Initial Profiles

1. `cap_profile_bedroom_v1`
2. `cap_profile_livingroom_v1`
3. `cap_profile_bathroom_v1`
4. `cap_profile_generic_v1`

Threshold values and rationale are versioned in `backend/config/beta6_room_capability_gate_profiles.yaml`.
