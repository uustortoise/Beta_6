# Beta6 Bedroom Harmful-Day Segment Audit

## Scope

- Bedroom-first only.
- Harmful target from Task 3:
  - `2025-12-05`
- Good reference days:
  - `2025-12-04`
  - `2025-12-10`
- Audit artifacts:
  - coarse 120-minute pass:
    - `tmp/bedroom_day_segment_audit_20260311T0018Z/summary.json`
  - tightened 60-minute pass used for classification:
    - `tmp/bedroom_day_segment_audit_20260312T60m/summary.json`

## Method

For each Bedroom time block, compare `2025-12-05` against the good references on:

- class mix
- transition topology
- run-length structure
- sparsity / missingness

## Findings

### 1. The strongest harmful slices are coherent occupied-regime blocks, not sparse or broken data

Top 60-minute standout blocks from the tightened audit:

- `12:00-13:00`
  - target counts: `bedroom_normal_use=358`, `unoccupied=2`
  - reference delta: `bedroom_normal_use +0.994444`, `unoccupied -0.994444`
  - transitions: `unoccupied -> bedroom_normal_use = 1`
  - run lengths:
    - `bedroom_normal_use max=3580s`
    - `unoccupied max=20s`
- `20:00-21:00`
  - target counts: `bedroom_normal_use=360`
  - reference delta: `bedroom_normal_use +0.819445`, `unoccupied -0.819445`
  - transitions: none
  - run lengths:
    - `bedroom_normal_use max=3600s`
- `13:00-14:00`
  - target counts: `bedroom_normal_use=265`, `unoccupied=95`
  - reference delta: `bedroom_normal_use +0.736111`, `unoccupied -0.736111`
  - transitions:
    - `bedroom_normal_use -> unoccupied = 4`
    - `unoccupied -> bedroom_normal_use = 4`

These blocks have:

- full row density
- `largest_gap_seconds=10`
- zero missingness flags
- coherent contiguous runs rather than fragmented noise

That is not the profile of obvious corruption, truncation, or missing-sensor drift.

### 2. The harmful day introduces long Bedroom occupancy where the good references are mostly unoccupied

The most important difference is not sensor sparsity. It is a regime shift in label occupancy:

- `10:00-11:00`
  - target: `bedroom_normal_use=349`, `unoccupied=11`
  - reference delta: `bedroom_normal_use +0.677777`
- `11:00-12:00`
  - target: `bedroom_normal_use=320`, `unoccupied=40`
  - reference delta: `bedroom_normal_use +0.663889`
- `12:00-13:00`
  - target: almost entirely `bedroom_normal_use`
- `20:00-21:00`
  - target: entirely `bedroom_normal_use`

This matches the model failure direction from Task 3:

- adding `2025-12-05` increases `bedroom_normal_use -> unoccupied` and keeps `unoccupied -> bedroom_normal_use` high in replay
- the source day itself contains large, clean Bedroom-occupancy runs at times the good references mostly treat as `unoccupied`

### 3. The dawn anomaly is still coherent, not obviously broken

The strongest early-morning divergence is `05:00-06:00`:

- target counts: `unoccupied=360`
- reference delta: `sleep -0.633333`, `unoccupied +0.633333`
- transitions: none

This is unusual relative to the references, but again:

- no sparsity flags
- no missingness flags
- one stable label run, not churn

So even the off-pattern dawn block looks like a clean alternate behavior slice rather than a malformed data slice.

## Classification

Segment statuses:

- `10:00-14:00`: likely valid alternate regime
- `20:00-21:00`: likely valid alternate regime
- `05:00-06:00`: likely valid alternate regime

Reasoning:

- the standout blocks are dense and internally coherent
- the deviations are dominated by stable label-run changes, not missingness or impossible transition noise
- the harmful mechanism is consistent with a real behavior-pattern shift: longer Bedroom occupancy on `2025-12-05` than the good anchor days expose

## Decision Gate

Supported branch: **Branch B: valid alternate regime**

Explicit decision statement:

> The `2025-12-05` harmful slices do not look like bad or inconsistent data. They look like a valid alternate Bedroom occupancy regime that the current anchor-centered training pack does not handle robustly.

## Residual Uncertainty

- This does not prove every label on `2025-12-05` is clinically correct.
- It does show the current bounded forensic evidence is not pointing to a data-integrity break such as sparsity, missingness, or timestamp damage.
- The next bounded unknown for Branch B work is how to make Bedroom robust to this alternate regime without promoting a candidate solely because one replay looks better.

## Next Step

- Continue with the Branch B path from the plan.
