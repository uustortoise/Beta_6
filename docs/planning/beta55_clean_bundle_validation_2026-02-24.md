# Beta 5.5 Clean Bundle Validation (2026-02-24)

## What was validated
A full orchestrated run using the new clean bundle script completed end-to-end on `pre_arrival_quick` profile:
- matrix attempt
- fallback variant/seed execution
- aggregation
- go/no-go evaluation
- ranking artifact generation

## Validation command executed
```bash
python3 /Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_lr_fragmentation_sweep_clean.py \
  --profiles-yaml /Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_matrix_profiles.yaml \
  --profile pre_arrival_quick \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --output-dir /tmp/beta55_clean_bundle_quick_20260224 \
  --go-no-go-config /Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_go_no_go.yaml \
  --max-workers 1 \
  --seed-timeout-seconds 240 \
  --seed-retries 1 \
  --matrix-timeout-seconds 900 \
  --cleanup-resource-trackers
```

## Validation artifacts
- Manifest:
  - `/tmp/beta55_clean_bundle_quick_20260224/pre_arrival_quick/clean_sweep_manifest.json`
- Ranking CSV:
  - `/tmp/beta55_clean_bundle_quick_20260224/pre_arrival_quick/ranking.csv`
- Ranking Markdown:
  - `/tmp/beta55_clean_bundle_quick_20260224/pre_arrival_quick/ranking.md`

## Full run command for clean worker (target profile)
```bash
/Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_lr_fragmentation_sweep_clean.sh
```

Equivalent explicit command:
```bash
python3 /Users/dicksonng/DT/Development/Beta_5.5/backend/scripts/run_lr_fragmentation_sweep_clean.py \
  --profiles-yaml /Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_matrix_profiles.yaml \
  --profile lr_fragmentation_sweep \
  --data-dir "/Users/dicksonng/DT/Development/New training files" \
  --elder-id HK0011_jessica \
  --output-dir /tmp/beta55_lr_frag_sweep_clean_$(date +%Y%m%d_%H%M%S) \
  --go-no-go-config /Users/dicksonng/DT/Development/Beta_5.5/backend/config/event_first_go_no_go.yaml \
  --max-workers 1 \
  --seed-timeout-seconds 300 \
  --seed-retries 1 \
  --matrix-timeout-seconds 1200 \
  --cleanup-resource-trackers
```
