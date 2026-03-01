# PostgreSQL Migration Checklist

## Phase 1: Preparation (Current)
- [x] Docker Compose configured
- [x] Schema defined (Hypertable, JSONB)
- [x] Database Abstraction Layer implemented
- [ ] **Verification**: Docker container running
- [ ] **Verification**: Schema successfully applied to Postgres
- [ ] **Verification**: Python can connect via `database.py`

## Phase 2: Dual-Write Rollout
- [ ] Switch `USE_POSTGRESQL=true` in `.env` (or mock for testing)
- [ ] Monitor logs for "Dual-write failed" errors
- [ ] Verify data appearing in both `.db` and Postgres
- [ ] Check performance overhead (ensure < 50ms latency penalty)

## Phase 3: Data Migration
- [ ] Stop writes (Maintenance Window)
- [ ] Run migration script (to be created)
- [ ] Validate row counts match
- [ ] Validate checksums match

## Phase 4: Cutover
- [ ] Update config to read from Postgres
- [ ] Restart services
- [ ] Verify Dashboard loads correctly
