# Correction Pipeline Fix - Quick Start Guide

## 🚨 Immediate Actions (Do This First)

### Step 1: Run Diagnostics (5 minutes)

```bash
cd /Users/dicksonng/DT/Development/Beta_5.5
python scripts/diagnostic_correction_pipeline.py
```

**Look for:**
- ✅ adl_history has corrections (count > 0)
- ✅ activity_segments has corrections (count > 0)
- ❌ orphaned_count = 0
- ❌ No "DRIFT DETECTED" messages

### Step 2: Clear All Caches (5 minutes)

**Streamlit:**
- Restart the Streamlit server (Ctrl+C, then restart)

**Web UI:**
```bash
cd web-ui
rm -rf .next
npm run dev
```

**Browser:**
- Hard refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)

### Step 3: Test a Correction (5 minutes)

1. Open Labeling Studio (Streamlit UI)
2. Make ONE correction
3. Click "Apply Corrections"
4. Wait for "Success!" message
5. Check Web UI Activity Timeline

**If correction appears:** ✅ Problem solved (was cache issue)  
**If correction doesn't appear:** Continue to Phase 2

---

## 🔧 Phase 2: Quick Fixes (15 minutes)

### Fix A: Increase merge_asof Tolerance

Edit `backend/ml/pipeline.py` line 143:
```python
tolerance=pd.Timedelta(seconds=30)  # Change from 10 to 30
```

Edit `backend/export_dashboard.py` line 549:
```python
tolerance=pd.Timedelta('30s')  # Change from '10s'
```

Restart services and test.

### Fix B: Unify Merge Thresholds

Edit `web-ui/app/lib/data.ts` line 533:
```typescript
const MERGE_THRESHOLD_MS = 5 * 60 * 1000;  // Change from 60 * 1000
```

Restart Web UI and test.

---

## 📊 Interpreting Diagnostic Output

### Scenario 1: adl_history = 0 corrections
```
adl_history corrections:        0
```
**Problem:** Labeling Studio not saving corrections  
**Fix:** Check `save_corrections_to_db()` function

### Scenario 2: activity_segments = 0 corrections
```
adl_history corrections:        50
activity_segments corrected:    0
```
**Problem:** Segment regeneration broken  
**Fix:** Apply Fix 3.1 (consolidate segment regeneration)

### Scenario 3: Orphaned corrections found
```
orphaned_count:                 10
pipeline_status:               CRITICAL: Orphaned corrections detected
```
**Problem:** Segment generation incomplete  
**Fix:** Check `regenerate_segments()` function

### Scenario 4: Timestamp drift
```
Maximum timestamp drift: 15.234 seconds
❌ WARNING: Drift exceeds 10 seconds
```
**Problem:** Timezone/rounding issues  
**Fix:** Apply Fix 3.3 (standardize timezone handling)

---

## 🔍 Common Issues & Solutions

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| Correction shows in audit trail but not timeline | Cache issue | Clear all caches |
| Correction in adl_history but not segments | Segment regeneration broken | Apply Fix 3.1 |
| Correction shows after refresh but not immediately | Web UI cache | Apply Fix 3.2 |
| Random corrections missing | Timestamp drift | Apply Fix 2.1 (increase tolerance) |
| Corrections appear on wrong date | Timezone issue | Apply Fix 3.3 |

---

## 📞 Escalation Path

1. **Tried all quick fixes and still not working?**
   - Run full diagnostic: `python scripts/diagnostic_correction_pipeline.py [ELDER_ID] [DATE]`
   - Save output to file: `python scripts/diagnostic_correction_pipeline.py > diagnostic_output.txt`
   - Send to Senior Backend Developer

2. **Database errors?**
   - Check PostgreSQL is running: `pg_isready`
   - Check connection settings in `.env`
   - Contact Database Administrator

3. **Frontend not loading?**
   - Check Next.js console for errors
   - Verify API routes are accessible
   - Contact Frontend Developer

---

## ✅ Verification Checklist

After applying fixes, verify:

- [ ] Run diagnostic script - all checks pass
- [ ] Make a correction - success message appears
- [ ] Check adl_history - correction is saved (is_corrected=1)
- [ ] Check activity_segments - segment has is_corrected=1
- [ ] Check Web UI - correction appears in Activity Timeline
- [ ] No console errors in browser
- [ ] No errors in Streamlit terminal

---

## 📚 Full Documentation

For detailed implementation steps, see:
- **Full Fix Plan:** `CORRECTION_PIPELINE_FIX_PLAN.md`
- **SQL Diagnostics:** `scripts/diagnostic_correction_pipeline.sql`
- **Python Diagnostics:** `scripts/diagnostic_correction_pipeline.py`

---

**Need Help?** Ask in #backend-support or #frontend-support channels.
