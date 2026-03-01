# Debug Guide: Corrections Not Showing in UI

## Current Situation

You've made a correction to Samuel's training file but can't see it in:
1. Activity Timeline
2. Audit Trail

## What We Know

✅ **Database Setup:** PostgreSQL only (`POSTGRES_ONLY=true`)
✅ **Tables Exist:** PostgreSQL has all required tables
❌ **SQLite is Empty:** The SQLite file exists but is 0 bytes (expected with POSTGRES_ONLY)

## Step-by-Step Debug Instructions

### Step 1: Check Streamlit Logs (Where you made the correction)

Look at the terminal where you're running `streamlit run export_dashboard.py`:

```bash
# Look for these messages after clicking "Apply Corrections":

# SUCCESS indicators:
[SAVE] Updated X rows in adl_history
[SAVE] Inserted audit entry in correction_history  
[SEGMENTS] Regenerated Y segments for samuel/bedroom/2026-02-07
✅ Success! Applied Z corrections...

# ERROR indicators:
❌ ERROR: Database error: ...
❌ Failed to regenerate segments
```

**What to look for:**
- Did you see "Success!" message?
- Were there any ERROR messages?
- How many rows were updated?

### Step 2: Direct PostgreSQL Check

Since you have PostgreSQL, run this query directly:

```bash
# Connect to PostgreSQL
psql -U your_username -d elderlycare -h localhost

# Check for samuel's corrections
SELECT 
    timestamp,
    room,
    activity_type,
    is_corrected,
    record_date
FROM adl_history
WHERE elder_id = 'samuel'
    AND is_corrected = 1
ORDER BY timestamp DESC
LIMIT 10;

# Check audit trail
SELECT 
    corrected_at,
    room,
    old_activity,
    new_activity,
    rows_affected
FROM correction_history
WHERE elder_id = 'samuel'
ORDER BY corrected_at DESC
LIMIT 5;

# Check segments
SELECT 
    room,
    activity_type,
    start_time,
    end_time,
    is_corrected
FROM activity_segments
WHERE elder_id = 'samuel'
    AND record_date = '2026-02-07'
ORDER BY start_time;
```

### Step 3: Check Web UI Database Connection

The Web UI might be connecting to the wrong database (SQLite instead of PostgreSQL).

**Check the Web UI's `.env` or config:**
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/web-ui

# Check if it has its own database config
cat .env.local 2>/dev/null || echo "No .env.local"
cat .env 2>/dev/null || echo "No .env"

# Check the database connection in the code
grep -r "DB_PATH\|DATABASE_URL\|postgres\|sqlite" app/lib/ --include="*.ts" | head -10
```

### Step 4: Common Issues & Fixes

#### Issue 1: Web UI Using SQLite Instead of PostgreSQL

**Symptom:** Corrections in PostgreSQL but Web UI shows empty

**Fix:** Check `web-ui/app/lib/db.ts` or data.ts:
```typescript
// Look for where the database connection is made
// It might be pointing to SQLite file instead of PostgreSQL
```

#### Issue 2: Cache Not Cleared

**Fix:** 
1. Hard refresh browser: **Ctrl+F5** (Windows) or **Cmd+Shift+R** (Mac)
2. Or click the refresh button in Web UI
3. Or clear browser cache completely

#### Issue 3: Wrong Date Selected

**Fix:** 
1. Check that you're viewing the correct date in Web UI
2. The correction might be on a different date than you're viewing

#### Issue 4: Segment Regeneration Failed

**Check:** Did you see "Regenerating segments..." in Streamlit logs?

If not, the segments weren't regenerated and the Web UI won't see the correction.

### Step 5: Immediate Test

Let's do a controlled test:

1. **Open Streamlit Labeling Studio**
2. **Select Samuel, pick a room, pick a date with data**
3. **Make ONE correction** (change one activity label)
4. **Click "Apply Corrections"**
5. **Watch the terminal output** - copy/paste what you see
6. **Wait for success message**
7. **Check Web UI** - refresh the page

### Step 6: Capture Debug Info

Run this in your terminal and share the output:

```bash
cd /Users/dicksonng/DT/Development/Beta_5.5

# Check what's in PostgreSQL (you'll need your PG credentials)
PGPASSWORD=your_password psql -U postgres -d elderlycare -c "
SELECT COUNT(*) as corrected_rows 
FROM adl_history 
WHERE elder_id = 'samuel' AND is_corrected = 1;
"

PGPASSWORD=your_password psql -U postgres -d elderlycare -c "
SELECT COUNT(*) as audit_entries 
FROM correction_history 
WHERE elder_id = 'samuel';
"

PGPASSWORD=your_password psql -U postgres -d elderlycare -c "
SELECT COUNT(*) as corrected_segments 
FROM activity_segments 
WHERE elder_id = 'samuel' AND is_corrected = 1;
"
```

## Most Likely Causes

Based on your setup, here are the most likely causes (in order):

1. **Web UI connecting to wrong database** (SQLite instead of PostgreSQL)
2. **Segment regeneration failed** (check Streamlit logs)
3. **Cache not cleared** (try hard refresh)
4. **Wrong date being viewed** (check date selector)

## What I Need From You

To help you further, please provide:

1. **Streamlit logs** after making a correction (last 20 lines)
2. **PostgreSQL query results** from Step 6 above
3. **Web UI database connection** - check if it's using SQLite or PostgreSQL
4. **Screenshot** of what you see in Web UI (Activity Timeline)

## Quick Fixes to Try

### Fix 1: Clear Web UI Cache
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/web-ui
rm -rf .next
npm run dev
```

### Fix 2: Restart Both Services
```bash
# Stop everything
pkill -f streamlit
pkill -f "next dev"

# Start fresh
cd backend && streamlit run export_dashboard.py &
cd web-ui && npm run dev &
```

### Fix 3: Check Web UI DB Connection
Make sure Web UI is using PostgreSQL, not SQLite.

## Next Steps

Please run the diagnostic steps above and share:
1. What you see in Streamlit logs after making a correction
2. The PostgreSQL query results
3. Whether Web UI hard refresh helps

This will help us pinpoint exactly where the breakdown is occurring.
