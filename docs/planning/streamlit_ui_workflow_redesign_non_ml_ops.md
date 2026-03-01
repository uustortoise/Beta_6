# Streamlit UI Redesign Plan (Non-ML Ops Friendly)

## Purpose
Redesign Streamlit to match care operations workflows, not ML workflows.

Primary users:
1. Care ops coordinator
2. Field ops reviewer
3. Team lead (service quality)

Out of scope:
1. Deep model debugging in default views
2. Exposing internal ML terms as primary UI language

## Design Principles
1. Workflow-first: each page maps to a real daily task.
2. Plain language: avoid terms like macro-F1, calibration, gate stack in default UI.
3. Decision-first: show “what needs action now” before charts.
4. Progressive disclosure: advanced ML details hidden in expandable “Technical Details”.
5. Safety-first: highlight confidence and uncertainty where decisions affect care.

## Operator Workflows to Support

### Workflow A: Daily Care Check (5-10 min)
Questions:
1. Is anything unusual today?
2. Did the elder sleep, shower, use kitchen/living room normally?
3. Is the home currently likely empty?

UI outcome:
1. Single “Today Overview” page with red/amber/green cards.

### Workflow B: Label Review and Correction (15-30 min)
Questions:
1. Which periods are most important to correct?
2. Can I quickly fix labels and retrain safely?

UI outcome:
1. Prioritized correction queue with “impact estimate” and one-click apply/retrain flow.

### Workflow C: Weekly Service Review (30-60 min)
Questions:
1. Are care KPIs improving or drifting?
2. Is model quality acceptable for operations?

UI outcome:
1. Weekly KPI report view with trend and exceptions list.

### Workflow D: Configuration and Governance
Questions:
1. Are rules/configs valid?
2. Are model updates safe and audited?

UI outcome:
1. Dedicated admin area with policy changes, audit logs, and release status.

## Information Architecture (New Navigation)

Replace current tab emphasis with:
1. **Today**
2. **Review Queue**
3. **Resident Timeline**
4. **Weekly Report**
5. **Admin & Audit**

### Mapping from current tabs
1. `📤 Data Export` -> `Admin & Audit` (subsection)
2. `🏷️ Labeling Studio` -> `Review Queue` + `Resident Timeline`
3. `📊 Model Insights` -> `Weekly Report` (technical section moved to expanders)
4. `🏠 Household Overview` -> `Today`
5. `⚙️ AI Configuration` -> `Admin & Audit`
6. `📝 Audit Trail` -> `Admin & Audit`

## Screen-Level Specification

### 1) Today
Target user: care ops coordinator  
Primary components:
1. Resident selector
2. “Needs Attention” list (top 3 items)
   - Strictly actionable care/clinical anomalies only (not routine ML low-confidence events)
   - Route routine low-confidence / uncertain predictions to `Review Queue` unless they block care decisions
3. KPI cards:
   - Sleep status
   - Bathroom routine status
   - Kitchen activity status
   - Living room activity status
   - Home occupancy status (`occupied / empty / unknown`)
4. Confidence badge per card (`High / Medium / Low`)
5. Suggested action buttons:
   - “Open timeline”
   - “Add review task”
6. Global pilot mode banner (when enabled):
   - `Beta 5.5 currently active (Beta 6 in Shadow)`

Data contract (minimal):
1. `sleep_duration_mae_minutes`
2. `shower_day_precision/recall`
3. `kitchen_use_mae_minutes`
4. `livingroom_active_mae_minutes`
5. `home_status`, `home_status_confidence`
6. `uncertainty_rate`

### 2) Review Queue
Target user: field ops reviewer  
Primary components:
1. Auto-prioritized queue (highest impact first)
2. Columns:
   - Resident
   - Room
   - Time range
   - Current label
   - Suggested label
   - Why flagged (plain language)
   - Estimated impact (`Low / Medium / High`)
3. Bulk actions:
   - Accept
   - Edit and accept
   - Skip
4. Safe retrain button:
   - “Apply approved changes and run safe retrain”

### 3) Resident Timeline
Target user: care ops + reviewer  
Primary components:
1. Single-day timeline view (room events + activity)
2. Toggle:
   - “Care view” (default)
   - “Technical details” (expander)
3. Event overlays:
   - Sleep blocks
   - Bathroom events
   - Kitchen/living activity blocks
   - Home empty intervals
4. Correction tools:
   - segment-level relabel
   - queue for review

### 4) Weekly Report
Target user: team lead  
Primary components:
1. Global pilot mode banner (when enabled):
   - `Beta 5.5 currently active (Beta 6 in Shadow)`
2. KPI trend lines (7/14/30 day)
3. Exception summary:
   - residents with highest drift
   - repeated uncertainty spikes
4. Export options:
   - CSV
   - PDF-ready summary
5. `ML Health Snapshot` panel (summary + technical details expander)
   - `Balanced Score (macro-F1)` latest candidate and champion
   - `Transition Quality (transition F1)`
   - `Safety Drift Threshold` (effective)
   - `Stability Accuracy`
   - threshold source (`default / env override / room override / policy`)
   - status badge (`Healthy / Watch / Action Needed / Not Available`)
   - plain-language reminder of active care baseline (`Beta 5.5` or `Beta 6`) during pilot/cutover windows
6. Technical expander:
   - model-level metrics and gate reasons (raw details, JSON, traces)

### 5) Admin & Audit
Target user: admin/ML ops  
Primary components:
1. Config editor (validated)
2. Registry/version status
3. Training/release status timeline
4. `ML Health Snapshot` panel (full version with threshold provenance and room overrides)
   - provenance rows should expose direct config-editor links/actions when available
   - show source file path and owning role/team next to override keys
5. Correction audit log and rollback tools
6. Data export and resident data management

## Language and Content Rules
Default labels should avoid ML jargon.

Use:
1. “Model confidence is low” instead of “low calibration support”
2. “Update paused by safety checks” instead of “gate failed”
3. “Not enough examples yet” instead of “class support insufficient”

Reserve technical terms for expandable sections only.
When APIs provide both plain-language and technical reasons, default UI copies must use the plain-language field.

## Visual Behavior Rules
1. Show at most 5 top KPIs on default screen.
2. Use traffic-light states:
   - Green: normal
   - Amber: watch
   - Red: action needed
3. Show uncertainty explicitly as a badge.
4. Always include timestamp of latest data refresh.
5. During shadow mode, display a persistent global banner stating the active care baseline (`Beta 5.5 active`, `Beta 6 in Shadow`).

## Implementation Plan (Incremental, low risk)

### Phase UI-1: Navigation and labels
Files:
1. `backend/export_dashboard.py`

Changes:
1. Reorganize tabs to new IA labels.
2. Add plain-language helper mapping for status/reasons.
3. Keep existing backend logic intact.

Acceptance:
1. Existing functionality still accessible.
2. Default landing page is `Today`.

### Phase UI-2: Today + Review Queue pages
Files:
1. `backend/export_dashboard.py`
2. `backend/ml/event_metrics.py` (read adapters only if needed)
3. `backend/ml/home_empty_fusion.py` (read adapters only if needed)

Changes:
1. Introduce KPI cards and action list.
2. Add queue table with impact sorting.
3. Add routing rules so routine ML uncertainty goes to `Review Queue`, not `Today` alerts, unless care-blocking.
4. Add global pilot mode banner hook (feature-flagged).

Acceptance:
1. User can complete daily check in < 10 minutes.
2. User can submit queued corrections without opening technical panels.
3. Routine low-confidence items do not flood the `Needs Attention` list during pilot.

### Phase UI-3: Resident Timeline + Weekly Report
Files:
1. `backend/export_dashboard.py`
2. `backend/processors/adl_processor.py` (only if timeline payload needs extension)
3. `backend/health_server.py` (if adding `ml-snapshot` aggregate endpoint)

Changes:
1. Add event overlays and care/technical toggles.
2. Add weekly trend and exceptions summary.
3. Add `ML Health Snapshot` panel and technical expander in Weekly Report.
4. Show active-system banner during shadow mode and laddered rollout.

Acceptance:
1. Weekly drift review possible without reading raw logs.
2. Team lead can view macro-F1 / transition F1 / effective drift threshold from one panel.

### Phase UI-4: Admin and governance consolidation
Files:
1. `backend/export_dashboard.py`
2. `backend/ml/policy_config.py` (if config keys exposed)
3. `backend/health_server.py` (if threshold provenance endpoint is added)

Changes:
1. Move advanced controls to admin area.
2. Add config validation feedback in UI.
3. Add full `ML Health Snapshot` panel with threshold source/provenance.
4. Add direct navigation/actions from threshold provenance rows to config editor sections.

Acceptance:
1. Non-admin users do not need to interact with ML tuning controls.
2. Admins can verify threshold source (default/env/room/policy) without reading code.
3. Admins can navigate to the relevant threshold editor/file target from the panel.

## Testing Approach

### Functional tests (manual + scripted checklist)
1. Daily workflow completion test (Today -> open timeline -> queue correction)
2. Correction workflow test (queue -> apply -> retrain trigger)
3. Weekly report export test
4. Admin config validation test
5. `ML Health Snapshot` panel data reconciliation test (vs `walk-forward` and `promotion-gates` endpoints)
6. Shadow-mode banner test (all core pages display active-system banner correctly)
7. Alert-routing test (`Today` shows only care-actionable anomalies while routine ML uncertainty lands in `Review Queue`)

### Usability acceptance criteria
1. Non-ML ops can explain what action to take from Today screen without technical training.
2. 80% of correction tasks completed from Review Queue without entering Technical Details.
3. No critical action depends on understanding terms like F1, calibration, or gate.
4. Team lead can read exact macro-F1 / transition F1 / drift threshold values from the dedicated panel without leaving the page.
5. Ops can identify which system is actively driving care decisions during shadow mode in under 5 seconds.

### Regression criteria
1. Existing correction persistence still works.
2. Existing retrain trigger still works.
3. Existing audit export still works.

## Suggested Team Assignment
1. UI lead: IA and component redesign in `export_dashboard.py`
2. Ops representative: language review and workflow acceptance
3. ML/backend reviewer: ensure KPI semantics and confidence displays stay correct
4. QA: workflow and regression checklist signoff

## Immediate Next Implementation Ticket
1. Create `Today` page as the new default landing view.
2. Add “Needs Attention” cards with plain-language reasons and confidence.
3. Keep old technical sections in expanders during transition.
4. Define and implement `ML Health Snapshot` panel contract (`/health/model/ml-snapshot`) in parallel with UI wiring.
5. Add active-system shadow banner and alert-routing rules (`Today` vs `Review Queue`).
