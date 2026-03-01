# Knowledge Base Admin UI - Design Document

## Overview
A web-based admin interface for managing the Health Advisory Chatbot's knowledge base through JSON/CSV file uploads.

---

## 🎨 Main Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🏥 Health Advisory Chatbot                    Admin    Documentation  🔔  │
│                    KNOWLEDGE BASE MANAGEMENT                                │
├──────────────┬──────────────────────────────────────────────────────────────┤
│              │                                                               │
│  📁 SECTIONS │  ┌─────────────────────────────────────────────────────────┐ │
│              │  │  📊 Dashboard Overview                                   │ │
│  ○ Dashboard │  │                                                          │ │
│  ● Guidelines│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │ │
│  ○ Drugs     │  │  │ Guidelines  │ │    Drugs    │ │   Research  │        │ │
│  ○ Research  │  │  │    47       │ │   248       │ │    89       │        │ │
│  ○ FAQ       │  │  └─────────────┘ └─────────────┘ └─────────────┘        │ │
│  ○ Upload    │  │                                                          │ │
│  History     │  │  Recent Activity                                         │ │
│              │  │  ─────────────────                                       │ │
│  ─────────── │  │  🟢 Guidelines uploaded    2026-02-08 14:23    Dr. Chen  │ │
│              │  │  🟡 Drugs pending review   2026-02-08 11:15    Pharmacy  │ │
│  ⚙️ Settings │  │  🟢 Research papers added  2026-02-07 16:45    Research  │ │
│              │  └─────────────────────────────────────────────────────────┘ │ │
│              │                                                               │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 📤 File Upload Interface

### Step 1: Select Upload Type

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📤 Upload Knowledge Base Data                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Select Data Type:                                                          │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  📋          │  │  💊          │  │  📄          │  │  ❓          │    │
│  │ Guidelines   │  │   Drugs      │  │  Research    │  │     FAQ      │    │
│  │  (JSON)      │  │ (JSON/CSV)   │  │  (JSON/PDF)  │  │   (JSON)     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                             │
│  Current: 📋 Guidelines                                                     │
│                                                                             │
│  Expected format: JSON file with guideline definitions                      │
│  📥 Download Template  |  📖 View Schema Documentation                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Continue to Upload]  [Cancel]                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 2: File Upload Area

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📤 Upload Guidelines                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │                                                                     │   │
│  │              📁                                                     │   │
│  │                                                                     │   │
│  │         Drag & drop files here                                      │   │
│  │              or                                                     │   │
│  │        [Browse Files]                                               │   │
│  │                                                                     │   │
│  │   Supported: .json, .csv (max 10MB)                                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  📎 Selected: ags_falls_2023_updated.json (245 KB)                         │
│                                                                             │
│  Upload Options:                                                            │
│  ☑ Validate only (don't apply yet)                                          │
│  ☐ Overwrite existing entries with same ID                                  │
│  ☐ Create backup of current data                                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Upload & Validate]  [Cancel]                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 3: Validation Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ✅ Validation Complete - Ready to Apply                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  File: ags_falls_2023_updated.json                                          │
│                                                                             │
│  Summary:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ ✅ Schema Valid        │  Structure matches expected format          │  │
│  │ ✅ 47 Entries Found    │  15 new, 30 modified, 2 unchanged           │  │
│  │ ✅ No Duplicates       │  All IDs unique                             │  │
│  │ ⚠️  3 Warnings         │  See details below                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Changes Preview:                                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  🆕 NEW ENTRIES (15)                    [View All] [Download List]          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ID              │ Category          │ Evidence │ Status             │   │
│  ├─────────────────┼───────────────────┼──────────┼────────────────────┤   │
│  │ fall_prev_048   │ Home Safety       │ A        │ ✅ Valid           │   │
│  │ fall_prev_049   │ Exercise          │ B        │ ✅ Valid           │   │
│  │ fall_prev_050   │ Vision Check      │ A        │ ✅ Valid           │   │
│  └─────────────────┴───────────────────┴──────────┴────────────────────┘   │
│                                                                             │
│  ✏️ MODIFIED ENTRIES (30)               [View All] [Compare Changes]        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ID              │ Field Changed     │ Old Value      │ New Value    │   │
│  ├─────────────────┼───────────────────┼────────────────┼──────────────┤   │
│  │ fall_prev_001   │ evidence_level    │ B              │ A            │   │
│  │ fall_prev_015   │ recommendation    │ [View Diff]    │              │   │
│  │ fall_prev_022   │ source            │ AGS 2019       │ AGS 2023     │   │
│  └─────────────────┴───────────────────┴────────────────┴──────────────┘   │
│                                                                             │
│  ⚠️ WARNINGS (3)                                                            │
│  • Entry fall_prev_012: Evidence level 'C' used for strong recommendation   │
│  • Entry fall_prev_031: Source URL returns 404 (may be outdated)            │
│  • Entry fall_prev_045: Contraindications list empty                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Save to Staging]  [Apply to Production]  [Cancel]                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Guidelines Browser/Editor

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📋 Clinical Guidelines                              [+ Add New] [Upload]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Filters:                                                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │
│  │ Category ▼   │ │ Evidence ▼   │ │ Source ▼     │ │ Search...        │   │
│  │ All          │ │ All          │ │ All          │ │                  │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘   │
│                                                                             │
│  Active Filters: (None)                           [Clear All]               │
│                                                                             │
│  Showing 47 guidelines  [Export JSON]  [Export CSV]                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 🏠 Home Safety Modifications                                        │   │
│  │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │   │
│  │ ID: fall_prev_001    Category: Home Modifications                   │   │
│  │                                                                     │   │
│  │ Evidence Level: A  │  Source: AGS Falls Guidelines 2023             │   │
│  │                                                                     │   │
│  │ Recommendations:                                                    │   │
│  │ • Remove loose rugs and clutter from walkways                       │   │
│  │ • Install grab bars in bathroom (near toilet and shower)            │   │
│  │ • Improve lighting in hallways and staircases                       │   │
│  │ • Secure loose cables and cords                                     │   │
│  │                                                                     │   │
│  │ Tags: fall_risk, home_safety, environmental                         │   │
│  │                                                                     │   │
│  │ [Edit]  [View Source]  [🗑️ Delete]                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 🚶 Exercise & Physical Therapy                                      │   │
│  │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │   │
│  │ ID: fall_prev_002    Category: Exercise Intervention                │   │
│  │ ...                                                                 │   │
│  │ [Edit]  [View Source]  [🗑️ Delete]                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [← Previous]  Page 1 of 5  [Next →]                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 💊 Drug Database Manager

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  💊 Drug Database                                    [+ Add New] [Upload]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Search: [Lorazepam                              ]  [🔍 Search]            │
│                                                                             │
│  Filters: ACB Score ▼  |  Fall Risk ▼  |  Beers Criteria ▼  |  Class ▼     │
│                                                                             │
│  Quick Stats:                                                               │
│  Total Drugs: 248  |  High Risk: 47  |  Moderate Risk: 89  |  Low Risk: 112│
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Drug Name      │ Class        │ ACB │ Fall Risk │ Interactions │ ⚠️  │   │
│  ├────────────────┼──────────────┼─────┼───────────┼──────────────┼─────┤   │
│  │ Lorazepam      │ Benzodiazepine│ 3  │ 🔴 High   │     12       │ ⚠️  │   │
│  │ Diphenhydramine│ Antihistamine│  3  │ 🔴 High   │      8       │ ⚠️  │   │
│  │ Donepezil      │ Cholinesterase│ 1  │ 🟡 Mod    │      6       │     │   │
│  │ Metformin      │ Biguanide    │  0  │ 🟢 Low    │      3       │     │   │
│  └────────────────┴──────────────┴─────┴───────────┴──────────────┴─────┘   │
│                                                                             │
│  Selected: Lorazepam                                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 💊 Lorazepam (Ativan)                                               │   │
│  │ Benzodiazepine | ACB Score: 3 (High) | Beers Criteria: ⚠️ AVOID     │   │
│  │                                                                     │   │
│  │ INTERACTIONS:                                                       │   │
│  │ • Zolpidem - Major: Additive CNS depression → Avoid combination     │   │
│  │ • Morphine - Major: Respiratory depression risk → Reduce dose       │   │
│  │ • Alcohol - Severe: Dangerous sedation → Absolute contraindication  │   │
│  │                                                                     │   │
│  │ CONTRAINDICATIONS:                                                  │   │
│  │ • Sleep apnea                                                       │   │
│  │ • Severe respiratory insufficiency                                  │   │
│  │ • Myasthenia gravis                                                 │   │
│  │                                                                     │   │
│  │ [Edit Drug]  [View Interactions]  [🗑️ Delete]                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 Research Papers Manager

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📄 Research Papers                                  [+ Add New] [Upload]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Search: [sleep apnea cognitive decline         ]  [🔍 Search] [Advanced]  │
│                                                                             │
│  Tags: Sleep (23)  |  Cognition (18)  |  Falls (15)  |  Diabetes (12) ...   │
│                                                                             │
│  Evidence Level: Systematic Review ▼  |  Year: 2020-2025 ▼                  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 📄 Sleep disturbances and cognitive decline in elderly adults       │   │
│  │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │   │
│  │ 📊 Evidence: Cohort Study    |    📅 2024    |    ⭐ Confidence: 85% │   │
│  │                                                                     │   │
│  │ Authors: Smith J, Johnson A, et al.                                 │   │
│  │ Journal: Journal of Gerontology                                     │   │
│  │ PMID: 36801234  |  DOI: 10.1002/j.2024.001                          │   │
│  │                                                                     │   │
│  │ 🔑 KEY FINDINGS:                                                    │   │
│  │ • Poor sleep increases dementia risk by 30% (HR 1.30, 95% CI 1.15-1.47)│  │
│  │ • Sleep apnea treatment slows cognitive decline by 40%              │   │
│  │ • 7-8 hours optimal; <6 or >9 hours associated with worse outcomes  │   │
│  │                                                                     │   │
│  │ Tags: #sleep #cognitive_decline #dementia #elderly                  │   │
│  │                                                                     │   │
│  │ 📎 Full Text: [📄 View PDF]  |  🔗 [View on PubMed]                 │   │
│  │                                                                     │   │
│  │ [Edit]  [🗑️ Delete]  [📊 View Citation Stats]                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Load More Papers...]                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📜 Upload History & Audit Log

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📜 Upload History                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Filter: [All Types ▼]  |  [All Users ▼]  |  [Last 30 Days ▼]               │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Today                                                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  🟢 14:23  Guidelines    ags_falls_2023.json        Dr. Chen     47 entries │
│      Status: Applied ✓                                                      │
│                                                                             │
│  🟡 11:15  Drugs         new_drugs_batch.csv        Pharmacy    Pending    │
│      Status: Staging (awaiting approval)                                    │
│      [Review]  [Approve]  [Reject]                                          │
│                                                                             │
│  Yesterday                                                                  │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  🟢 16:45  Research      papers_q1_2026.json        Research    12 papers  │
│      Status: Applied ✓                                                      │
│                                                                             │
│  🟢 09:12  FAQ           common_questions_v2.json    Admin       5 entries  │
│      Status: Applied ✓                                                      │
│                                                                             │
│  2026-02-06                                                                 │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  🔴 15:30  Drugs         drug_updates.json          Dr. Smith   Failed     │
│      Status: Validation Failed                                              │
│      Error: Duplicate drug IDs found: lorazepam, zolpidem                   │
│      [View Details]  [Retry]                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Settings Page

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚙️ Knowledge Base Settings                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Upload Settings                                                     │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ ☑ Require validation before applying to production                  │   │
│  │ ☑ Auto-backup before applying changes                               │   │
│  │ ☑ Send email notification on successful upload                      │   │
│  │ ☑ Enable staging environment                                        │   │
│  │                                                                     │   │
│  │ Max file size: [10] MB                                              │   │
│  │ Keep upload history: [90] days                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Validation Rules                                                    │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ ☑ Strict schema validation (reject if unknown fields)               │   │
│  │ ☐ Require source URL for all guidelines                             │   │
│  │ ☑ Require PMID for research papers                                  │   │
│  │ ☑ Warn on evidence level mismatch                                   │   │
│  │ ☑ Check for broken URLs                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Backup & Restore                                                    │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ Last backup: 2026-02-08 03:00 AM                                    │   │
│  │                                                                     │   │
│  │ [Create Backup Now]  [Download Latest]  [Restore from Backup]       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                                    [Save Settings]  [Reset to Defaults]     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔔 Notification Examples

### Toast Notifications

```
┌─────────────────────────────────────────────────────────┐
│  ✅ Upload Successful                                     │
│  47 guidelines validated and saved to staging.            │
│                                     [View] [Dismiss]      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  ⚠️ Validation Warning                                    │
│  3 warnings found. Review before applying.                │
│                                     [Review] [Dismiss]    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  ❌ Upload Failed                                         │
│  Schema validation failed. Check file format.             │
│                                     [View Errors]         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎨 Design Specifications

### Color Scheme
| Element | Color | Hex |
|---------|-------|-----|
| Primary (Buttons) | Blue | `#2563EB` |
| Success | Green | `#10B981` |
| Warning | Yellow | `#F59E0B` |
| Error | Red | `#EF4444` |
| Background | Light Gray | `#F9FAFB` |
| Card | White | `#FFFFFF` |
| Text Primary | Dark Gray | `#1F2937` |
| Text Secondary | Medium Gray | `#6B7280` |

### Typography
| Element | Font | Size | Weight |
|---------|------|------|--------|
| Page Title | System/Sans | 24px | Bold |
| Section Header | System/Sans | 18px | Semibold |
| Card Title | System/Sans | 16px | Semibold |
| Body Text | System/Sans | 14px | Normal |
| Labels | System/Sans | 12px | Medium |

### Spacing
- Card padding: `24px`
- Card gap: `16px`
- Section margin: `32px`
- Input height: `40px`
- Button height: `40px` (primary), `32px` (small)

---

## 🔄 User Workflows

### Workflow 1: Upload New Guidelines
1. Click [Upload] button on Guidelines page
2. Select "Guidelines" data type
3. Download template (optional)
4. Drag & drop or browse for JSON file
5. Click [Upload & Validate]
6. Review validation results
7. Click [Save to Staging] or [Apply to Production]
8. See success notification

### Workflow 2: Edit Single Entry
1. Browse guidelines list
2. Click [Edit] on desired guideline
3. Edit fields in modal/form
4. Click [Save Changes]
5. Changes applied immediately (with backup)

### Workflow 3: Bulk Drug Update
1. Export current drug database as CSV
2. Edit in Excel/LibreOffice
3. Upload CSV file
4. Review diff (changes highlighted)
5. Approve and apply

---

## 📱 Responsive Behavior

| Breakpoint | Layout |
|------------|--------|
| Desktop (>1200px) | Sidebar + Main content |
| Tablet (768-1200px) | Collapsible sidebar |
| Mobile (<768px) | Bottom nav, stacked cards |
