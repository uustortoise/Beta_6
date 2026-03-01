# Knowledge Base Admin UI

A web-based interface for managing the Health Advisory Chatbot's knowledge base.

## Features

- 📤 **File Upload** - Upload JSON files for guidelines, drugs, and FAQ
- ✅ **Validation** - Automatic schema validation with detailed error messages
- 📊 **Dashboard** - Overview of knowledge base content
- 🔍 **Search & Filter** - Browse and filter guidelines, drugs, FAQ
- 📜 **Activity Log** - Audit trail of all changes
- 📥 **Templates** - Download starter templates for each data type
- 💾 **Backup** - Create backups of the knowledge base

## Quick Start

### Option 1: Standalone Server (Recommended for testing)

```bash
cd health_advisory_chatbot/admin
python server.py
```

Open http://localhost:8080 in your browser.

### Option 2: Integrate with existing backend

Add the admin routes to your FastAPI or Flask app:

```python
from admin.api.admin_routes import create_admin_routes

# For FastAPI
app = FastAPI()
create_admin_routes(app)

# For Flask
app = Flask(__name__)
create_admin_routes(app)
```

Then serve the frontend files from `admin/frontend/`.

## Data File Formats

### Clinical Guidelines (JSON)

```json
{
  "metadata": {
    "version": "1.0.0",
    "last_updated": "2026-02-08"
  },
  "guidelines": [
    {
      "id": "unique_id",
      "source": "AGS_BEERS",
      "version": "2023",
      "category": "medication_safety",
      "title": "Guideline Title",
      "description": "Description",
      "evidence_grade": "A",
      "action_steps": ["step 1", "step 2"]
    }
  ]
}
```

### Drug Database (JSON or CSV)

```json
{
  "drugs": [
    {
      "id": "drug_lorazepam",
      "drug_name": "Lorazepam",
      "generic_name": "lorazepam",
      "drug_class": "benzodiazepine",
      "acb_score": 3,
      "fall_risk": "high",
      "interactions": [...]
    }
  ]
}
```

### FAQ (JSON)

```json
{
  "faqs": [
    {
      "id": "faq_001",
      "question_pattern": "Why do I wake up at night?",
      "response": "Response text...",
      "category": "sleep",
      "keywords": ["night", "wake"]
    }
  ]
}
```

## Directory Structure

```
knowledge_base_data/
├── guidelines/
│   └── clinical_guidelines.json
├── drugs/
│   └── drug_database.json
├── faq/
│   └── common_questions.json
├── templates/
│   ├── guideline_template.json
│   ├── drug_template.json
│   └── faq_template.json
└── backups/
    └── backup_20260208_120000/
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/stats` | GET | Get knowledge base statistics |
| `/api/admin/guidelines` | GET | Get guidelines (with filters) |
| `/api/admin/upload` | POST | Upload and validate file |
| `/api/admin/activity` | GET | Get activity log |
| `/api/admin/templates/{type}` | GET | Download template |
| `/api/admin/backup` | POST | Create backup |
| `/api/admin/categories` | GET | Get available categories |

## Validation Rules

### Guidelines
- Required: `id`, `source`, `title`, `description`, `evidence_grade`
- Evidence grade must be: A, B, C, or D
- No duplicate IDs allowed

### Drugs
- Required: `id`, `drug_name`, `generic_name`, `drug_class`
- ACB score must be 0-3
- Fall risk must be: low, moderate, or high

### FAQ
- Required: `id`, `question_pattern`, `response`, `category`
- Response cannot be empty
- No duplicate IDs allowed

## Screenshots

### Dashboard
- Overview cards showing counts
- Recent activity feed

### Upload
- Drag & drop zone
- File type selection
- Validation results display

### Guidelines Browser
- Filter by category, source
- Search functionality
- Table view with badges

## Future Enhancements

- [ ] Edit individual entries in browser
- [ ] CSV import for bulk drug updates
- [ ] Version control / rollback
- [ ] User authentication
- [ ] Approval workflow
- [ ] Research paper PDF upload
- [ ] Integration with vector database
