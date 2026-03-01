"""
Knowledge Base Admin API Routes

Provides REST endpoints for:
- File upload and validation
- Knowledge base CRUD operations
- Audit logging
- Backup/restore
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid

# FastAPI imports
try:
    from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
    from fastapi.responses import JSONResponse, FileResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Flask fallback - handle gracefully if Flask also not installed
    try:
        from flask import Blueprint, request, jsonify, send_file
        FLASK_AVAILABLE = True
    except ImportError:
        FLASK_AVAILABLE = False


# Define base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "knowledge_base_data"
GUIDELINES_DIR = DATA_DIR / "guidelines"
DRUGS_DIR = DATA_DIR / "drugs"
RESEARCH_DIR = DATA_DIR / "research"
FAQ_DIR = DATA_DIR / "faq"
TEMPLATES_DIR = DATA_DIR / "templates"
BACKUP_DIR = DATA_DIR / "backups"

# Ensure directories exist
for dir_path in [GUIDELINES_DIR, DRUGS_DIR, RESEARCH_DIR, FAQ_DIR, BACKUP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Pydantic Models
# ============================================================================

class UploadResponse(BaseModel):
    """Response model for file uploads."""
    success: bool
    message: str
    file_id: Optional[str] = None
    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseStats(BaseModel):
    """Statistics for knowledge base content."""
    guidelines_count: int
    drugs_count: int
    research_count: int
    faq_count: int
    last_updated: Optional[str] = None


class GuidelineSummary(BaseModel):
    """Summary of a guideline entry."""
    id: str
    title: str
    category: str
    source: str
    evidence_grade: str


class ActivityLogEntry(BaseModel):
    """Activity log entry for audit trail."""
    id: str
    timestamp: str
    action: str
    user: str
    file_type: str
    filename: str
    status: str
    message: Optional[str] = None


# ============================================================================
# Activity Logger
# ============================================================================

class ActivityLogger:
    """Simple file-based activity logger."""
    
    def __init__(self):
        self.log_file = DATA_DIR / "activity_log.json"
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        if not self.log_file.exists():
            self.log_file.write_text(json.dumps({"entries": []}, indent=2))
    
    def log(self, action: str, user: str, file_type: str, filename: str, 
            status: str, message: Optional[str] = None):
        """Log an activity."""
        data = json.loads(self.log_file.read_text())
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "file_type": file_type,
            "filename": filename,
            "status": status,
            "message": message
        }
        data["entries"].insert(0, entry)  # Newest first
        
        # Keep only last 1000 entries
        data["entries"] = data["entries"][:1000]
        
        self.log_file.write_text(json.dumps(data, indent=2))
        return entry
    
    def get_entries(self, limit: int = 50) -> List[Dict]:
        """Get recent activity entries."""
        if not self.log_file.exists():
            return []
        data = json.loads(self.log_file.read_text())
        return data.get("entries", [])[:limit]


activity_logger = ActivityLogger()


# ============================================================================
# Validation Functions
# ============================================================================

def validate_guidelines_json(data: Dict) -> tuple[List[str], List[str]]:
    """Validate guidelines JSON structure. Returns (errors, warnings)."""
    errors = []
    warnings = []
    
    # Check required top-level keys
    if "guidelines" not in data:
        errors.append("Missing 'guidelines' array")
        return errors, warnings
    
    if not isinstance(data["guidelines"], list):
        errors.append("'guidelines' must be an array")
        return errors, warnings
    
    # Validate each guideline
    seen_ids = set()
    for idx, guideline in enumerate(data["guidelines"]):
        prefix = f"Guideline #{idx + 1}"
        
        # Required fields
        required = ["id", "source", "title", "description", "evidence_grade"]
        for field in required:
            if field not in guideline:
                errors.append(f"{prefix}: Missing required field '{field}'")
        
        # Check for duplicate IDs
        if "id" in guideline:
            if guideline["id"] in seen_ids:
                errors.append(f"{prefix}: Duplicate ID '{guideline['id']}'")
            seen_ids.add(guideline["id"])
        
        # Validate evidence grade
        if "evidence_grade" in guideline:
            if guideline["evidence_grade"] not in ["A", "B", "C", "D"]:
                warnings.append(
                    f"{prefix}: Evidence grade '{guideline['evidence_grade']}' "
                    f"should be A, B, C, or D"
                )
        
        # Validate category
        valid_categories = [
            "medication_safety", "fall_prevention", "cognitive_health",
            "mobility", "nutrition", "sensory", "sleep", "cardiovascular"
        ]
        if "category" in guideline:
            if guideline["category"] not in valid_categories:
                warnings.append(
                    f"{prefix}: Unusual category '{guideline['category']}'"
                )
    
    return errors, warnings


def validate_drugs_json(data: Dict) -> tuple[List[str], List[str]]:
    """Validate drugs JSON structure. Returns (errors, warnings)."""
    errors = []
    warnings = []
    
    if "drugs" not in data:
        errors.append("Missing 'drugs' array")
        return errors, warnings
    
    if not isinstance(data["drugs"], list):
        errors.append("'drugs' must be an array")
        return errors, warnings
    
    seen_ids = set()
    for idx, drug in enumerate(data["drugs"]):
        prefix = f"Drug #{idx + 1}"
        
        required = ["id", "drug_name", "generic_name", "drug_class"]
        for field in required:
            if field not in drug:
                errors.append(f"{prefix}: Missing required field '{field}'")
        
        if "id" in drug:
            if drug["id"] in seen_ids:
                errors.append(f"{prefix}: Duplicate ID '{drug['id']}'")
            seen_ids.add(drug["id"])
        
        # Validate ACB score
        if "acb_score" in drug:
            if not 0 <= drug["acb_score"] <= 3:
                warnings.append(
                    f"{prefix}: ACB score {drug['acb_score']} should be 0-3"
                )
        
        # Validate fall risk
        if "fall_risk" in drug:
            if drug["fall_risk"] not in ["low", "moderate", "high"]:
                warnings.append(
                    f"{prefix}: Fall risk should be 'low', 'moderate', or 'high'"
                )
    
    return errors, warnings


def validate_faq_json(data: Dict) -> tuple[List[str], List[str]]:
    """Validate FAQ JSON structure. Returns (errors, warnings)."""
    errors = []
    warnings = []
    
    if "faqs" not in data:
        errors.append("Missing 'faqs' array")
        return errors, warnings
    
    seen_ids = set()
    for idx, faq in enumerate(data["faqs"]):
        prefix = f"FAQ #{idx + 1}"
        
        required = ["id", "question_pattern", "response", "category"]
        for field in required:
            if field not in faq:
                errors.append(f"{prefix}: Missing required field '{field}'")
        
        if "id" in faq:
            if faq["id"] in seen_ids:
                errors.append(f"{prefix}: Duplicate ID '{faq['id']}'")
            seen_ids.add(faq["id"])
        
        # Check for empty response
        if "response" in faq and not faq["response"].strip():
            errors.append(f"{prefix}: Response cannot be empty")
    
    return errors, warnings


# ============================================================================
# API Routes (FastAPI)
# ============================================================================

if FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/api/admin", tags=["admin"])
    
    @router.get("/stats", response_model=KnowledgeBaseStats)
    async def get_stats():
        """Get knowledge base statistics."""
        stats = {"guidelines_count": 0, "drugs_count": 0, "research_count": 0, "faq_count": 0}
        
        # Count guidelines
        guidelines_file = GUIDELINES_DIR / "clinical_guidelines.json"
        if guidelines_file.exists():
            data = json.loads(guidelines_file.read_text())
            stats["guidelines_count"] = len(data.get("guidelines", []))
        
        # Count drugs
        drugs_file = DRUGS_DIR / "drug_database.json"
        if drugs_file.exists():
            data = json.loads(drugs_file.read_text())
            stats["drugs_count"] = len(data.get("drugs", []))
        
        # Count FAQ
        faq_file = FAQ_DIR / "common_questions.json"
        if faq_file.exists():
            data = json.loads(faq_file.read_text())
            stats["faq_count"] = len(data.get("faqs", []))
        
        stats["last_updated"] = datetime.now().isoformat()
        return stats
    
    @router.get("/guidelines", response_model=List[GuidelineSummary])
    async def get_guidelines(
        category: Optional[str] = Query(None),
        source: Optional[str] = Query(None),
        search: Optional[str] = Query(None)
    ):
        """Get guidelines with optional filtering."""
        guidelines_file = GUIDELINES_DIR / "clinical_guidelines.json"
        if not guidelines_file.exists():
            return []
        
        data = json.loads(guidelines_file.read_text())
        guidelines = data.get("guidelines", [])
        
        # Apply filters
        if category:
            guidelines = [g for g in guidelines if g.get("category") == category]
        if source:
            guidelines = [g for g in guidelines if g.get("source") == source]
        if search:
            search_lower = search.lower()
            guidelines = [
                g for g in guidelines 
                if search_lower in g.get("title", "").lower() 
                or search_lower in g.get("description", "").lower()
            ]
        
        return [
            {
                "id": g["id"],
                "title": g["title"],
                "category": g["category"],
                "source": g["source"],
                "evidence_grade": g["evidence_grade"]
            }
            for g in guidelines
        ]
    
    @router.post("/upload", response_model=UploadResponse)
    async def upload_file(
        file: UploadFile = File(...),
        file_type: str = Form(...),
        validate_only: bool = Form(False)
    ):
        """Upload and validate a knowledge base file."""
        
        # Validate file extension
        if not file.filename.endswith(('.json', '.csv')):
            raise HTTPException(400, "Only JSON and CSV files are supported")
        
        # Read and parse JSON
        try:
            content = await file.read()
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return UploadResponse(
                success=False,
                message=f"Invalid JSON: {str(e)}",
                validation_errors=[f"JSON parse error: {str(e)}"]
            )
        
        # Validate based on file type
        if file_type == "guidelines":
            errors, warnings = validate_guidelines_json(data)
        elif file_type == "drugs":
            errors, warnings = validate_drugs_json(data)
        elif file_type == "faq":
            errors, warnings = validate_faq_json(data)
        else:
            raise HTTPException(400, f"Unknown file type: {file_type}")
        
        # If only validating, return results without saving
        if validate_only:
            activity_logger.log(
                "VALIDATE", "admin", file_type, file.filename,
                "VALIDATED" if not errors else "VALIDATION_FAILED"
            )
            return UploadResponse(
                success=len(errors) == 0,
                message="Validation complete" if not errors else "Validation failed",
                validation_errors=errors,
                warnings=warnings,
                stats={"entries_found": len(data.get(file_type, []))}
            )
        
        # If errors exist, don't save
        if errors:
            activity_logger.log(
                "UPLOAD_REJECTED", "admin", file_type, file.filename,
                "REJECTED", f"{len(errors)} validation errors"
            )
            return UploadResponse(
                success=False,
                message="File not saved due to validation errors",
                validation_errors=errors,
                warnings=warnings
            )
        
        # Save file
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"{file_type}_{timestamp}_{file_id}.json"
        
        if file_type == "guidelines":
            save_path = GUIDELINES_DIR / save_filename
        elif file_type == "drugs":
            save_path = DRUGS_DIR / save_filename
        elif file_type == "faq":
            save_path = FAQ_DIR / save_filename
        else:
            save_path = DATA_DIR / save_filename
        
        save_path.write_text(json.dumps(data, indent=2))
        
        # Log activity
        activity_logger.log(
            "UPLOAD", "admin", file_type, file.filename,
            "SUCCESS", f"Saved as {save_filename}"
        )
        
        return UploadResponse(
            success=True,
            message=f"File uploaded successfully as {save_filename}",
            file_id=file_id,
            warnings=warnings,
            stats={
                "entries_found": len(data.get(file_type if file_type != "guidelines" else "guidelines", [])),
                "filename": save_filename
            }
        )
    
    @router.get("/activity")
    async def get_activity(limit: int = Query(50, le=100)):
        """Get recent activity log."""
        return activity_logger.get_entries(limit)
    
    @router.get("/templates/{template_name}")
    async def download_template(template_name: str):
        """Download a template file."""
        template_map = {
            "guidelines": "guideline_template.json",
            "drugs": "drug_template.json",
            "faq": "faq_template.json"
        }
        
        if template_name not in template_map:
            raise HTTPException(404, "Template not found")
        
        template_path = TEMPLATES_DIR / template_map[template_name]
        if not template_path.exists():
            raise HTTPException(404, "Template file not found")
        
        return FileResponse(
            template_path,
            media_type="application/json",
            filename=template_map[template_name]
        )
    
    @router.post("/backup")
    async def create_backup():
        """Create a backup of current knowledge base."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = BACKUP_DIR / f"backup_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        # Copy current files
        for source_dir, name in [
            (GUIDELINES_DIR, "guidelines"),
            (DRUGS_DIR, "drugs"),
            (FAQ_DIR, "faq")
        ]:
            if source_dir.exists():
                shutil.copytree(source_dir, backup_dir / name, dirs_exist_ok=True)
        
        activity_logger.log(
            "BACKUP", "admin", "all", "full_backup",
            "SUCCESS", f"Backup created: {backup_dir.name}"
        )
        
        return {"success": True, "backup_id": backup_dir.name}
    
    @router.get("/categories")
    async def get_categories():
        """Get available categories and sources for filtering."""
        return {
            "guideline_categories": [
                "medication_safety", "fall_prevention", "cognitive_health",
                "mobility", "nutrition", "sensory", "sleep", "cardiovascular"
            ],
            "guideline_sources": [
                "AGS_BEERS", "AGS_FALLS", "WHO_ICOPE", "NICE", "AASM", "ADA"
            ],
            "drug_classes": [
                "benzodiazepine", "antihistamine", "ssri", "nsaid",
                "biguanide", "sulfonylurea", "ace_inhibitor", "beta_blocker"
            ],
            "faq_categories": [
                "sleep", "falls", "medication", "cognitive", 
                "nutrition", "exercise", "general"
            ]
        }

elif FLASK_AVAILABLE:
    # Flask Blueprint fallback
    router = Blueprint('admin', __name__, url_prefix='/api/admin')
    
    @router.route('/stats', methods=['GET'])
    def get_stats():
        stats = {"guidelines_count": 0, "drugs_count": 0, "research_count": 0, "faq_count": 0}
        
        guidelines_file = GUIDELINES_DIR / "clinical_guidelines.json"
        if guidelines_file.exists():
            data = json.loads(guidelines_file.read_text())
            stats["guidelines_count"] = len(data.get("guidelines", []))
        
        drugs_file = DRUGS_DIR / "drug_database.json"
        if drugs_file.exists():
            data = json.loads(drugs_file.read_text())
            stats["drugs_count"] = len(data.get("drugs", []))
        
        faq_file = FAQ_DIR / "common_questions.json"
        if faq_file.exists():
            data = json.loads(faq_file.read_text())
            stats["faq_count"] = len(data.get("faqs", []))
        
        stats["last_updated"] = datetime.now().isoformat()
        return jsonify(stats)

else:
    # Neither FastAPI nor Flask available - create dummy router
    router = None
    

def create_admin_routes(app):
    """Register admin routes with the app."""
    if FASTAPI_AVAILABLE:
        app.include_router(router)
    elif FLASK_AVAILABLE:
        app.register_blueprint(router)
    else:
        raise ImportError("Either FastAPI or Flask must be installed")
