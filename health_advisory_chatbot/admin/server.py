"""
Simple server for Knowledge Base Admin UI

Usage:
    python server.py
    
Then open http://localhost:8080 in your browser
"""

import http.server
import socketserver
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cgi
import io

# Import admin routes
from api.admin_routes import (
    GUIDELINES_DIR, DRUGS_DIR, RESEARCH_DIR, FAQ_DIR, TEMPLATES_DIR,
    activity_logger, validate_guidelines_json, validate_drugs_json, validate_faq_json,
    DATA_DIR
)

PORT = 8080
FRONTEND_DIR = Path(__file__).parent / "frontend"


class AdminHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for admin API and static files."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # API routes
        if path.startswith('/api/admin/'):
            self.handle_api_get(path, parse_qs(parsed.query))
            return
        
        # Static files
        if path == '/' or path == '/index.html':
            self.serve_file(FRONTEND_DIR / 'index.html', 'text/html')
        else:
            # Try to serve from frontend directory
            file_path = FRONTEND_DIR / path.lstrip('/')
            if file_path.exists():
                content_type = 'text/css' if path.endswith('.css') else 'application/javascript'
                self.serve_file(file_path, content_type)
            else:
                self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path.startswith('/api/admin/'):
            self.handle_api_post(path)
            return
        
        self.send_error(404)
    
    def do_PUT(self):
        """Handle PUT requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        if path.startswith('/api/admin/'):
            self.handle_api_put(path, query)
            return
        
        self.send_error(404)
    
    def do_DELETE(self):
        """Handle DELETE requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        if path.startswith('/api/admin/'):
            self.handle_api_delete(path, query)
            return
        
        self.send_error(404)
    
    def serve_file(self, file_path, content_type):
        """Serve a file with proper headers."""
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))
    
    def send_json(self, data, status=200):
        """Send JSON response."""
        content = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)
    
    def handle_api_get(self, path, query):
        """Handle API GET requests."""
        try:
            if path == '/api/admin/stats':
                self.handle_stats()
            elif path == '/api/admin/guidelines':
                self.handle_guidelines(query)
            elif path == '/api/admin/research':
                self.handle_research(query)
            elif path == '/api/admin/research-detail':
                self.handle_research_detail(query)
            elif path == '/api/admin/drugs':
                self.handle_drugs(query)
            elif path == '/api/admin/drug-detail':
                self.handle_drug_detail(query)
            elif path == '/api/admin/faq':
                self.handle_faq(query)
            elif path == '/api/admin/faq-detail':
                self.handle_faq_detail(query)
            elif path == '/api/admin/guideline-detail':
                self.handle_guideline_detail(query)
            elif path == '/api/admin/activity':
                self.handle_activity(query)
            elif path == '/api/admin/categories':
                self.handle_categories()
            elif path.startswith('/api/admin/templates/'):
                template_name = path.split('/')[-1]
                self.handle_template(template_name)
            else:
                self.send_error(404)
        except Exception as e:
            self.send_json({'error': str(e)}, 500)
    
    def handle_api_post(self, path):
        """Handle API POST requests."""
        try:
            if path == '/api/admin/upload':
                self.handle_upload()
            elif path == '/api/admin/backup':
                self.handle_backup()
            elif path == '/api/admin/guideline':
                self.handle_create_entry('guideline')
            elif path == '/api/admin/drug':
                self.handle_create_entry('drug')
            elif path == '/api/admin/research':
                self.handle_create_entry('research')
            elif path == '/api/admin/faq':
                self.handle_create_entry('faq')
            else:
                self.send_error(404)
        except Exception as e:
            self.send_json({'error': str(e)}, 500)
    
    def handle_api_put(self, path, query):
        """Handle API PUT requests."""
        try:
            if path == '/api/admin/guideline':
                self.handle_update_entry('guideline', query)
            elif path == '/api/admin/drug':
                self.handle_update_entry('drug', query)
            elif path == '/api/admin/research':
                self.handle_update_entry('research', query)
            elif path == '/api/admin/faq':
                self.handle_update_entry('faq', query)
            else:
                self.send_error(404)
        except Exception as e:
            self.send_json({'error': str(e)}, 500)
    
    def handle_api_delete(self, path, query):
        """Handle API DELETE requests."""
        try:
            if path == '/api/admin/guideline':
                self.handle_delete_entry('guideline', query)
            elif path == '/api/admin/drug':
                self.handle_delete_entry('drug', query)
            elif path == '/api/admin/research':
                self.handle_delete_entry('research', query)
            elif path == '/api/admin/faq':
                self.handle_delete_entry('faq', query)
            else:
                self.send_error(404)
        except Exception as e:
            self.send_json({'error': str(e)}, 500)
    
    def handle_stats(self):
        """Get knowledge base statistics."""
        stats = {
            "guidelines_count": 0,
            "drugs_count": 0,
            "research_count": 0,
            "faq_count": 0
        }
        
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
        
        # Count research papers
        research_file = RESEARCH_DIR / "research_papers.json"
        if research_file.exists():
            data = json.loads(research_file.read_text())
            stats["research_count"] = len(data.get("papers", []))
        
        from datetime import datetime
        stats["last_updated"] = datetime.now().isoformat()
        self.send_json(stats)
    
    def handle_guidelines(self, query):
        """Get guidelines with filtering."""
        guidelines_file = GUIDELINES_DIR / "clinical_guidelines.json"
        if not guidelines_file.exists():
            self.send_json([])
            return
        
        data = json.loads(guidelines_file.read_text())
        guidelines = data.get("guidelines", [])
        
        # Apply filters
        category = query.get('category', [None])[0]
        source = query.get('source', [None])[0]
        search = query.get('search', [None])[0]
        
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
        
        # Return summary
        result = [
            {
                "id": g["id"],
                "title": g["title"],
                "category": g["category"],
                "source": g["source"],
                "evidence_grade": g["evidence_grade"]
            }
            for g in guidelines
        ]
        self.send_json(result)
    
    def handle_research(self, query):
        """Get research papers with filtering."""
        research_file = RESEARCH_DIR / "research_papers.json"
        if not research_file.exists():
            self.send_json([])
            return
        
        data = json.loads(research_file.read_text())
        papers = data.get("papers", [])
        
        # Apply filters
        tag = query.get('tag', [None])[0]
        year = query.get('year', [None])[0]
        search = query.get('search', [None])[0]
        
        if tag:
            papers = [p for p in papers if tag in p.get("tags", [])]
        if year:
            papers = [p for p in papers if str(p.get("year")) == year]
        if search:
            search_lower = search.lower()
            papers = [
                p for p in papers 
                if search_lower in p.get("title", "").lower() 
                or search_lower in p.get("abstract_summary", "").lower()
            ]
        
        # Return summary
        result = [
            {
                "id": p["id"],
                "pmid": p.get("pmid"),
                "title": p["title"],
                "authors": p.get("authors", []),
                "journal": p.get("journal"),
                "year": p.get("year"),
                "evidence_level": p.get("evidence_level"),
                "tags": p.get("tags", [])
            }
            for p in papers
        ]
        self.send_json(result)
    
    def handle_research_detail(self, query):
        """Get full details of a research paper by ID."""
        research_file = RESEARCH_DIR / "research_papers.json"
        if not research_file.exists():
            self.send_json({'error': 'Research file not found'}, 404)
            return
        
        paper_id = query.get('id', [None])[0]
        if not paper_id:
            self.send_json({'error': 'Missing paper ID'}, 400)
            return
        
        data = json.loads(research_file.read_text())
        papers = data.get("papers", [])
        
        paper = next((p for p in papers if p.get("id") == paper_id), None)
        if not paper:
            self.send_json({'error': 'Paper not found'}, 404)
            return
        
        self.send_json(paper)
    
    def handle_guideline_detail(self, query):
        """Get full details of a clinical guideline by ID."""
        guidelines_file = GUIDELINES_DIR / "clinical_guidelines.json"
        if not guidelines_file.exists():
            self.send_json({'error': 'Guidelines file not found'}, 404)
            return
        
        guideline_id = query.get('id', [None])[0]
        if not guideline_id:
            self.send_json({'error': 'Missing guideline ID'}, 400)
            return
        
        data = json.loads(guidelines_file.read_text())
        guidelines = data.get("guidelines", [])
        
        guideline = next((g for g in guidelines if g.get("id") == guideline_id), None)
        if not guideline:
            self.send_json({'error': 'Guideline not found'}, 404)
            return
        
        self.send_json(guideline)
    
    def handle_drugs(self, query):
        """Get drugs with filtering."""
        drugs_file = DRUGS_DIR / "drug_database.json"
        if not drugs_file.exists():
            self.send_json([])
            return
        
        data = json.loads(drugs_file.read_text())
        drugs = data.get("drugs", [])
        
        # Apply filters
        drug_class = query.get('class', [None])[0]
        beers_only = query.get('beers_only', [None])[0]
        search = query.get('search', [None])[0]
        
        if drug_class:
            drugs = [d for d in drugs if d.get("drug_class") == drug_class]
        if beers_only == 'true':
            drugs = [d for d in drugs if d.get("beers_criteria")]
        if search:
            search_lower = search.lower()
            drugs = [
                d for d in drugs 
                if search_lower in d.get("drug_name", "").lower() 
                or search_lower in d.get("generic_name", "").lower()
                or search_lower in d.get("drug_class", "").lower()
            ]
        
        # Return summary
        result = [
            {
                "id": d["id"],
                "drug_name": d["drug_name"],
                "generic_name": d.get("generic_name"),
                "drug_class": d.get("drug_class"),
                "acb_score": d.get("acb_score"),
                "fall_risk": d.get("fall_risk"),
                "beers_criteria": d.get("beers_criteria", False)
            }
            for d in drugs
        ]
        self.send_json(result)
    
    def handle_drug_detail(self, query):
        """Get full details of a drug by ID."""
        drugs_file = DRUGS_DIR / "drug_database.json"
        if not drugs_file.exists():
            self.send_json({'error': 'Drug database not found'}, 404)
            return
        
        drug_id = query.get('id', [None])[0]
        if not drug_id:
            self.send_json({'error': 'Missing drug ID'}, 400)
            return
        
        data = json.loads(drugs_file.read_text())
        drugs = data.get("drugs", [])
        
        drug = next((d for d in drugs if d.get("id") == drug_id), None)
        if not drug:
            self.send_json({'error': 'Drug not found'}, 404)
            return
        
        self.send_json(drug)
    
    def handle_faq(self, query):
        """Get FAQ entries with filtering."""
        faq_file = FAQ_DIR / "common_questions.json"
        if not faq_file.exists():
            self.send_json([])
            return
        
        data = json.loads(faq_file.read_text())
        faqs = data.get("faqs", [])
        
        # Apply filters
        category = query.get('category', [None])[0]
        search = query.get('search', [None])[0]
        
        if category:
            faqs = [f for f in faqs if f.get("category") == category]
        if search:
            search_lower = search.lower()
            faqs = [
                f for f in faqs 
                if search_lower in f.get("question_pattern", "").lower() 
                or search_lower in f.get("response", "").lower()
                or any(search_lower in kw.lower() for kw in f.get("keywords", []))
            ]
        
        # Return summary
        result = [
            {
                "id": f["id"],
                "question_pattern": f["question_pattern"],
                "category": f.get("category"),
                "keywords": f.get("keywords", []),
                "evidence_level": f.get("evidence_level")
            }
            for f in faqs
        ]
        self.send_json(result)
    
    def handle_faq_detail(self, query):
        """Get full details of an FAQ entry by ID."""
        faq_file = FAQ_DIR / "common_questions.json"
        if not faq_file.exists():
            self.send_json({'error': 'FAQ file not found'}, 404)
            return
        
        faq_id = query.get('id', [None])[0]
        if not faq_id:
            self.send_json({'error': 'Missing FAQ ID'}, 400)
            return
        
        data = json.loads(faq_file.read_text())
        faqs = data.get("faqs", [])
        
        faq = next((f for f in faqs if f.get("id") == faq_id), None)
        if not faq:
            self.send_json({'error': 'FAQ not found'}, 404)
            return
        
        self.send_json(faq)
    
    def handle_activity(self, query):
        """Get activity log."""
        limit = int(query.get('limit', [50])[0])
        entries = activity_logger.get_entries(limit)
        self.send_json(entries)
    
    def handle_categories(self):
        """Get available categories."""
        self.send_json({
            "guideline_categories": [
                "medication_safety", "fall_prevention", "cognitive_health",
                "mobility", "nutrition", "sensory", "sleep", "cardiovascular"
            ],
            "guideline_sources": [
                "AGS_BEERS", "AGS_FALLS", "WHO_ICOPE", "NICE", "AASM", "ADA"
            ],
            "drug_classes": [
                "benzodiazepine", "antihistamine", "ssri", "nsaid",
                "biguanide", "sulfonylurea"
            ],
            "faq_categories": [
                "sleep", "falls", "medication", "cognitive", 
                "nutrition", "exercise", "general"
            ]
        })
    
    def handle_template(self, template_name):
        """Download a template file."""
        template_map = {
            "guidelines": "guideline_template.json",
            "drugs": "drug_template.json",
            "faq": "faq_template.json"
        }
        
        if template_name not in template_map:
            self.send_error(404, "Template not found")
            return
        
        template_path = TEMPLATES_DIR / template_map[template_name]
        if not template_path.exists():
            self.send_error(404, "Template file not found")
            return
        
        self.serve_file(template_path, 'application/json')
    
    def handle_upload(self):
        """Handle file upload."""
        content_type = self.headers.get('Content-Type', '')
        
        if not content_type.startswith('multipart/form-data'):
            self.send_json({'error': 'Expected multipart/form-data'}, 400)
            return
        
        # Parse multipart form data
        environ = {'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type}
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ=environ
        )
        
        # Get form fields
        file_type = form.getvalue('file_type', '')
        validate_only = form.getvalue('validate_only', 'false').lower() == 'true'
        
        if 'file' not in form:
            self.send_json({'error': 'No file provided'}, 400)
            return
        
        file_item = form['file']
        if not file_item.filename:
            self.send_json({'error': 'No file selected'}, 400)
            return
        
        # Read and parse JSON
        try:
            content = file_item.file.read()
            data = json.loads(content)
        except json.JSONDecodeError as e:
            self.send_json({
                'success': False,
                'message': f'Invalid JSON: {str(e)}',
                'validation_errors': [f'JSON parse error: {str(e)}']
            })
            return
        
        # Validate based on file type
        if file_type == "guidelines":
            errors, warnings = validate_guidelines_json(data)
        elif file_type == "drugs":
            errors, warnings = validate_drugs_json(data)
        elif file_type == "faq":
            errors, warnings = validate_faq_json(data)
        else:
            self.send_json({'error': f'Unknown file type: {file_type}'}, 400)
            return
        
        # If only validating, return results
        if validate_only:
            activity_logger.log(
                "VALIDATE", "admin", file_type, file_item.filename,
                "VALIDATED" if not errors else "VALIDATION_FAILED"
            )
            self.send_json({
                'success': len(errors) == 0,
                'message': 'Validation complete' if not errors else 'Validation failed',
                'validation_errors': errors,
                'warnings': warnings,
                'stats': {'entries_found': len(data.get(file_type if file_type != 'guidelines' else 'guidelines', []))}
            })
            return
        
        # If errors exist, don't save
        if errors:
            activity_logger.log(
                "UPLOAD_REJECTED", "admin", file_type, file_item.filename,
                "REJECTED", f"{len(errors)} validation errors"
            )
            self.send_json({
                'success': False,
                'message': 'File not saved due to validation errors',
                'validation_errors': errors,
                'warnings': warnings
            })
            return
        
        # Save file
        import uuid
        from datetime import datetime
        
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
            "UPLOAD", "admin", file_type, file_item.filename,
            "SUCCESS", f"Saved as {save_filename}"
        )
        
        self.send_json({
            'success': True,
            'message': f'File uploaded successfully as {save_filename}',
            'file_id': file_id,
            'warnings': warnings,
            'stats': {
                'entries_found': len(data.get(file_type if file_type != 'guidelines' else 'guidelines', [])),
                'filename': save_filename
            }
        })
    
    def handle_backup(self):
        """Create a backup."""
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = DATA_DIR / "backups" / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        self.send_json({'success': True, 'backup_id': backup_dir.name})
    
    def _read_json_body(self):
        """Read and parse JSON body from request."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return None
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))
    
    def handle_create_entry(self, entry_type):
        """Create a new entry in the knowledge base."""
        data = self._read_json_body()
        if not data:
            self.send_json({'success': False, 'error': 'No data provided'}, 400)
            return
        
        # Determine file path based on type
        if entry_type == 'guideline':
            file_path = GUIDELINES_DIR / "clinical_guidelines.json"
            key = "guidelines"
        elif entry_type == 'drug':
            file_path = DRUGS_DIR / "drug_database.json"
            key = "drugs"
        elif entry_type == 'research':
            file_path = RESEARCH_DIR / "research_papers.json"
            key = "papers"
        elif entry_type == 'faq':
            file_path = FAQ_DIR / "common_questions.json"
            key = "faqs"
        else:
            self.send_json({'success': False, 'error': 'Invalid entry type'}, 400)
            return
        
        # Load existing data
        if file_path.exists():
            file_data = json.loads(file_path.read_text())
        else:
            file_data = {key: []}
        
        entries = file_data.get(key, [])
        
        # Check for duplicate ID
        entry_id = data.get('id')
        if any(e.get('id') == entry_id for e in entries):
            self.send_json({'success': False, 'error': f'Entry with ID "{entry_id}" already exists'}, 409)
            return
        
        # Add new entry
        entries.append(data)
        file_data[key] = entries
        
        # Save back to file
        file_path.write_text(json.dumps(file_data, indent=2))
        
        # Log activity
        activity_logger.log(
            "CREATE", "admin", entry_type, entry_id,
            "SUCCESS", f"Created {entry_type} entry"
        )
        
        self.send_json({'success': True, 'id': entry_id})
    
    def handle_update_entry(self, entry_type, query):
        """Update an existing entry in the knowledge base."""
        entry_id = query.get('id', [None])[0]
        if not entry_id:
            self.send_json({'success': False, 'error': 'Missing entry ID'}, 400)
            return
        
        data = self._read_json_body()
        if not data:
            self.send_json({'success': False, 'error': 'No data provided'}, 400)
            return
        
        # Determine file path based on type
        if entry_type == 'guideline':
            file_path = GUIDELINES_DIR / "clinical_guidelines.json"
            key = "guidelines"
        elif entry_type == 'drug':
            file_path = DRUGS_DIR / "drug_database.json"
            key = "drugs"
        elif entry_type == 'research':
            file_path = RESEARCH_DIR / "research_papers.json"
            key = "papers"
        elif entry_type == 'faq':
            file_path = FAQ_DIR / "common_questions.json"
            key = "faqs"
        else:
            self.send_json({'success': False, 'error': 'Invalid entry type'}, 400)
            return
        
        # Load existing data
        if not file_path.exists():
            self.send_json({'success': False, 'error': 'Data file not found'}, 404)
            return
        
        file_data = json.loads(file_path.read_text())
        entries = file_data.get(key, [])
        
        # Find and update entry
        entry_index = None
        for i, entry in enumerate(entries):
            if entry.get('id') == entry_id:
                entry_index = i
                break
        
        if entry_index is None:
            self.send_json({'success': False, 'error': 'Entry not found'}, 404)
            return
        
        # Preserve ID in case it wasn't in the update data
        data['id'] = entry_id
        entries[entry_index] = data
        file_data[key] = entries
        
        # Save back to file
        file_path.write_text(json.dumps(file_data, indent=2))
        
        # Log activity
        activity_logger.log(
            "UPDATE", "admin", entry_type, entry_id,
            "SUCCESS", f"Updated {entry_type} entry"
        )
        
        self.send_json({'success': True, 'id': entry_id})
    
    def handle_delete_entry(self, entry_type, query):
        """Delete an entry from the knowledge base."""
        entry_id = query.get('id', [None])[0]
        if not entry_id:
            self.send_json({'success': False, 'error': 'Missing entry ID'}, 400)
            return
        
        # Determine file path based on type
        if entry_type == 'guideline':
            file_path = GUIDELINES_DIR / "clinical_guidelines.json"
            key = "guidelines"
        elif entry_type == 'drug':
            file_path = DRUGS_DIR / "drug_database.json"
            key = "drugs"
        elif entry_type == 'research':
            file_path = RESEARCH_DIR / "research_papers.json"
            key = "papers"
        elif entry_type == 'faq':
            file_path = FAQ_DIR / "common_questions.json"
            key = "faqs"
        else:
            self.send_json({'success': False, 'error': 'Invalid entry type'}, 400)
            return
        
        # Load existing data
        if not file_path.exists():
            self.send_json({'success': False, 'error': 'Data file not found'}, 404)
            return
        
        file_data = json.loads(file_path.read_text())
        entries = file_data.get(key, [])
        
        # Find and remove entry
        original_count = len(entries)
        entries = [e for e in entries if e.get('id') != entry_id]
        
        if len(entries) == original_count:
            self.send_json({'success': False, 'error': 'Entry not found'}, 404)
            return
        
        file_data[key] = entries
        
        # Save back to file
        file_path.write_text(json.dumps(file_data, indent=2))
        
        # Log activity
        activity_logger.log(
            "DELETE", "admin", entry_type, entry_id,
            "SUCCESS", f"Deleted {entry_type} entry"
        )
        
        self.send_json({'success': True, 'id': entry_id})


def main():
    """Start the admin server."""
    with socketserver.TCPServer(("", PORT), AdminHandler) as httpd:
        print(f"=" * 60)
        print(f"Knowledge Base Admin Server")
        print(f"=" * 60)
        print(f"\nServer running at: http://localhost:{PORT}")
        print(f"\nFeatures:")
        print(f"  • Upload JSON files for guidelines, drugs, FAQ")
        print(f"  • Validate data before saving")
        print(f"  • Browse and search knowledge base")
        print(f"  • View activity log")
        print(f"  • Download templates")
        print(f"\nPress Ctrl+C to stop")
        print(f"=" * 60)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")


if __name__ == "__main__":
    main()
