import os
files = [
    "app/pages/1_correction_studio.py",
    "app/pages/2_export.py",
    "app/pages/3_audit_trail.py",
    "app/pages/4_ops_dashboard.py",
    "services/ops_service.py"
]

for f in files:
    with open(f, 'r') as file:
        content = file.read()
        
    old_str = """# Ensure backend root is in sys.path
backend_root = str(Path(__file__).resolve().parent.parent.parent)
if backend_root not in sys.path:
    sys.path.append(backend_root)"""

    new_str = """# Ensure backend and project root are in sys.path
backend_root = str(Path(__file__).resolve().parent.parent.parent)
project_root = str(Path(backend_root).parent)

if backend_root not in sys.path:
    sys.path.append(backend_root)
if project_root not in sys.path:
    sys.path.append(project_root)"""

    content = content.replace(old_str, new_str)
    
    # Specific fix for ops_service.py
    old_ops = """# Ensure backend root is in sys.path for health_server's backend.* imports
backend_root = str(Path(__file__).resolve().parent.parent)
if backend_root not in sys.path:
    sys.path.append(backend_root)"""

    new_ops = """# Ensure backend and project root are in sys.path
backend_root = str(Path(__file__).resolve().parent.parent)
project_root = str(Path(backend_root).parent)

if backend_root not in sys.path:
    sys.path.append(backend_root)
if project_root not in sys.path:
    sys.path.append(project_root)"""

    content = content.replace(old_ops, new_ops)
    
    with open(f, 'w') as file:
        file.write(content)
