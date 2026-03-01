import sys
import os
from pathlib import Path

# Add PROJECT_ROOT and BACKEND_DIR to path
# PROJECT_ROOT (for 'import backend...')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# BACKEND_DIR (for 'import elderlycare_v1_16...')
sys.path.append(str(Path(__file__).resolve().parent.parent))

from elderlycare_v1_16.database import db

def init():
    schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    print(f"Loading schema from {schema_path}")
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
        
    db.init_schema(schema_sql)
    print("Database initialized.")

if __name__ == "__main__":
    init()
