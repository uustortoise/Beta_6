"""
Centralised sys.path setup for all Streamlit pages.
Import this module at the top of every page file:

    import app._path_setup  # noqa: F401
"""
import sys
from pathlib import Path

_backend_root = str(Path(__file__).resolve().parent.parent)  # backend/
_project_root = str(Path(_backend_root).parent)              # Beta_6/

if _backend_root not in sys.path:
    sys.path.append(_backend_root)
if _project_root not in sys.path:
    sys.path.append(_project_root)
