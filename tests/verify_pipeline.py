
import sys
import os
from pathlib import Path

# Ensure backend in path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent / "backend"
sys.path.append(str(backend_dir))

# Mock env to prevent creating real files/folders in production locations
os.environ["LOGS_DIR"] = "/tmp/logs"
os.environ["MODEL_DIR"] = "/tmp/models"

def verify_pipeline_init():
    try:
        from ml.pipeline import UnifiedPipeline
        print("Initializing UnifiedPipeline...")
        pipeline = UnifiedPipeline()
        print("Pipeline initialized successfully.")
        
        # We can't easily train without data, but we can verify the class loaded and init ran
        print("✅ Pipeline integrity check passed.")
        
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_pipeline_init()
