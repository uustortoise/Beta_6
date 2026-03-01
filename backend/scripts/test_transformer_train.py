import sys
import os
import logging

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.pipeline import UnifiedPipeline
from elderlycare_v1_16.config.settings import RAW_DATA_DIR

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training():
    logger.info("Initializing Pipeline (Default: Transformer)...")
    pipeline = UnifiedPipeline()
    
    file_name = "HK001_jessica_training.parquet"
    file_path = RAW_DATA_DIR / file_name
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
        
    logger.info(f"Starting training on {file_path}...")
    try:
        results, metrics = pipeline.train_and_predict(file_path, "HK001")
        
        logger.info("\n=== Training Results ===")
        for m in metrics:
            logger.info(f"Room: {m['room']} | Accuracy: {m['accuracy']:.2%} | Epochs: {m['epochs']}")
            
        logger.info("\n=== Prediction Sample ===")
        for room, df in results.items():
            logger.info(f"Room {room}: {len(df)} predictions generated")
            
        print("✅ Transformer Training Test Passed")
        
    except Exception as e:
        logger.error(f"Training Failed: {e}", exc_info=True)
        print("❌ Transformer Training Test Failed")

if __name__ == "__main__":
    test_training()
