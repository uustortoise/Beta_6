"""
Elder Data Manager for Multi-Elderly Care Monitoring Platform v1.11

This module provides disk persistence for elder-specific data including:
- Enhanced profiles (personal info, medical history, health metrics)
- Sleep analysis results
- ICOPE assessment results
- Prediction data
- Elder metadata
- Model files
"""

import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
import logging
from typing import Dict, Any, Optional, List
import zipfile
import io

logger = logging.getLogger(__name__)


class ElderDataManager:
    """Manages disk persistence for elder-specific data"""
    
    def __init__(self, base_dir: str = "elder_data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Initialized ElderDataManager with base directory: {base_dir}")
    
    def get_elder_dir(self, elder_id: str) -> str:
        """Get directory path for a specific elder"""
        elder_dir = os.path.join(self.base_dir, elder_id)
        os.makedirs(elder_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "profiles",           # Enhanced profiles (v1.11)
            "sleep_analysis",
            "icope_assessments", 
            "predictions",
            "models",
            "metadata",
            "backups",
            "exports",
            "training_history"    # v1.14: Training history tracking
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(elder_dir, subdir), exist_ok=True)
        
        return elder_dir
    
    
    def save_sleep_analysis(self, elder_id: str, sleep_results: Dict[str, Any], 
                           room_name: str = None, custom_timestamp: datetime = None) -> str:
        """Save sleep analysis results to disk"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            sleep_dir = os.path.join(elder_dir, "sleep_analysis")
            
            if custom_timestamp:
                timestamp = custom_timestamp.strftime("%Y%m%d_%H%M%S")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
            room_suffix = f"_{room_name}" if room_name else ""
            filename = f"sleep_analysis{room_suffix}_{timestamp}.pkl"
            filepath = os.path.join(sleep_dir, filename)
            
            # Add metadata
            sleep_results['_metadata'] = {
                'elder_id': elder_id,
                'room_name': room_name,
                'saved_at': datetime.now().isoformat(),
                'data_timestamp': custom_timestamp.isoformat() if custom_timestamp else None,
                'version': '1.1'
            }
            
            joblib.dump(sleep_results, filepath)
            logger.info(f"Saved sleep analysis for {elder_id} to {filepath}")
            
            # Also save as JSON for readability
            json_filename = f"sleep_analysis{room_suffix}_{timestamp}.json"
            json_filepath = os.path.join(sleep_dir, json_filename)
            
            # Convert to JSON-serializable format
            json_data = self._convert_to_json_serializable(sleep_results)
            with open(json_filepath, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save sleep analysis for {elder_id}: {e}")
            raise
    
    def load_sleep_analysis(self, elder_id: str, filename: str = None) -> Dict[str, Any]:
        """Load sleep analysis results from disk"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            sleep_dir = os.path.join(elder_dir, "sleep_analysis")
            
            if filename:
                filepath = os.path.join(sleep_dir, filename)
            else:
                # Load most recent file
                files = [f for f in os.listdir(sleep_dir) if f.endswith('.pkl')]
                if not files:
                    raise FileNotFoundError(f"No sleep analysis files found for {elder_id}")
                
                files.sort(reverse=True)  # Most recent first
                filepath = os.path.join(sleep_dir, files[0])
            
            results = joblib.load(filepath)
            logger.info(f"Loaded sleep analysis for {elder_id} from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load sleep analysis for {elder_id}: {e}")
            raise
    
    def save_icope_assessment(self, elder_id: str, icope_results: Dict[str, Any], custom_timestamp: datetime = None) -> str:
        """Save ICOPE assessment results to disk"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            icope_dir = os.path.join(elder_dir, "icope_assessments")
            
            if custom_timestamp:
                timestamp = custom_timestamp.strftime("%Y%m%d_%H%M%S")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"icope_assessment_{timestamp}.pkl"
            filepath = os.path.join(icope_dir, filename)
            
            # Add metadata
            icope_results['_metadata'] = {
                'elder_id': elder_id,
                'saved_at': datetime.now().isoformat(),
                'version': '1.1'
            }
            
            joblib.dump(icope_results, filepath)
            logger.info(f"Saved ICOPE assessment for {elder_id} to {filepath}")
            
            # Also save as JSON for readability
            json_filename = f"icope_assessment_{timestamp}.json"
            json_filepath = os.path.join(icope_dir, json_filename)
            
            # Convert to JSON-serializable format
            json_data = self._convert_to_json_serializable(icope_results)
            with open(json_filepath, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save ICOPE assessment for {elder_id}: {e}")
            raise
    
    def load_icope_assessment(self, elder_id: str, filename: str = None) -> Dict[str, Any]:
        """Load ICOPE assessment results from disk"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            icope_dir = os.path.join(elder_dir, "icope_assessments")
            
            if filename:
                filepath = os.path.join(icope_dir, filename)
            else:
                # Load most recent file
                files = [f for f in os.listdir(icope_dir) if f.endswith('.pkl')]
                if not files:
                    raise FileNotFoundError(f"No ICOPE assessment files found for {elder_id}")
                
                files.sort(reverse=True)  # Most recent first
                filepath = os.path.join(icope_dir, files[0])
            
            results = joblib.load(filepath)
            logger.info(f"Loaded ICOPE assessment for {elder_id} from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load ICOPE assessment for {elder_id}: {e}")
            raise
    
    def save_prediction_data(self, elder_id: str, prediction_data: Dict[str, pd.DataFrame]) -> str:
        """Save prediction data to disk"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            pred_dir = os.path.join(elder_dir, "predictions")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_data_{timestamp}.pkl"
            filepath = os.path.join(pred_dir, filename)
            
            # Save as pickle
            joblib.dump(prediction_data, filepath)
            logger.info(f"Saved prediction data for {elder_id} to {filepath}")
            
            # Also save as Excel for readability
            excel_filename = f"prediction_data_{timestamp}.xlsx"
            excel_filepath = os.path.join(pred_dir, excel_filename)
            
            with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
                for room_name, df in prediction_data.items():
                    # Limit sheet name to 31 characters (Excel limitation)
                    sheet_name = room_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save prediction data for {elder_id}: {e}")
            raise
    
    def save_elder_metadata(self, elder_id: str, metadata: Dict[str, Any]) -> str:
        """Save legacy elder metadata to disk (backward compatibility)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            meta_dir = os.path.join(elder_dir, "metadata")
            
            filename = "profile.json"
            filepath = os.path.join(meta_dir, filename)
            
            # Add timestamp
            metadata['last_updated'] = datetime.now().isoformat()
            metadata['elder_id'] = elder_id
            
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved legacy metadata for {elder_id} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save metadata for {elder_id}: {e}")
            raise
    
    def load_elder_metadata(self, elder_id: str) -> Dict[str, Any]:
        """Load legacy elder metadata from disk (backward compatibility)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            meta_dir = os.path.join(elder_dir, "metadata")
            
            filepath = os.path.join(meta_dir, "profile.json")
            
            if not os.path.exists(filepath):
                return {
                    'elder_id': elder_id,
                    'name': f"Elderly {elder_id}",
                    'age': None,
                    'gender': None,
                    'notes': "",
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
            
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded legacy metadata for {elder_id} from {filepath}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load metadata for {elder_id}: {e}")
            raise
    
    def save_enhanced_profile(self, elder_id: str, profile: Dict[str, Any]) -> str:
        """Save enhanced profile to disk (v1.11 format)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            profile_dir = os.path.join(elder_dir, "profiles")
            
            # Use a standard filename for the main profile
            filename = "enhanced_profile.json"
            filepath = os.path.join(profile_dir, filename)
            
            # Create backup of existing profile if it exists
            if os.path.exists(filepath):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filepath = os.path.join(profile_dir, f"enhanced_profile_backup_{timestamp}.json")
                shutil.copy2(filepath, backup_filepath)
                logger.info(f"Created backup of profile at {backup_filepath}")
            
            # Update system metadata
            if 'system_metadata' not in profile:
                profile['system_metadata'] = {}
            
            profile['system_metadata']['last_updated'] = datetime.now().isoformat()
            profile['system_metadata']['version'] = "1.11.0"
            
            with open(filepath, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            
            logger.info(f"Saved enhanced profile for {elder_id} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save enhanced profile for {elder_id}: {e}")
            raise
    
    def load_enhanced_profile(self, elder_id: str) -> Dict[str, Any]:
        """Load enhanced profile from disk (v1.11 format)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            profile_dir = os.path.join(elder_dir, "profiles")
            
            filepath = os.path.join(profile_dir, "enhanced_profile.json")
            
            if not os.path.exists(filepath):
                # Return empty template if no profile exists
                from elderlycare_v1_11.profile.templates import DEFAULT_PROFILE_TEMPLATE
                profile = DEFAULT_PROFILE_TEMPLATE.copy()
                profile['system_metadata']['created_at'] = datetime.now().isoformat()
                profile['system_metadata']['last_updated'] = datetime.now().isoformat()
                profile['system_metadata']['version'] = "1.11.0"
                return profile
            
            with open(filepath, 'r') as f:
                profile = json.load(f)
            
            logger.info(f"Loaded enhanced profile for {elder_id} from {filepath}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to load enhanced profile for {elder_id}: {e}")
            raise
    
    def migrate_legacy_to_enhanced(self, elder_id: str) -> Dict[str, Any]:
        """Migrate legacy metadata to enhanced profile format"""
        try:
            # Load legacy metadata
            legacy_metadata = self.load_elder_metadata(elder_id)
            
            # Load existing enhanced profile or create new one
            enhanced_profile = self.load_enhanced_profile(elder_id)
            
            # Map legacy fields to enhanced profile
            if 'name' in legacy_metadata and legacy_metadata['name']:
                enhanced_profile['personal_info']['full_name'] = legacy_metadata['name']
                enhanced_profile['personal_info']['preferred_name'] = legacy_metadata['name']
            
            if 'age' in legacy_metadata and legacy_metadata['age']:
                enhanced_profile['personal_info']['age'] = legacy_metadata['age']
            
            if 'gender' in legacy_metadata and legacy_metadata['gender']:
                enhanced_profile['personal_info']['gender'] = legacy_metadata['gender']
            
            if 'notes' in legacy_metadata and legacy_metadata['notes']:
                # Try to extract medical info from notes
                notes = legacy_metadata['notes']
                enhanced_profile['personal_info']['notes'] = notes
            
            # Update timestamps
            enhanced_profile['system_metadata']['migrated_at'] = datetime.now().isoformat()
            enhanced_profile['system_metadata']['legacy_source'] = "v1.1_metadata"
            
            # Save the migrated profile
            self.save_enhanced_profile(elder_id, enhanced_profile)
            
            logger.info(f"Successfully migrated legacy data for {elder_id} to enhanced profile")
            return enhanced_profile
            
        except Exception as e:
            logger.error(f"Failed to migrate legacy data for {elder_id}: {e}")
            raise
    
    def create_backup(self, elder_id: str) -> str:
        """Create a backup of all elder data"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            backup_dir = os.path.join(elder_dir, "backups")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"backup_{timestamp}.zip"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Create zip archive
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(elder_dir):
                    # Skip backups directory to avoid recursion
                    if "backups" in root:
                        continue
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(elder_dir))
                        zipf.write(file_path, arcname)
            
            logger.info(f"Created backup for {elder_id} at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup for {elder_id}: {e}")
            raise
    
    def export_elder_data(self, elder_id: str, format: str = "zip") -> bytes:
        """Export all elder data as a downloadable file"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            export_dir = os.path.join(elder_dir, "exports")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == "zip":
                # Create in-memory zip file
                memory_file = io.BytesIO()
                with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(elder_dir):
                        # Skip exports directory to avoid recursion
                        if "exports" in root:
                            continue
                        
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.path.dirname(elder_dir))
                            zipf.write(file_path, arcname)
                
                memory_file.seek(0)
                return memory_file.getvalue()
                
            elif format == "json":
                # Export as JSON summary
                export_data = {
                    'elder_id': elder_id,
                    'exported_at': datetime.now().isoformat(),
                    'metadata': self.load_elder_metadata(elder_id),
                    'sleep_analyses': self._list_files(os.path.join(elder_dir, "sleep_analysis")),
                    'icope_assessments': self._list_files(os.path.join(elder_dir, "icope_assessments")),
                    'predictions': self._list_files(os.path.join(elder_dir, "predictions"))
                }
                
                return json.dumps(export_data, indent=2, default=str).encode('utf-8')
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export data for {elder_id}: {e}")
            raise
    
    def list_sleep_analyses(self, elder_id: str) -> List[Dict[str, str]]:
        """List all sleep analysis files for an elder"""
        return self._list_files_in_category(elder_id, "sleep_analysis")
    
    def list_icope_assessments(self, elder_id: str) -> List[Dict[str, str]]:
        """List all ICOPE assessment files for an elder"""
        return self._list_files_in_category(elder_id, "icope_assessments")
    
    def list_predictions(self, elder_id: str) -> List[Dict[str, str]]:
        """List all prediction data files for an elder"""
        return self._list_files_in_category(elder_id, "predictions")
    
    def list_backups(self, elder_id: str) -> List[Dict[str, str]]:
        """List all backup files for an elder"""
        return self._list_files_in_category(elder_id, "backups")
    
    def list_profiles(self, elder_id: str) -> List[Dict[str, str]]:
        """List all profile files for an elder (including backups)"""
        return self._list_files_in_category(elder_id, "profiles")
    
    def save_training_history(self, elder_id: str, training_history: List[Dict[str, Any]]) -> str:
        """Save training history to disk (v1.14 feature)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            
            # Create training_history subdirectory
            history_dir = os.path.join(elder_dir, "training_history")
            os.makedirs(history_dir, exist_ok=True)
            
            # Use a standard filename for the main history
            filename = "training_history.json"
            filepath = os.path.join(history_dir, filename)
            
            # Create backup of existing history if it exists
            if os.path.exists(filepath):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filepath = os.path.join(history_dir, f"training_history_backup_{timestamp}.json")
                shutil.copy2(filepath, backup_filepath)
                logger.info(f"Created backup of training history at {backup_filepath}")
            
            # Prepare data with metadata
            history_data = {
                'elder_id': elder_id,
                'training_history': training_history,
                'last_updated': datetime.now().isoformat(),
                'version': '1.13.0',
                'total_trainings': len(training_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            logger.info(f"Saved training history for {elder_id} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save training history for {elder_id}: {e}")
            raise
    
    def load_training_history(self, elder_id: str) -> Dict[str, Any]:
        """Load training history from disk (v1.14 feature)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            history_dir = os.path.join(elder_dir, "training_history")
            
            filepath = os.path.join(history_dir, "training_history.json")
            
            if not os.path.exists(filepath):
                # Return empty structure if no history exists
                return {
                    'elder_id': elder_id,
                    'training_history': [],
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.13.0',
                    'total_trainings': 0
                }
            
            with open(filepath, 'r') as f:
                history_data = json.load(f)
            
            logger.info(f"Loaded training history for {elder_id} from {filepath}")
            return history_data
            
        except Exception as e:
            logger.error(f"Failed to load training history for {elder_id}: {e}")
            raise
    
    def save_prediction_history_log(self, elder_id: str, history_log: List[Dict[str, Any]]) -> str:
        """Save prediction history log to disk (v1.14 feature)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            history_dir = os.path.join(elder_dir, "prediction_history")
            os.makedirs(history_dir, exist_ok=True)
            
            filename = "prediction_history.json"
            filepath = os.path.join(history_dir, filename)
            
            # Create backup if exists
            if os.path.exists(filepath):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copy2(filepath, os.path.join(history_dir, f"prediction_history_backup_{timestamp}.json"))
            
            log_data = {
                'elder_id': elder_id,
                'history': history_log,
                'last_updated': datetime.now().isoformat(),
                'version': '1.14.0'
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
                
            return filepath
        except Exception as e:
            logger.error(f"Failed to save prediction history log for {elder_id}: {e}")
            raise

    def load_prediction_history_log(self, elder_id: str) -> List[Dict[str, Any]]:
        """Load prediction history log from disk (v1.14 feature)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            history_dir = os.path.join(elder_dir, "prediction_history")
            filepath = os.path.join(history_dir, "prediction_history.json")
            
            if not os.path.exists(filepath):
                return []
                
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('history', [])
        except Exception as e:
            logger.error(f"Failed to load prediction history log for {elder_id}: {e}")
            return []
    
    def save_adl_history(self, elder_id: str, adl_history: List[Dict[str, Any]]) -> str:
        """Save ADL history to disk (v1.15 feature)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            history_dir = os.path.join(elder_dir, "training_history") # Reusing training_history dir for now
            os.makedirs(history_dir, exist_ok=True)
            
            filename = "adl_history.json"
            filepath = os.path.join(history_dir, filename)
            
            # Create backup
            if os.path.exists(filepath):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copy2(filepath, os.path.join(history_dir, f"adl_history_backup_{timestamp}.json"))
            
            history_data = {
                'elder_id': elder_id,
                'adl_history': adl_history,
                'last_updated': datetime.now().isoformat(),
                'version': '1.15.0',
                'record_count': len(adl_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
            logger.info(f"Saved ADL history for {elder_id} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save ADL history for {elder_id}: {e}")
            raise

    def load_adl_history(self, elder_id: str) -> List[Dict[str, Any]]:
        """Load ADL history from disk (v1.15 feature)"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            history_dir = os.path.join(elder_dir, "training_history")
            filepath = os.path.join(history_dir, "adl_history.json")
            
            if not os.path.exists(filepath):
                return []
                
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get('adl_history', [])
        except Exception as e:
            logger.error(f"Failed to load ADL history for {elder_id}: {e}")
            return []
    
    def list_training_history(self, elder_id: str) -> List[Dict[str, str]]:
        """List training history files for an elder"""
        return self._list_files_in_category(elder_id, "training_history")
    
    def _list_files_in_category(self, elder_id: str, category: str) -> List[Dict[str, str]]:
        """List files in a specific category"""
        try:
            elder_dir = self.get_elder_dir(elder_id)
            category_dir = os.path.join(elder_dir, category)
            
            if not os.path.exists(category_dir):
                return []
            
            files = []
            for filename in os.listdir(category_dir):
                if filename.startswith('.'):
                    continue
                
                filepath = os.path.join(category_dir, filename)
                stat = os.stat(filepath)
                
                files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'size_bytes': stat.st_size,
                    'size_human': self._human_readable_size(stat.st_size),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'category': category
                })
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x['modified'], reverse=True)
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files in {category} for {elder_id}: {e}")
            return []
    
    def _list_files(self, directory: str) -> List[str]:
        """List files in a directory"""
        if not os.path.exists(directory):
            return []
        
        return [f for f in os.listdir(directory) if not f.startswith('.')]
    
    def _human_readable_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _convert_to_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_json_serializable(item) for item in data]
        elif isinstance(data, (pd.Timestamp, datetime)):
            return data.isoformat()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, '__dict__'):
            return self._convert_to_json_serializable(data.__dict__)
        else:
            return data


# Global instance for easy access
data_manager = ElderDataManager()


def save_current_elder_data(elder_id: str, elder_data: Dict[str, Any]) -> Dict[str, str]:
    """Save all current elder data to disk"""
    saved_files = {}
    
    try:
        # Save enhanced profile if available
        if 'enhanced_profile' in elder_data and elder_data['enhanced_profile']:
            saved_files['enhanced_profile'] = data_manager.save_enhanced_profile(
                elder_id, elder_data['enhanced_profile']
            )
        # Fallback to legacy metadata if no enhanced profile
        elif 'metadata' in elder_data:
            saved_files['metadata'] = data_manager.save_elder_metadata(elder_id, elder_data['metadata'])
        
        # Save prediction data
        if 'pred_data' in elder_data and elder_data['pred_data']:
            saved_files['prediction_data'] = data_manager.save_prediction_data(elder_id, elder_data['pred_data'])
        
        # Save sleep analysis results (if available in session state)
        import streamlit as st
        sleep_results = st.session_state.get('sleep__analysis_results', {})
        if sleep_results:
            for room_name, results in sleep_results.items():
                saved_files[f'sleep_analysis_{room_name}'] = data_manager.save_sleep_analysis(
                    elder_id, results, room_name
                )
        
        # Save ICOPE assessment results (if available)
        if 'icope_processor' in elder_data and elder_data['icope_processor']:
            try:
                # Try to get assessment results from ICOPE processor
                icope_processor = elder_data['icope_processor']
                if hasattr(icope_processor, 'get_current_assessment'):
                    icope_results = icope_processor.get_current_assessment()
                elif hasattr(icope_processor, 'current_assessment'):
                    icope_results = icope_processor.current_assessment
                else:
                    icope_results = None
                
                if icope_results:
                    saved_files['icope_assessment'] = data_manager.save_icope_assessment(
                        elder_id, icope_results
                    )
            except Exception as e:
                logger.warning(f"Could not save ICOPE assessment for {elder_id}: {e}")
        
        # Create a backup
        saved_files['backup'] = data_manager.create_backup(elder_id)
        
        logger.info(f"Saved all data for {elder_id}: {list(saved_files.keys())}")
        return saved_files
        
    except Exception as e:
        logger.error(f"Failed to save elder data for {elder_id}: {e}")
        raise


def load_elder_data(elder_id: str) -> Dict[str, Any]:
    """Load all elder data from disk"""
    loaded_data = {}
    
    try:
        # Try to load enhanced profile first
        try:
            loaded_data['enhanced_profile'] = data_manager.load_enhanced_profile(elder_id)
            logger.info(f"Loaded enhanced profile for {elder_id}")
        except Exception as e:
            logger.warning(f"Could not load enhanced profile for {elder_id}: {e}")
            # Fallback to legacy metadata
            loaded_data['metadata'] = data_manager.load_elder_metadata(elder_id)
            logger.info(f"Loaded legacy metadata for {elder_id}")
        
        # Note: Prediction data, sleep analysis, and ICOPE assessments
        # are loaded on-demand when needed
        
        return loaded_data
        
    except Exception as e:
        logger.error(f"Failed to load elder data for {elder_id}: {e}")
        raise


def get_elder_data_summary(elder_id: str) -> Dict[str, Any]:
    """Get summary of all data available for an elder"""
    summary = {
        'elder_id': elder_id,
        'timestamp': datetime.now().isoformat(),
        'data_available': {}
    }
    
    try:
        # Check enhanced profiles
        profile_files = data_manager.list_profiles(elder_id)
        summary['data_available']['profiles'] = {
            'count': len(profile_files),
            'latest': profile_files[0] if profile_files else None
        }
        
        # Check sleep analyses
        sleep_files = data_manager.list_sleep_analyses(elder_id)
        summary['data_available']['sleep_analyses'] = {
            'count': len(sleep_files),
            'latest': sleep_files[0] if sleep_files else None
        }
        
        # Check ICOPE assessments
        icope_files = data_manager.list_icope_assessments(elder_id)
        summary['data_available']['icope_assessments'] = {
            'count': len(icope_files),
            'latest': icope_files[0] if icope_files else None
        }
        
        # Check predictions
        pred_files = data_manager.list_predictions(elder_id)
        summary['data_available']['predictions'] = {
            'count': len(pred_files),
            'latest': pred_files[0] if pred_files else None
        }
        
        # Check backups
        backup_files = data_manager.list_backups(elder_id)
        summary['data_available']['backups'] = {
            'count': len(backup_files),
            'latest': backup_files[0] if backup_files else None
        }
        
        # Check models
        elder_dir = data_manager.get_elder_dir(elder_id)
        models_dir = os.path.join(elder_dir, "models")
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
            summary['data_available']['models'] = {
                'count': len(model_files),
                'files': model_files
            }
        
        # Check training history (v1.14)
        history_files = data_manager.list_training_history(elder_id)
        summary['data_available']['training_history'] = {
            'count': len(history_files),
            'latest': history_files[0] if history_files else None
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get data summary for {elder_id}: {e}")
        summary['error'] = str(e)
        return summary
