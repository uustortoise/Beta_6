"""
Health check module for the ElderlyCarePlatform backend.

Provides health check endpoints that can be used by monitoring systems,
load balancers, and container orchestrators (Docker, Kubernetes).
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    timestamp: str
    version: str
    components: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "components": self.components
        }


class HealthChecker:
    """
    Health checker that validates system components.
    
    Usage:
        checker = HealthChecker(db_path="/path/to/db", models_dir="/path/to/models")
        health = checker.check_all()
        print(health.to_dict())
    """
    
    VERSION = "5.5.1"  # Beta 5.5 version
    
    def __init__(
        self,
        db_path: Path = None,
        models_dir: Path = None,
        check_postgresql: bool = True
    ):
        self.db_path = db_path
        self.models_dir = models_dir
        self.check_postgresql = check_postgresql
    
    def check_database(self) -> ComponentHealth:
        """Check database connectivity via adapter."""
        try:
            from backend.db.legacy_adapter import LegacyDatabaseAdapter
            import time
            
            start = time.time()
            adapter = LegacyDatabaseAdapter()
            with adapter.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            latency = (time.time() - start) * 1000
            
            return ComponentHealth("database", HealthStatus.HEALTHY, "Connected (PostgreSQL)", latency)
        except Exception as e:
            return ComponentHealth("database", HealthStatus.UNHEALTHY, str(e))

    def check_sqlite(self) -> ComponentHealth:
        """Deprecated: Redirects to check_database."""
        return self.check_database()
    
    def check_postgresql(self) -> ComponentHealth:
        """Check PostgreSQL connectivity (Alias for check_database)."""
        return self.check_database()
    
    def check_models(self) -> ComponentHealth:
        """Check ML models availability."""
        if not self.models_dir:
            return ComponentHealth("ml_models", HealthStatus.DEGRADED, "Models dir not configured")
        
        try:
            models_path = Path(self.models_dir)
            if not models_path.exists():
                return ComponentHealth("ml_models", HealthStatus.DEGRADED, "Models directory missing")
            
            # Count model files
            model_files = list(models_path.rglob("*.keras"))
            if not model_files:
                return ComponentHealth("ml_models", HealthStatus.DEGRADED, "No models found")
            
            return ComponentHealth(
                "ml_models", 
                HealthStatus.HEALTHY, 
                f"{len(model_files)} model(s) available"
            )
        except Exception as e:
            return ComponentHealth("ml_models", HealthStatus.UNHEALTHY, str(e))
    
    def check_disk_space(self) -> ComponentHealth:
        """Check available disk space."""
        try:
            import shutil
            
            # Check root or data directory
            check_path = self.db_path.parent if self.db_path else Path.cwd()
            total, used, free = shutil.disk_usage(check_path)
            
            free_gb = free / (1024 ** 3)
            free_pct = (free / total) * 100
            
            if free_pct < 5:
                return ComponentHealth("disk", HealthStatus.UNHEALTHY, f"{free_gb:.1f}GB free ({free_pct:.1f}%)")
            elif free_pct < 15:
                return ComponentHealth("disk", HealthStatus.DEGRADED, f"{free_gb:.1f}GB free ({free_pct:.1f}%)")
            else:
                return ComponentHealth("disk", HealthStatus.HEALTHY, f"{free_gb:.1f}GB free")
        except Exception as e:
            return ComponentHealth("disk", HealthStatus.DEGRADED, str(e))
    
    def check_liveness(self) -> Dict[str, Any]:
        """
        Simple liveness check (is the process running?).
        
        Returns:
            {"status": "ok"} if alive
        """
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
    
    def check_readiness(self) -> Dict[str, Any]:
        """
        Readiness check (is the system ready to accept requests?).
        
        Checks critical components: database and models.
        
        Returns:
            {"ready": True/False, "components": {...}}
        """
        db_health = self.check_database()
        models_health = self.check_models()
        
        # Ready if DB is healthy (or at least reachable) and models are okay
        ready = (
            db_health.status != HealthStatus.UNHEALTHY and
            models_health.status != HealthStatus.UNHEALTHY
        )
        
        return {
            "ready": ready,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": {"status": db_health.status.value, "message": db_health.message},
                "models": {"status": models_health.status.value, "message": models_health.message}
            }
        }
    
    def check_all(self) -> SystemHealth:
        """
        Deep health check of all system components.
        
        Returns:
            SystemHealth object with full status
        """
        components = [
            self.check_database(),
            self.check_models(),
            self.check_disk_space()
        ]
        
        # Overall status is worst of all components
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return SystemHealth(
            status=overall,
            timestamp=datetime.now().isoformat(),
            version=self.VERSION,
            components=[asdict(c) for c in components]
        )


# Convenience function for quick health checks
def get_health_status(db_path: Path = None, models_dir: Path = None) -> Dict[str, Any]:
    """
    Get current system health status.
    
    Args:
        db_path: Path to SQLite database
        models_dir: Path to models directory
        
    Returns:
        Health status dictionary
    """
    checker = HealthChecker(db_path=db_path, models_dir=models_dir)
    return checker.check_all().to_dict()
