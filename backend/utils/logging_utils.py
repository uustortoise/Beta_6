"""
Logging utilities with correlation ID support for request tracing.

Provides structured logging with correlation IDs that can be used to trace
requests across the system.
"""

import logging
import uuid
import contextvars
from datetime import datetime
from typing import Optional
from functools import wraps

# Context variable for correlation ID (thread-safe)
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')


def get_correlation_id() -> str:
    """Get the current correlation ID, or generate a new one if not set."""
    cid = _correlation_id.get()
    if not cid:
        cid = generate_correlation_id()
        _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(cid)


def generate_correlation_id() -> str:
    """Generate a new 8-character correlation ID."""
    return str(uuid.uuid4())[:8]


class CorrelationIdFilter(logging.Filter):
    """
    Logging filter that adds correlation_id to all log records.
    
    Usage:
        handler = logging.StreamHandler()
        handler.addFilter(CorrelationIdFilter())
        handler.setFormatter(logging.Formatter('[%(correlation_id)s] %(message)s'))
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


def setup_structured_logging(
    logger_name: str = 'elderlycare',
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up structured logging with correlation ID support.
    
    Args:
        logger_name: Name for the logger
        level: Logging level
        log_file: Optional file path for logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Format with correlation ID
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(correlation_id)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.addFilter(CorrelationIdFilter())
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.addFilter(CorrelationIdFilter())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def with_correlation_id(func):
    """
    Decorator that ensures a correlation ID exists for the duration of the function.
    
    Usage:
        @with_correlation_id
        def process_request(data):
            logger.info("Processing...")  # Will include correlation ID
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate new ID if not already set
        if not _correlation_id.get():
            _correlation_id.set(generate_correlation_id())
        return func(*args, **kwargs)
    return wrapper


def log_operation(logger: logging.Logger, operation: str):
    """
    Context manager for logging operation start/end with timing.
    
    Usage:
        with log_operation(logger, "training model"):
            train_model(data)
    """
    class OperationLogger:
        def __init__(self, logger, operation):
            self.logger = logger
            self.operation = operation
            self.start_time = None
            
        def __enter__(self):
            self.start_time = datetime.now()
            self.logger.info(f"Starting: {self.operation}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type:
                self.logger.error(f"Failed: {self.operation} ({duration:.2f}s) - {exc_val}")
            else:
                self.logger.info(f"Completed: {self.operation} ({duration:.2f}s)")
            return False
    
    return OperationLogger(logger, operation)
