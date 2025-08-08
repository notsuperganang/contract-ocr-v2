"""
Logging utilities for Telkom Contract Extractor
Provides structured logging with file and console output
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class TelkomLogger:
    """Custom logger for Telkom Contract Extractor"""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        self.log_level = log_level
        self.log_file = log_file
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup loguru logger with custom configuration"""
        # Remove default handler
        logger.remove()
        
        # Add console handler with colored output
        logger.add(
            sys.stderr,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # Add file handler if specified
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                self.log_file,
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB",
                retention="30 days",
                compression="zip",
                encoding="utf-8"
            )
    
    @staticmethod
    def get_logger(name: str = "telkom_extractor"):
        """Get logger instance"""
        return logger.bind(name=name)


def setup_logger(log_level: str = "INFO", log_file: str = "logs/telkom_extractor.log") -> None:
    """Setup global logger configuration"""
    TelkomLogger(log_level=log_level, log_file=log_file)


def get_logger(name: str = "telkom_extractor"):
    """Get logger instance for specific module"""
    return logger.bind(name=name)


# Performance logging decorators
def log_execution_time(func):
    """Decorator to log function execution time"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper


def log_memory_usage(func):
    """Decorator to log memory usage"""
    import psutil
    import os
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.debug(f"Memory before {func.__name__}: {memory_before:.1f} MB")
        
        try:
            result = func(*args, **kwargs)
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before
            
            logger.debug(f"Memory after {func.__name__}: {memory_after:.1f} MB (diff: {memory_diff:+.1f} MB)")
            
            return result
        except Exception as e:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            logger.error(f"Memory at error in {func.__name__}: {memory_after:.1f} MB")
            raise
    
    return wrapper


# Logging helpers for specific operations
class ExtractionLogger:
    """Specialized logger for extraction operations"""
    
    def __init__(self, document_name: str):
        self.document_name = document_name
        self.logger = get_logger("extraction")
    
    def log_start(self):
        """Log extraction start"""
        self.logger.info(f"Starting extraction for: {self.document_name}")
    
    def log_completion(self, extracted_fields: int, total_fields: int, processing_time: float):
        """Log successful extraction completion"""
        success_rate = (extracted_fields / total_fields) * 100 if total_fields > 0 else 0
        self.logger.info(
            f"Completed extraction for {self.document_name}: "
            f"{extracted_fields}/{total_fields} fields ({success_rate:.1f}%) "
            f"in {processing_time:.2f}s"
        )
    
    def log_field_extracted(self, field_name: str, value: str, confidence: float):
        """Log successful field extraction"""
        self.logger.debug(f"Extracted {field_name}: '{value}' (confidence: {confidence:.2f})")
    
    def log_field_failed(self, field_name: str, reason: str):
        """Log failed field extraction"""
        self.logger.warning(f"Failed to extract {field_name}: {reason}")
    
    def log_preprocessing(self, operation: str, result: str):
        """Log preprocessing operations"""
        self.logger.debug(f"Preprocessing {operation}: {result}")
    
    def log_postprocessing(self, operation: str, before: str, after: str):
        """Log postprocessing corrections"""
        if before != after:
            self.logger.debug(f"Text correction {operation}: '{before}' → '{after}'")
    
    def log_table_extraction(self, table_count: int, rows_extracted: int):
        """Log table extraction results"""
        self.logger.info(f"Extracted {table_count} tables with {rows_extracted} total rows")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log extraction errors"""
        error_msg = f"Extraction error for {self.document_name}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"
        self.logger.error(error_msg, exc_info=True)


# Global logger setup function
def initialize_logging(config_manager=None):
    """Initialize logging system with configuration"""
    if config_manager:
        log_config = config_manager.logging
        setup_logger(log_level=log_config.level, log_file=log_config.file_path)
    else:
        setup_logger()
    
    # Log system initialization
    logger.info("Telkom Contract Extractor logging system initialized")
    logger.info(f"Log level: {logger._core.min_level}")


# Export main logger for easy access
telkom_logger = get_logger("telkom_extractor")