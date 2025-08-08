"""
Configuration management for Telkom Contract Extractor
Handles loading and validation of YAML configuration files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PPStructureConfig:
    """PP-StructureV3 configuration settings"""
    text_recognition_model_name: str = "en_PP-OCRv4_mobile_rec"
    text_recognition_batch_size: int = 4
    text_rec_score_thresh: float = 0.5
    text_detection_model_name: str = "PP-OCRv5_mobile_det"
    text_det_limit_side_len: int = 960
    text_det_thresh: float = 0.3
    text_det_box_thresh: float = 0.6
    layout_detection_model_name: str = "PP-DocLayout-M"
    layout_threshold: float = 0.4
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = True
    use_table_recognition: bool = True
    enable_hpi: bool = True
    cpu_threads: int = 4
    device: str = "cpu"


@dataclass
class ExtractionConfig:
    """Data extraction configuration"""
    target_fields: list
    confidence_thresholds: Dict[str, float]


@dataclass
class PerformanceConfig:
    """Performance and resource configuration"""
    processing_timeout: int = 300
    max_memory_usage: int = 8192
    batch_size: int = 1


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    file_path: str = "logs/telkom_extractor.log"


class ConfigManager:
    """Manages configuration loading and access"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
        
        # Initialize configuration objects
        self.pp_structure = self._load_pp_structure_config()
        self.extraction = self._load_extraction_config()
        self.performance = self._load_performance_config()
        self.logging = self._load_logging_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _load_pp_structure_config(self) -> PPStructureConfig:
        """Load PP-StructureV3 configuration"""
        pp_config = self.config_data.get('pp_structure', {})
        return PPStructureConfig(**pp_config)
    
    def _load_extraction_config(self) -> ExtractionConfig:
        """Load extraction configuration"""
        extraction_config = self.config_data.get('extraction', {})
        target_fields = extraction_config.get('target_fields', [])
        
        # Load confidence thresholds
        confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        return ExtractionConfig(
            target_fields=target_fields,
            confidence_thresholds=confidence_thresholds
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration"""
        perf_config = self.config_data.get('performance', {})
        return PerformanceConfig(**perf_config)
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration"""
        log_config = self.config_data.get('logging', {})
        return LoggingConfig(**log_config)
    
    def get_field_mappings_path(self) -> Path:
        """Get path to field mappings configuration"""
        return Path("config/field_mappings.yaml")
    
    def load_field_mappings(self) -> Dict[str, Any]:
        """Load field mappings configuration"""
        mappings_path = self.get_field_mappings_path()
        
        if not mappings_path.exists():
            raise FileNotFoundError(f"Field mappings file not found: {mappings_path}")
        
        try:
            with open(mappings_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in field mappings file: {e}")
    
    def get_model_config_dict(self) -> Dict[str, Any]:
        """Get PP-StructureV3 configuration as dictionary for model initialization"""
        config = self.pp_structure
        return {
            'text_recognition_model_name': config.text_recognition_model_name,
            'text_recognition_batch_size': config.text_recognition_batch_size,
            'text_rec_score_thresh': config.text_rec_score_thresh,
            'text_detection_model_name': config.text_detection_model_name,
            'text_det_limit_side_len': config.text_det_limit_side_len,
            'text_det_thresh': config.text_det_thresh,
            'text_det_box_thresh': config.text_det_box_thresh,
            'layout_detection_model_name': config.layout_detection_model_name,
            'layout_threshold': config.layout_threshold,
            'use_doc_orientation_classify': config.use_doc_orientation_classify,
            'use_doc_unwarping': config.use_doc_unwarping,
            'use_textline_orientation': config.use_textline_orientation,
            'use_table_recognition': config.use_table_recognition,
            'enable_hpi': config.enable_hpi,
            'cpu_threads': config.cpu_threads,
            'device': config.device
        }
    
    def get_target_fields(self) -> list:
        """Get list of target extraction fields"""
        return self.extraction.target_fields
    
    def get_confidence_threshold(self, level: str = 'medium') -> float:
        """Get confidence threshold for specified level"""
        return self.extraction.confidence_thresholds.get(level, 0.6)
    
    def get_processing_timeout(self) -> int:
        """Get processing timeout in seconds"""
        return self.performance.processing_timeout
    
    def validate_config(self) -> bool:
        """Validate configuration completeness"""
        required_fields = [
            'nama_pelanggan', 'alamat', 'npwp', 'nomor_kontrak',
            'jangka_waktu', 'connectivity_telkom', 'rincian_layanan_tabel'
        ]
        
        missing_fields = [
            field for field in required_fields 
            if field not in self.extraction.target_fields
        ]
        
        if missing_fields:
            raise ValueError(f"Missing required target fields: {missing_fields}")
        
        return True


def load_config(config_path: str = "config/pipeline_config.yaml") -> ConfigManager:
    """Load configuration manager"""
    return ConfigManager(config_path)