"""
Main extraction pipeline using PP-StructureV3 for Telkom contracts
Integrates all components for end-to-end contract data extraction
"""

import time
import re  # ← FIX: Add missing import
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

try:
    from paddleocr import PPStructureV3
except ImportError:
    print("PaddleOCR not found. Please install it with: pip install paddleocr")
    raise

from ..models.data_models import (
    TelkomContractData, ExtractedField, ExtractionStatus, 
    ExtractionResult, ServiceRow
)
from .preprocessor import DocumentPreprocessor
from .postprocessor import IndonesianTextCorrector, DataCleaner
from ..utils.config import ConfigManager
from ..utils.logger import get_logger, ExtractionLogger, log_execution_time, log_memory_usage

logger = get_logger("extractor")


class TelkomContractExtractor:
    """Main extractor class for Telkom contract data extraction"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize the extractor with configuration
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing Telkom Contract Extractor")
        
        # Load configuration
        self.config = ConfigManager(config_path)
        self.field_mappings = self.config.load_field_mappings()
        
        # Initialize components
        self.preprocessor = DocumentPreprocessor()
        self.text_corrector = IndonesianTextCorrector()
        self.data_cleaner = DataCleaner()
        
        # Initialize PP-StructureV3
        self._initialize_pp_structure()
        
        logger.info("Telkom Contract Extractor initialized successfully")
    
    def _initialize_pp_structure(self):
        """Initialize PP-StructureV3 pipeline"""
        try:
            logger.info("Initializing PP-StructureV3 pipeline")
            
            # Get model configuration
            model_config = self.config.get_model_config_dict()
            
            # Initialize PP-StructureV3
            self.pp_structure = PPStructureV3(**model_config)
            
            logger.info("PP-StructureV3 pipeline initialized")
            logger.debug(f"Model config: {model_config}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PP-StructureV3: {e}")
            raise
    
    @log_execution_time
    @log_memory_usage
    def extract_from_file(self, file_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract data from a contract file
        
        Args:
            file_path: Path to contract file (PDF or image)
            
        Returns:
            ExtractionResult containing the extracted data
        """
        file_path = Path(file_path)
        document_name = file_path.stem
        
        # Initialize extraction logger
        extraction_logger = ExtractionLogger(document_name)
        extraction_logger.log_start()
        
        start_time = time.time()
        
        try:
            # Create result object
            contract_data = TelkomContractData(document_name=document_name)
            
            # Step 1: Preprocess document
            logger.info(f"Step 1: Preprocessing document: {file_path.name}")
            images = self.preprocessor.process_document(file_path)
            
            if not images:
                raise ValueError("No images extracted from document")
            
            extraction_logger.log_preprocessing("document_conversion", f"{len(images)} images")
            
            # Step 2: Process each page with PP-StructureV3
            logger.info(f"Step 2: Processing {len(images)} pages with PP-StructureV3")
            all_results = []
            
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)}")
                
                # Run PP-StructureV3
                page_results = self.pp_structure.predict(image)
                all_results.extend(page_results)
                
                logger.debug(f"Page {i+1} processed: {len(page_results)} results")
            
            # Step 3: Extract structured data
            logger.info("Step 3: Extracting structured data from OCR results")
            self._extract_structured_data(all_results, contract_data, extraction_logger)
            
            # Step 4: Apply postprocessing corrections
            logger.info("Step 4: Applying text corrections and validation")
            self._apply_postprocessing(contract_data, extraction_logger)
            
            # Step 5: Calculate metrics
            processing_time = time.time() - start_time
            contract_data.processing_time = processing_time
            contract_data.calculate_overall_confidence()
            
            # Determine overall extraction status
            successful_fields = sum(
                1 for field in contract_data.extracted_fields.values()
                if field.status == ExtractionStatus.SUCCESS
            )
            total_fields = len(self.config.get_target_fields())
            
            if successful_fields == 0:
                contract_data.extraction_status = ExtractionStatus.FAILED
            elif successful_fields < total_fields * 0.5:
                contract_data.extraction_status = ExtractionStatus.PARTIAL
            else:
                contract_data.extraction_status = ExtractionStatus.SUCCESS
            
            # Log completion
            extraction_logger.log_completion(
                successful_fields, total_fields, processing_time
            )
            
            return ExtractionResult(
                contract_data=contract_data,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            extraction_logger.log_error(e, "main_extraction")
            
            logger.error(f"Extraction failed for {file_path.name}: {e}")
            
            return ExtractionResult(
                contract_data=TelkomContractData(document_name=document_name),
                success=False,
                error_message=str(e)
            )
    
    def _extract_structured_data(self, pp_results: List, contract_data: TelkomContractData, 
                               extraction_logger: ExtractionLogger):
        """Extract structured data from PP-StructureV3 results"""
        
        # Combine all text from all results
        all_text_blocks = []
        all_table_data = []
        
        for result in pp_results:
            # Extract text blocks
            if hasattr(result, 'overall_ocr_res') and result.overall_ocr_res:
                ocr_res = result.overall_ocr_res
                if 'rec_texts' in ocr_res:
                    texts = ocr_res['rec_texts']
                    for text in texts:
                        if text and text.strip():
                            all_text_blocks.append(text.strip())
            
            # Extract table data
            if hasattr(result, 'table_res_list') and result.table_res_list:
                for table_res in result.table_res_list:
                    if 'pred_html' in table_res:
                        html_table = table_res['pred_html']
                        # Parse HTML table to structured data
                        table_rows = self._parse_html_table(html_table)
                        all_table_data.extend(table_rows)
        
        logger.info(f"Extracted {len(all_text_blocks)} text blocks and {len(all_table_data)} table rows")
        
        # Extract individual fields
        self._extract_fields_from_text(all_text_blocks, contract_data, extraction_logger)
        
        # Extract table data
        if all_table_data:
            self._extract_table_data(all_table_data, contract_data, extraction_logger)
    
    def _extract_fields_from_text(self, text_blocks: List[str], contract_data: TelkomContractData,
                                extraction_logger: ExtractionLogger):
        """Extract individual fields from text blocks"""
        
        # Combine all text for pattern matching
        combined_text = ' '.join(text_blocks)
        
        target_fields = self.config.get_target_fields()
        field_mappings = self.field_mappings.get('field_mappings', {})
        
        for field_name in target_fields:
            if field_name in field_mappings:
                field_config = field_mappings[field_name]
                
                # Try to extract field value
                extracted_value = self._extract_single_field(
                    combined_text, field_name, field_config
                )
                
                if extracted_value:
                    # Calculate confidence (simplified)
                    confidence = self._calculate_field_confidence(
                        extracted_value, field_name, field_config
                    )
                    
                    # Create extracted field
                    extracted_field = ExtractedField(
                        name=field_name,
                        value=extracted_value,
                        confidence=confidence,
                        status=ExtractionStatus.SUCCESS,
                        extraction_method="text_pattern_matching"
                    )
                    
                    contract_data.add_extracted_field(extracted_field)
                    extraction_logger.log_field_extracted(field_name, extracted_value, confidence)
                    
                else:
                    # Field not found
                    extracted_field = ExtractedField(
                        name=field_name,
                        value="",
                        confidence=0.0,
                        status=ExtractionStatus.NOT_FOUND,
                        extraction_method="text_pattern_matching"
                    )
                    
                    contract_data.add_extracted_field(extracted_field)
                    extraction_logger.log_field_failed(field_name, "Not found in text")
    
    def _extract_single_field(self, text: str, field_name: str, field_config: Dict) -> Optional[str]:
        """Extract a single field value from text"""
        
        keywords = field_config.get('keywords', [])
        
        # Look for field value near keywords
        for keyword in keywords:
            # Pattern to find text after keyword
            patterns = [
                rf'{re.escape(keyword)}\s*:?\s*([^\n\r]+)',
                rf'{re.escape(keyword)}\s+([^\n\r]+)',
                rf'([^\n\r]*{re.escape(keyword)}[^\n\r]*)'
            ]
            
            for pattern in patterns:
                # ← FIX: Remove local import re
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    
                    # Clean and validate candidate
                    candidate = self._clean_field_candidate(candidate, field_name)
                    
                    if self._validate_field_candidate(candidate, field_name, field_config):
                        return candidate
        
        # Try structured data extraction
        structured_value = self.text_corrector.extract_structured_data(text, field_name)
        if structured_value:
            return structured_value
        
        return None
    
    def _clean_field_candidate(self, candidate: str, field_name: str) -> str:
        """Clean field candidate value"""
        # ← FIX: Remove local import re
        
        # Remove common OCR artifacts
        candidate = re.sub(r'[|\\]', '', candidate)
        candidate = re.sub(r'\s+', ' ', candidate).strip()
        
        # Remove trailing punctuation that's likely OCR noise
        candidate = re.sub(r'[.,:;]+$', '', candidate)
        
        # Field-specific cleaning
        if field_name == "nomor_kontrak":
            # Keep only alphanumeric, dots, slashes, hyphens
            candidate = re.sub(r'[^A-Za-z0-9./-]', '', candidate)
        elif field_name == "npwp":
            # Keep only digits and basic formatting
            candidate = re.sub(r'[^\d.-]', '', candidate)
        
        return candidate
    
    def _validate_field_candidate(self, candidate: str, field_name: str, field_config: Dict) -> bool:
        """Validate field candidate"""
        if not candidate or len(candidate.strip()) < 2:
            return False
        
        # Check validation pattern if provided
        validation_pattern = field_config.get('validation_pattern')
        if validation_pattern:
            # ← FIX: Remove local import re
            if not re.match(validation_pattern, candidate):
                return False
        
        # Field-specific validation
        if field_name == "npwp":
            # NPWP should have 15 digits
            digits = re.sub(r'\D', '', candidate)
            return len(digits) == 15
        elif field_name == "nomor_kontrak":
            # Contract number should be reasonable length
            return 5 <= len(candidate) <= 50
        
        return True
    
    def _calculate_field_confidence(self, value: str, field_name: str, field_config: Dict) -> float:
        """Calculate confidence score for extracted field"""
        base_confidence = 0.5
        
        # Boost confidence if value matches validation pattern
        validation_pattern = field_config.get('validation_pattern')
        if validation_pattern:
            # ← FIX: Remove local import re
            if re.match(validation_pattern, value):
                base_confidence += 0.3
        
        # Field-specific confidence boosts
        if field_name == "nomor_kontrak" and "K.TEL" in value.upper():
            base_confidence += 0.2
        elif field_name == "nama_pelanggan" and any(prefix in value.upper() for prefix in ["PT", "CV", "TBK"]):
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _extract_table_data(self, table_data: List[Dict], contract_data: TelkomContractData,
                          extraction_logger: ExtractionLogger):
        """Extract service table data"""
        
        # Clean table data
        cleaned_table_data = self.data_cleaner.clean_table_data(table_data)
        
        # Convert to ServiceRow objects
        service_rows = []
        for row_data in cleaned_table_data:
            service_row = ServiceRow(
                no=row_data.get('NO'),
                layanan=row_data.get('LAYANAN'),
                jumlah=row_data.get('JUMLAH'),
                biaya=row_data.get('BIAYA'),
                lokasi=row_data.get('LOKASI'),
                alamat_instalasi=row_data.get('ALAMAT_INSTALASI'),
                bulanan=row_data.get('BULANAN'),
                tahunan=row_data.get('TAHUNAN'),
                keterangan=row_data.get('KETERANGAN')
            )
            service_rows.append(service_row)
        
        contract_data.rincian_layanan_tabel = service_rows
        
        # Log table extraction
        extraction_logger.log_table_extraction(1, len(service_rows))
        
        # Create extracted field for table
        if service_rows:
            extracted_field = ExtractedField(
                name="rincian_layanan_tabel",
                value=f"{len(service_rows)} rows extracted",
                confidence=0.8,
                status=ExtractionStatus.SUCCESS,
                extraction_method="table_recognition"
            )
            contract_data.add_extracted_field(extracted_field)
    
    def _parse_html_table(self, html_table: str) -> List[Dict]:
        """Parse HTML table to structured data"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')
            
            if not table:
                return []
            
            rows = table.find_all('tr')
            if not rows:
                return []
            
            # Get headers from first row
            header_row = rows[0]
            headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            # Extract data rows
            data_rows = []
            for row in rows[1:]:
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                
                if cells:
                    # Create row dict
                    row_dict = {}
                    for i, cell in enumerate(cells):
                        header = headers[i] if i < len(headers) else f"col_{i}"
                        row_dict[header] = cell
                    
                    data_rows.append(row_dict)
            
            return data_rows
            
        except Exception as e:
            logger.warning(f"Failed to parse HTML table: {e}")
            return []
    
    def _apply_postprocessing(self, contract_data: TelkomContractData, 
                            extraction_logger: ExtractionLogger):
        """Apply postprocessing corrections"""
        
        for field_name, extracted_field in contract_data.extracted_fields.items():
            if extracted_field.value and extracted_field.status == ExtractionStatus.SUCCESS:
                
                # Apply text correction
                corrected_value = self.text_corrector.correct_text(
                    extracted_field.value, field_name
                )
                
                # Validate corrected value
                is_valid, error_msg = self.text_corrector.validate_extracted_data(
                    corrected_value, field_name
                )
                
                # Update field
                if corrected_value != extracted_field.value:
                    extraction_logger.log_postprocessing(
                        f"text_correction_{field_name}",
                        extracted_field.value,
                        corrected_value
                    )
                    extracted_field.value = corrected_value
                    
                    # Update contract data
                    contract_data._update_data_field(extracted_field)
                
                extracted_field.validation_passed = is_valid
                
                if not is_valid:
                    logger.warning(f"Validation failed for {field_name}: {error_msg}")
    
    def batch_extract(self, input_dir: Union[str, Path], 
                     output_dir: Union[str, Path]) -> List[ExtractionResult]:
        """
        Extract data from multiple files in batch
        
        Args:
            input_dir: Directory containing contract files
            output_dir: Directory to save results
            
        Returns:
            List of extraction results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF and image files
        file_patterns = ['*.pdf', '*.png', '*.jpg', '*.jpeg', '*.tiff']
        files = []
        for pattern in file_patterns:
            files.extend(input_dir.glob(pattern))
        
        logger.info(f"Found {len(files)} files for batch processing")
        
        results = []
        for i, file_path in enumerate(files):
            logger.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")
            
            try:
                result = self.extract_from_file(file_path)
                results.append(result)
                
                # Save individual result
                if result.success:
                    json_path = output_dir / f"{file_path.stem}_extracted.json"
                    result.contract_data.save_to_json(str(json_path))
                    logger.info(f"Saved extraction result: {json_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                results.append(ExtractionResult(
                    contract_data=TelkomContractData(document_name=file_path.stem),
                    success=False,
                    error_message=str(e)
                ))
        
        # Save batch summary
        summary = self._create_batch_summary(results)
        summary_path = output_dir / "batch_summary.json"
        
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Batch processing completed. Summary saved: {summary_path}")
        
        return results
    
    def _create_batch_summary(self, results: List[ExtractionResult]) -> Dict:
        """Create summary of batch processing results"""
        total_files = len(results)
        successful_extractions = sum(1 for r in results if r.success)
        
        summary = {
            'batch_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': total_files,
            'successful_extractions': successful_extractions,
            'success_rate': (successful_extractions / total_files * 100) if total_files > 0 else 0,
            'files': []
        }
        
        for result in results:
            file_summary = {
                'document_name': result.contract_data.document_name,
                'success': result.success,
                'error_message': result.error_message
            }
            
            if result.success:
                file_summary.update(result.contract_data.get_extraction_summary())
            
            summary['files'].append(file_summary)
        
        return summary
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'preprocessor'):
            self.preprocessor.cleanup()
        
        logger.info("Extractor cleanup completed")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()