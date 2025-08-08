"""
Text postprocessor for Telkom Contract Extractor
Handles Indonesian text corrections and data cleaning
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger("postprocessor")


class IndonesianTextCorrector:
    """Corrects common OCR errors in Indonesian text for Telkom contracts"""
    
    def __init__(self):
        # Common OCR corrections for Indonesian text
        self.common_corrections = {
            # Telkom specific
            "tcikom": "telkom",
            "teikom": "telkom", 
            "tclkom": "telkom",
            "tcIkom": "telkom",
            "teleom": "telkom",
            "teIkom": "telkom",
            
            # Company types
            "pt.": "pt ",
            "cv.": "cv ",
            "ltd.": "ltd ",
            "tbk.": "tbk ",
            "pt ": "pt ",
            
            # Common Indonesian words
            "pcrusahaan": "perusahaan",
            "perusahan": "perusahaan",
            "prusahaan": "perusahaan",
            "alamet": "alamat",
            "alamt": "alamat",
            "kontak": "kontak",
            "kontrak": "kontrak",
            "kontrak": "kontrak",
            "nomor": "nomor",
            "nornor": "nomor",
            "tangaal": "tanggal",
            "tanggaI": "tanggal",
            "jangka": "jangka",
            "waktu": "waktu",
            "waktuu": "waktu",
            
            # Service related
            "astinet": "astinet",
            "indihome": "indihome",
            "layanan": "layanan",
            "layanar": "layanan",
            "connectivity": "connectivity",
            "bundling": "bundling",
            "bundIing": "bundling",
            
            # Numbers and currency
            "rupiah": "rupiah",
            "ribu": "ribu",
            "juta": "juta",
            "milyar": "milyar",
            
            # Common OCR character errors
            "0": "o",  # in words, not numbers
            "1": "l",  # in words
            "5": "s",  # in some contexts
        }
        
        # Patterns for specific types of data
        self.patterns = {
            'currency': [
                r'rp\.?\s*([0-9.,]+)',
                r'idr\s*([0-9.,]+)',
                r'([0-9.,]+)\s*rupiah'
            ],
            'phone': [
                r'(\+?62\s*\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4})',
                r'(0\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4})',
                r'(\d{4}[\s-]?\d{4}[\s-]?\d{4})'
            ],
            'email': [
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ],
            'contract_number': [
                r'(K\.TEL\.[A-Z0-9./\-]+)',
                r'([A-Z0-9./\-]{10,})'
            ],
            'npwp': [
                r'(\d{2}\.?\d{3}\.?\d{3}\.?\d{1}[\-.]?\d{3}\.?\d{3})',
                r'(\d{15})'
            ],
            'date': [
                r'(\d{1,2}[\s/-]\d{1,2}[\s/-]\d{4})',
                r'(\d{4}[\s/-]\d{1,2}[\s/-]\d{1,2})',
                r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
            ]
        }
    
    def correct_text(self, text: str, field_type: Optional[str] = None) -> str:
        """
        Apply text corrections based on field type
        
        Args:
            text: Raw text from OCR
            field_type: Type of field (optional for targeted corrections)
            
        Returns:
            Corrected text
        """
        if not text or not isinstance(text, str):
            return text
        
        original_text = text
        
        # Basic cleaning
        text = self._basic_cleaning(text)
        
        # Apply common corrections
        text = self._apply_common_corrections(text)
        
        # Apply field-specific corrections
        if field_type:
            text = self._apply_field_specific_corrections(text, field_type)
        
        # Log corrections
        if text != original_text:
            logger.debug(f"Text corrected ({field_type}): '{original_text}' → '{text}'")
        
        return text
    
    def _basic_cleaning(self, text: str) -> str:
        """Apply basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix common spacing issues
        text = re.sub(r'\s*([,.:])\s*', r'\1 ', text)
        text = re.sub(r'\s+([,.:])', r'\1', text)
        
        # Remove multiple dots/commas
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        
        return text
    
    def _apply_common_corrections(self, text: str) -> str:
        """Apply common OCR corrections"""
        text_lower = text.lower()
        
        for incorrect, correct in self.common_corrections.items():
            # Case-insensitive replacement while preserving original case pattern
            pattern = re.escape(incorrect)
            matches = re.finditer(pattern, text_lower)
            
            for match in reversed(list(matches)):
                start, end = match.span()
                original_case = text[start:end]
                
                # Preserve case pattern
                if original_case.isupper():
                    replacement = correct.upper()
                elif original_case.istitle():
                    replacement = correct.title()
                else:
                    replacement = correct
                
                text = text[:start] + replacement + text[end:]
        
        return text
    
    def _apply_field_specific_corrections(self, text: str, field_type: str) -> str:
        """Apply corrections specific to field type"""
        
        if field_type == "nama_pelanggan":
            return self._correct_company_name(text)
        elif field_type == "alamat":
            return self._correct_address(text)
        elif field_type == "npwp":
            return self._correct_npwp(text)
        elif field_type == "nomor_kontrak":
            return self._correct_contract_number(text)
        elif field_type in ["kontak_person", "kontak_person_telkom"]:
            return self._correct_contact_info(text)
        elif "tanggal" in field_type or "waktu" in field_type:
            return self._correct_date(text)
        elif "biaya" in field_type or "harga" in field_type:
            return self._correct_currency(text)
        
        return text
    
    def _correct_company_name(self, text: str) -> str:
        """Correct company name"""
        # Fix PT/CV formatting
        text = re.sub(r'\bpt\b\.?\s*', 'PT ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcv\b\.?\s*', 'CV ', text, flags=re.IGNORECASE)
        text = re.sub(r'\btbk\b\.?\s*', 'TBK ', text, flags=re.IGNORECASE)
        
        # Title case for company names
        words = text.split()
        corrected_words = []
        
        for word in words:
            if word.upper() in ['PT', 'CV', 'TBK', 'LTD']:
                corrected_words.append(word.upper())
            else:
                corrected_words.append(word.title())
        
        return ' '.join(corrected_words)
    
    def _correct_address(self, text: str) -> str:
        """Correct address text"""
        # Fix common address terms
        address_corrections = {
            'jalan': 'Jalan',
            'jl.': 'Jl.',
            'jl ': 'Jl. ',
            'no.': 'No.',
            'no ': 'No. ',
            'rt.': 'RT.',
            'rw.': 'RW.',
            'kel.': 'Kel.',
            'kec.': 'Kec.',
            'kota': 'Kota',
            'kabupaten': 'Kabupaten',
            'provinsi': 'Provinsi'
        }
        
        text_lower = text.lower()
        for incorrect, correct in address_corrections.items():
            text = re.sub(r'\b' + re.escape(incorrect) + r'\b', correct, text, flags=re.IGNORECASE)
        
        return text
    
    def _correct_npwp(self, text: str) -> str:
        """Correct NPWP format"""
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', text)
        
        if len(digits) == 15:
            # Format as XX.XXX.XXX.X-XXX.XXX
            formatted = f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}.{digits[8]}-{digits[9:12]}.{digits[12:15]}"
            return formatted
        
        return text
    
    def _correct_contract_number(self, text: str) -> str:
        """Correct contract number format"""
        # Keep alphanumeric, dots, slashes, and hyphens
        text = re.sub(r'[^A-Za-z0-9./-]', '', text)
        
        # Fix common contract number patterns
        if 'KTEL' in text.upper():
            text = text.upper()
            text = re.sub(r'K\.?TEL', 'K.TEL', text)
        
        return text
    
    def _correct_contact_info(self, text: str) -> str:
        """Correct contact information"""
        # Check if it looks like a phone number
        if re.search(r'\d{4,}', text):
            # Clean phone number
            text = re.sub(r'[^\d+\-\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _correct_date(self, text: str) -> str:
        """Correct date format"""
        # Try to standardize date format to YYYY-MM-DD
        
        # Pattern: DD/MM/YYYY or DD-MM-YYYY
        match = re.search(r'(\d{1,2})[\s/-](\d{1,2})[\s/-](\d{4})', text)
        if match:
            day, month, year = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # Pattern: YYYY/MM/DD or YYYY-MM-DD
        match = re.search(r'(\d{4})[\s/-](\d{1,2})[\s/-](\d{1,2})', text)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        return text
    
    def _correct_currency(self, text: str) -> str:
        """Correct currency format"""
        # Extract currency amount
        for pattern in self.patterns['currency']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1)
                # Clean and format amount
                amount = re.sub(r'[^\d.,]', '', amount)
                return f"Rp. {amount}"
        
        return text
    
    def extract_structured_data(self, text: str, field_type: str) -> Optional[str]:
        """
        Extract structured data using patterns
        
        Args:
            text: Text to extract from
            field_type: Type of data to extract
            
        Returns:
            Extracted data or None
        """
        if field_type not in self.patterns:
            return None
        
        for pattern in self.patterns[field_type]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1)
                logger.debug(f"Extracted {field_type}: '{extracted}' from '{text}'")
                return extracted
        
        return None
    
    def validate_extracted_data(self, data: str, field_type: str) -> Tuple[bool, str]:
        """
        Validate extracted data
        
        Args:
            data: Extracted data
            field_type: Type of field
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not data or not isinstance(data, str):
            return False, "Empty or invalid data"
        
        if field_type == "npwp":
            # Check NPWP format
            clean_npwp = re.sub(r'\D', '', data)
            if len(clean_npwp) != 15:
                return False, f"NPWP must be 15 digits, got {len(clean_npwp)}"
        
        elif field_type == "email":
            # Basic email validation
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, data):
                return False, "Invalid email format"
        
        elif field_type in ["tanggal_mulai", "tanggal_akhir"]:
            # Date validation
            date_pattern = r'^\d{4}-\d{2}-\d{2}$'
            if not re.match(date_pattern, data):
                return False, "Date must be in YYYY-MM-DD format"
        
        return True, ""


class DataCleaner:
    """Additional data cleaning utilities"""
    
    @staticmethod
    def clean_table_data(table_data: List[Dict]) -> List[Dict]:
        """Clean table data"""
        cleaned_data = []
        corrector = IndonesianTextCorrector()
        
        for row in table_data:
            cleaned_row = {}
            for key, value in row.items():
                if value and isinstance(value, str):
                    # Apply text correction
                    cleaned_value = corrector.correct_text(value)
                    
                    # Additional cleaning for specific columns
                    if 'biaya' in key.lower() or 'harga' in key.lower():
                        cleaned_value = corrector._correct_currency(cleaned_value)
                    elif 'jumlah' in key.lower():
                        # Extract numeric value
                        numeric = re.search(r'(\d+)', cleaned_value)
                        if numeric:
                            cleaned_value = numeric.group(1)
                    
                    cleaned_row[key] = cleaned_value
                else:
                    cleaned_row[key] = value
            
            cleaned_data.append(cleaned_row)
        
        return cleaned_data
    
    @staticmethod
    def standardize_field_names(data: Dict) -> Dict:
        """Standardize field names"""
        field_name_mapping = {
            'nama': 'nama_pelanggan',
            'perusahaan': 'nama_pelanggan',
            'customer': 'nama_pelanggan',
            'alamat_perusahaan': 'alamat',
            'address': 'alamat',
            'contact_person': 'kontak_person',
            'pic': 'kontak_person',
            'contract_number': 'nomor_kontrak',
            'no_kontrak': 'nomor_kontrak',
            'start_date': 'tanggal_mulai',
            'end_date': 'tanggal_akhir',
            'payment_method': 'tata_cara_pembayaran'
        }
        
        standardized = {}
        for key, value in data.items():
            standardized_key = field_name_mapping.get(key.lower(), key)
            standardized[standardized_key] = value
        
        return standardized
    
    @staticmethod
    def remove_duplicates(data_list: List[Dict], key_field: str = 'nama_pelanggan') -> List[Dict]:
        """Remove duplicate entries based on key field"""
        seen = set()
        unique_data = []
        
        for item in data_list:
            key_value = item.get(key_field, '')
            if key_value not in seen:
                seen.add(key_value)
                unique_data.append(item)
        
        return unique_data