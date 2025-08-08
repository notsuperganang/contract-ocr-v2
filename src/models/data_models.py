"""
Data models for Telkom Contract Extractor
Defines the structure for extracted contract data
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import json


class ExtractionStatus(Enum):
    """Status of field extraction"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_FOUND = "not_found"


class ConfidenceLevel(Enum):
    """Confidence levels for extracted data"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExtractedField:
    """Represents a single extracted field with metadata"""
    name: str
    value: str
    confidence: float
    status: ExtractionStatus
    source_region: Optional[str] = None
    coordinates: Optional[Dict[str, int]] = None
    extraction_method: Optional[str] = None
    validation_passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'name': self.name,
            'value': self.value,
            'confidence': self.confidence,
            'status': self.status.value,
            'source_region': self.source_region,
            'coordinates': self.coordinates,
            'extraction_method': self.extraction_method,
            'validation_passed': self.validation_passed
        }


@dataclass
class Perwakilan:
    """Representative information"""
    nama: Optional[str] = None
    jabatan: Optional[str] = None
    
    def is_complete(self) -> bool:
        return bool(self.nama and self.jabatan)


@dataclass
class JangkaWaktu:
    """Contract duration information"""
    mulai: Optional[str] = None
    akhir: Optional[str] = None
    
    def is_complete(self) -> bool:
        return bool(self.mulai and self.akhir)
    
    def get_duration_days(self) -> Optional[int]:
        """Calculate duration in days if both dates are present"""
        if not self.is_complete():
            return None
        
        try:
            from datetime import datetime
            start = datetime.strptime(self.mulai, "%Y-%m-%d")
            end = datetime.strptime(self.akhir, "%Y-%m-%d")
            return (end - start).days
        except ValueError:
            return None


@dataclass
class LayananUtama:
    """Main service information"""
    connectivity_telkom: Optional[int] = None
    non_connectivity_telkom: Optional[int] = None
    bundling: Optional[int] = None
    
    def get_total_services(self) -> int:
        """Get total number of services"""
        total = 0
        if self.connectivity_telkom:
            total += self.connectivity_telkom
        if self.non_connectivity_telkom:
            total += self.non_connectivity_telkom
        if self.bundling:
            total += self.bundling
        return total


@dataclass
class ServiceRow:
    """Single row in service details table"""
    no: Optional[str] = None
    layanan: Optional[str] = None
    jumlah: Optional[str] = None
    biaya: Optional[str] = None
    lokasi: Optional[str] = None
    alamat_instalasi: Optional[str] = None
    bulanan: Optional[str] = None
    tahunan: Optional[str] = None
    keterangan: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'NO': self.no,
            'LAYANAN': self.layanan,
            'JUMLAH': self.jumlah,
            'BIAYA': self.biaya,
            'LOKASI': self.lokasi,
            'ALAMAT_INSTALASI': self.alamat_instalasi,
            'BULANAN': self.bulanan,
            'TAHUNAN': self.tahunan,
            'KETERANGAN': self.keterangan
        }


@dataclass
class InformasiPelanggan:
    """Customer information section"""
    nama_pelanggan: Optional[str] = None
    alamat: Optional[str] = None
    npwp: Optional[str] = None
    perwakilan: Optional[Perwakilan] = field(default_factory=Perwakilan)
    kontak_person: Optional[str] = None
    
    def get_completeness_score(self) -> float:
        """Calculate completeness score (0-1)"""
        fields = [
            self.nama_pelanggan,
            self.alamat,
            self.npwp,
            self.kontak_person
        ]
        filled_fields = sum(1 for f in fields if f)
        
        # Add perwakilan score
        if self.perwakilan and self.perwakilan.is_complete():
            filled_fields += 1
        
        return filled_fields / 5  # Total 5 fields including perwakilan


@dataclass
class InformasiKontrak:
    """Contract information section"""
    nomor_kontrak: Optional[str] = None
    jangka_waktu: Optional[JangkaWaktu] = field(default_factory=JangkaWaktu)
    
    def is_complete(self) -> bool:
        return bool(self.nomor_kontrak and self.jangka_waktu.is_complete())


@dataclass
class TelkomContractData:
    """Complete Telkom contract data structure"""
    # Document metadata
    document_name: str
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    
    # Main data sections
    informasi_pelanggan: InformasiPelanggan = field(default_factory=InformasiPelanggan)
    informasi_kontrak: InformasiKontrak = field(default_factory=InformasiKontrak)
    layanan_utama: LayananUtama = field(default_factory=LayananUtama)
    
    # Service details table
    rincian_layanan_tabel: List[ServiceRow] = field(default_factory=list)
    
    # Additional fields
    tata_cara_pembayaran: Optional[str] = None
    kontak_person_telkom: Optional[str] = None
    
    # Extraction metadata
    extracted_fields: Dict[str, ExtractedField] = field(default_factory=dict)
    overall_confidence: Optional[float] = None
    extraction_status: ExtractionStatus = ExtractionStatus.NOT_FOUND
    
    def add_extracted_field(self, field: ExtractedField):
        """Add extracted field to metadata"""
        self.extracted_fields[field.name] = field
        
        # Update the actual data field
        self._update_data_field(field)
    
    def _update_data_field(self, field: ExtractedField):
        """Update the actual data field based on extracted field"""
        field_name = field.name
        value = field.value
        
        if field_name == "nama_pelanggan":
            self.informasi_pelanggan.nama_pelanggan = value
        elif field_name == "alamat":
            self.informasi_pelanggan.alamat = value
        elif field_name == "npwp":
            self.informasi_pelanggan.npwp = value
        elif field_name == "kontak_person":
            self.informasi_pelanggan.kontak_person = value
        elif field_name == "nomor_kontrak":
            self.informasi_kontrak.nomor_kontrak = value
        elif field_name == "tata_cara_pembayaran":
            self.tata_cara_pembayaran = value
        elif field_name == "kontak_person_telkom":
            self.kontak_person_telkom = value
        elif field_name == "connectivity_telkom":
            try:
                self.layanan_utama.connectivity_telkom = int(value)
            except (ValueError, TypeError):
                pass
        elif field_name == "non_connectivity_telkom":
            try:
                self.layanan_utama.non_connectivity_telkom = int(value)
            except (ValueError, TypeError):
                pass
        elif field_name == "bundling":
            try:
                self.layanan_utama.bundling = int(value)
            except (ValueError, TypeError):
                pass
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall extraction confidence"""
        if not self.extracted_fields:
            return 0.0
        
        total_confidence = sum(field.confidence for field in self.extracted_fields.values())
        self.overall_confidence = total_confidence / len(self.extracted_fields)
        return self.overall_confidence
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of extraction results"""
        total_fields = len(self.extracted_fields)
        successful_extractions = sum(
            1 for field in self.extracted_fields.values() 
            if field.status == ExtractionStatus.SUCCESS
        )
        
        return {
            'document_name': self.document_name,
            'extraction_timestamp': self.extraction_timestamp.isoformat(),
            'processing_time': self.processing_time,
            'total_fields': total_fields,
            'successful_extractions': successful_extractions,
            'success_rate': (successful_extractions / total_fields * 100) if total_fields > 0 else 0,
            'overall_confidence': self.overall_confidence,
            'extraction_status': self.extraction_status.value,
            'customer_completeness': self.informasi_pelanggan.get_completeness_score(),
            'contract_complete': self.informasi_kontrak.is_complete(),
            'service_rows_count': len(self.rincian_layanan_tabel),
            'total_services': self.layanan_utama.get_total_services()
        }
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            'document_name': self.document_name,
            'extraction_timestamp': self.extraction_timestamp.isoformat(),
            'processing_time': self.processing_time,
            'informasi_pelanggan': {
                'nama_pelanggan': self.informasi_pelanggan.nama_pelanggan,
                'alamat': self.informasi_pelanggan.alamat,
                'npwp': self.informasi_pelanggan.npwp,
                'perwakilan': {
                    'nama': self.informasi_pelanggan.perwakilan.nama,
                    'jabatan': self.informasi_pelanggan.perwakilan.jabatan
                },
                'kontak_person': self.informasi_pelanggan.kontak_person
            },
            'informasi_kontrak': {
                'nomor_kontrak': self.informasi_kontrak.nomor_kontrak,
                'jangka_waktu': {
                    'mulai': self.informasi_kontrak.jangka_waktu.mulai,
                    'akhir': self.informasi_kontrak.jangka_waktu.akhir
                }
            },
            'layanan_utama': {
                'connectivity_telkom': self.layanan_utama.connectivity_telkom,
                'non_connectivity_telkom': self.layanan_utama.non_connectivity_telkom,
                'bundling': self.layanan_utama.bundling
            },
            'rincian_layanan_tabel': [row.to_dict() for row in self.rincian_layanan_tabel],
            'tata_cara_pembayaran': self.tata_cara_pembayaran,
            'kontak_person_telkom': self.kontak_person_telkom,
            'extracted_fields': {name: field.to_dict() for name, field in self.extracted_fields.items()},
            'overall_confidence': self.overall_confidence,
            'extraction_status': self.extraction_status.value
        }
    
    def save_to_json(self, file_path: str):
        """Save contract data to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2)


@dataclass
class ExtractionResult:
    """Result of the extraction process"""
    contract_data: TelkomContractData
    success: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def add_warning(self, message: str):
        """Add warning message"""
        self.warnings.append(message)
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0