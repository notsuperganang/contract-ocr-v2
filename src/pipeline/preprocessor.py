"""
Document preprocessor for Telkom Contract Extractor
Handles PDF to image conversion and image preprocessing
"""

import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from typing import List, Tuple, Optional, Union
import tempfile

from ..utils.logger import get_logger, log_execution_time, log_memory_usage

logger = get_logger("preprocessor")


class DocumentPreprocessor:
    """Preprocesses documents for optimal OCR extraction"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="telkom_extractor_")
        logger.debug(f"Initialized preprocessor with temp dir: {self.temp_dir}")
    
    @log_execution_time
    def process_document(self, file_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Process document (PDF or image) and return list of image arrays
        
        Args:
            file_path: Path to PDF or image file
            
        Returns:
            List of image arrays (numpy arrays)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        logger.info(f"Processing document: {file_path.name}")
        
        # Determine file type and process accordingly
        if file_path.suffix.lower() == '.pdf':
            return self._process_pdf(file_path)
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    @log_memory_usage
    def _process_pdf(self, pdf_path: Path) -> List[np.ndarray]:
        """
        Convert PDF to images and preprocess them
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of preprocessed image arrays
        """
        logger.info(f"Converting PDF to images: {pdf_path.name}")
        
        try:
            # Open PDF document
            pdf_document = fitz.open(str(pdf_path))
            images = []
            
            logger.info(f"PDF has {len(pdf_document)} pages")
            
            for page_num in range(len(pdf_document)):
                logger.debug(f"Processing page {page_num + 1}")
                
                # Get page
                page = pdf_document[page_num]
                
                # Convert page to image with high DPI for better OCR
                # Using matrix for scaling (2.0 = 144 DPI, good for OCR)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                
                # Fix: Use BytesIO instead of fitz.open
                import io
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Preprocess the image
                processed_image = self._preprocess_image(pil_image)
                
                # Convert to numpy array
                image_array = np.array(processed_image)
                images.append(image_array)
                
                logger.debug(f"Page {page_num + 1} processed: {image_array.shape}")
            
            pdf_document.close()
            logger.info(f"Successfully converted PDF to {len(images)} images")
            
            return images
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _process_image(self, image_path: Path) -> List[np.ndarray]:
        """
        Process single image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            List with single preprocessed image array
        """
        logger.info(f"Processing image: {image_path.name}")
        
        try:
            # Open image
            pil_image = Image.open(image_path)
            
            # Preprocess the image
            processed_image = self._preprocess_image(pil_image)
            
            # Convert to numpy array
            image_array = np.array(processed_image)
            
            logger.info(f"Image processed: {image_array.shape}")
            
            return [image_array]
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing operations to improve OCR accuracy
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        logger.debug(f"Preprocessing image: {image.size}, mode: {image.mode}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.debug("Converted image to RGB")
        
        # Resize if image is too large (max 4000px on longest side)
        max_size = 4000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to: {new_size}")
        
        # Enhance contrast slightly for better OCR
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        logger.debug("Applied contrast enhancement")
        
        # Enhance sharpness slightly
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        logger.debug("Applied sharpness enhancement")
        
        # Apply slight denoising
        image = image.filter(ImageFilter.MedianFilter(size=3))
        logger.debug("Applied median filter for noise reduction")
        
        return image
    
    def save_debug_images(self, images: List[np.ndarray], document_name: str) -> List[str]:
        """
        Save preprocessed images for debugging purposes
        
        Args:
            images: List of image arrays
            document_name: Name of original document
            
        Returns:
            List of saved image paths
        """
        saved_paths = []
        
        for i, image_array in enumerate(images):
            # Convert numpy array back to PIL Image
            pil_image = Image.fromarray(image_array)
            
            # Create filename
            filename = f"{document_name}_page_{i+1:03d}.png"
            filepath = Path(self.temp_dir) / filename
            
            # Save image
            pil_image.save(filepath, 'PNG')
            saved_paths.append(str(filepath))
            
            logger.debug(f"Saved debug image: {filepath}")
        
        return saved_paths
    
    def get_image_info(self, image_array: np.ndarray) -> dict:
        """
        Get information about processed image
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Dictionary with image information
        """
        return {
            'shape': image_array.shape,
            'dtype': str(image_array.dtype),
            'min_value': int(image_array.min()),
            'max_value': int(image_array.max()),
            'mean_value': float(image_array.mean()),
            'size_mb': image_array.nbytes / (1024 * 1024)
        }
    
    def validate_image_quality(self, image_array: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate image quality for OCR processing
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check image dimensions
        height, width = image_array.shape[:2]
        
        if width < 300 or height < 300:
            issues.append(f"Image too small: {width}x{height} (minimum 300x300)")
        
        if width > 8000 or height > 8000:
            issues.append(f"Image very large: {width}x{height} (may cause memory issues)")
        
        # Check image quality metrics
        if len(image_array.shape) == 3:
            # Color image - convert to grayscale for analysis
            gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image_array
        
        # Check contrast
        contrast = gray.std()
        if contrast < 10:
            issues.append(f"Low contrast detected: {contrast:.1f}")
        
        # Check if image is too dark or too bright
        mean_brightness = gray.mean()
        if mean_brightness < 30:
            issues.append(f"Image too dark: mean brightness {mean_brightness:.1f}")
        elif mean_brightness > 225:
            issues.append(f"Image too bright: mean brightness {mean_brightness:.1f}")
        
        is_valid = len(issues) == 0
        
        if issues:
            logger.warning(f"Image quality issues detected: {issues}")
        
        return is_valid, issues
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Utility functions
def get_pdf_info(pdf_path: Union[str, Path]) -> dict:
    """
    Get information about PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with PDF information
    """
    pdf_path = Path(pdf_path)
    
    try:
        doc = fitz.open(str(pdf_path))
        info = {
            'filename': pdf_path.name,
            'page_count': len(doc),
            'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
            'metadata': doc.metadata,
            'is_encrypted': doc.needs_pass,
            'page_sizes': []
        }
        
        # Get page sizes
        for page_num in range(len(doc)):
            page = doc[page_num]
            rect = page.rect
            info['page_sizes'].append({
                'page': page_num + 1,
                'width': rect.width,
                'height': rect.height
            })
        
        doc.close()
        return info
        
    except Exception as e:
        logger.error(f"Error getting PDF info: {e}")
        return {'error': str(e)}


def estimate_processing_time(file_path: Union[str, Path]) -> float:
    """
    Estimate processing time based on file characteristics
    
    Args:
        file_path: Path to file
        
    Returns:
        Estimated processing time in seconds
    """
    file_path = Path(file_path)
    
    # Base time estimates (seconds)
    base_time_per_page = 30  # Base time for PP-StructureV3 processing
    preprocessing_time_per_page = 5  # Preprocessing time
    
    if file_path.suffix.lower() == '.pdf':
        pdf_info = get_pdf_info(file_path)
        if 'page_count' in pdf_info:
            pages = pdf_info['page_count']
            estimated_time = pages * (base_time_per_page + preprocessing_time_per_page)
            return estimated_time
    
    # For single images
    return base_time_per_page + preprocessing_time_per_page