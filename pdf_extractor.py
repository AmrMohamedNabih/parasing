"""
Hybrid PDF Content Extractor
Combines direct text extraction with OCR fallback and image extraction
"""

import os
import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse
import io


class HybridPDFExtractor:
    """Extract text and images from PDF files using hybrid approach"""
    
    def __init__(self, lang: str = 'ara+eng', text_threshold: float = 0.1):
        """
        Initialize Hybrid PDF Extractor
        
        Args:
            lang: Language(s) for OCR. Default is 'ara+eng' for Arabic and English
            text_threshold: Minimum text ratio to consider page as digital (0.0-1.0)
        """
        self.lang = lang
        self.text_threshold = text_threshold
        
    def validate_page_text(self, text: str, page_area: float) -> Tuple[bool, float]:
        """
        Validate if extracted text is sufficient
        
        Args:
            text: Extracted text
            page_area: Page area in pixels
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        if not text or not text.strip():
            return False, 0.0
        
        # Calculate text density
        word_count = len(text.split())
        char_count = len(text.strip())
        
        # Heuristic: good extraction should have reasonable word count
        if word_count < 5:  # Very few words, likely poor extraction
            return False, 0.2
        
        # Check for gibberish (too many single characters)
        single_chars = sum(1 for word in text.split() if len(word) == 1)
        if word_count > 0 and (single_chars / word_count) > 0.5:
            return False, 0.3
        
        # Calculate confidence based on word count and character count
        confidence = min(1.0, (word_count / 50) * 0.5 + (char_count / 500) * 0.5)
        
        return confidence > self.text_threshold, confidence
    
    def extract_text_direct(self, pdf_path: str, page_num: int) -> Tuple[str, float]:
        """
        Extract text directly from PDF using PyMuPDF and pdfplumber
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        text = ""
        
        try:
            # Try PyMuPDF first (faster)
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            text = page.get_text()
            page_area = page.rect.width * page.rect.height
            doc.close()
            
            # Validate extraction
            is_valid, confidence = self.validate_page_text(text, page_area)
            
            if is_valid:
                return text, confidence
            
            # If PyMuPDF fails, try pdfplumber (better for tables)
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num]
                text = page.extract_text() or ""
                page_area = page.width * page.height
            
            is_valid, confidence = self.validate_page_text(text, page_area)
            return text, confidence
            
        except Exception as e:
            print(f"Direct extraction error on page {page_num + 1}: {str(e)}")
            return "", 0.0
    
    def preprocess_image(self, image: Image.Image, method: str = 'adaptive') -> Image.Image:
        """
        Preprocess image to improve OCR accuracy
        
        Args:
            image: PIL Image object
            method: Preprocessing method ('adaptive', 'threshold', 'denoise', 'all')
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        if method == 'adaptive' or method == 'all':
            # Adaptive thresholding
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'threshold':
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'denoise':
            processed = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        else:
            processed = gray
            
        if method == 'all':
            # Apply denoising
            processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
            
            # Apply dilation and erosion
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.dilate(processed, kernel, iterations=1)
            processed = cv2.erode(processed, kernel, iterations=1)
        
        # Convert back to PIL Image
        return Image.fromarray(processed)
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR"""
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        return image
    
    def extract_text_ocr(self, pdf_path: str, page_num: int, 
                        preprocess: bool = True, dpi: int = 300) -> str:
        """
        Extract text using OCR
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            preprocess: Whether to apply preprocessing
            dpi: Resolution for conversion
            
        Returns:
            Extracted text
        """
        try:
            # Convert specific page to image
            images = convert_from_path(
                pdf_path, 
                dpi=dpi, 
                first_page=page_num + 1, 
                last_page=page_num + 1
            )
            
            if not images:
                return ""
            
            image = images[0]
            
            if preprocess:
                image = self.preprocess_image(image, method='all')
                image = self.enhance_image(image)
            
            # Configure pytesseract
            custom_config = r'--oem 3 --psm 6'
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=self.lang, config=custom_config)
            
            return text
            
        except Exception as e:
            print(f"OCR error on page {page_num + 1}: {str(e)}")
            return ""
    
    def extract_images_from_page(self, pdf_path: str, page_num: int, 
                                 output_dir: str = None) -> List[Dict]:
        """
        Extract images from a PDF page
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            output_dir: Directory to save images (optional)
            
        Returns:
            List of image info dictionaries
        """
        images_info = []
        
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Create image info
                img_info = {
                    'page': page_num + 1,
                    'index': img_index,
                    'format': image_ext,
                    'size': len(image_bytes)
                }
                
                # Save image if output directory provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    img_filename = f"page{page_num + 1}_img{img_index}.{image_ext}"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    img_info['path'] = img_path
                    img_info['filename'] = img_filename
                
                images_info.append(img_info)
            
            doc.close()
            
        except Exception as e:
            print(f"Image extraction error on page {page_num + 1}: {str(e)}")
        
        return images_info
    
    def extract_from_pdf(self, pdf_path: str, output_path: str = None,
                        preprocess: bool = True, dpi: int = 300,
                        extract_images: bool = True, images_dir: str = None) -> Dict:
        """
        Extract text and images from PDF using hybrid approach
        
        Args:
            pdf_path: Path to PDF file
            output_path: Optional path to save extracted text
            preprocess: Whether to apply OCR preprocessing
            dpi: Resolution for OCR
            extract_images: Whether to extract images
            images_dir: Directory to save extracted images
            
        Returns:
            Dictionary with extraction results
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"\nProcessing: {pdf_path}")
        print(f"Language: {self.lang}")
        print("-" * 50)
        
        # Get page count
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        
        results = {
            'pages': [],
            'total_pages': page_count,
            'images': [],
            'full_text': ""
        }
        
        full_text = ""
        
        # Process each page
        for page_num in range(page_count):
            print(f"Processing page {page_num + 1}/{page_count}...")
            
            page_result = {
                'page_number': page_num + 1,
                'method': 'unknown',
                'confidence': 0.0,
                'text': '',
                'images_count': 0
            }
            
            # Try direct text extraction first
            text, confidence = self.extract_text_direct(pdf_path, page_num)
            
            if confidence > self.text_threshold:
                # Direct extraction successful
                page_result['method'] = 'direct'
                page_result['confidence'] = confidence
                page_result['text'] = text
                print(f"  ✓ Direct extraction (confidence: {confidence:.2f})")
            else:
                # Fall back to OCR
                print(f"  ⚠ Direct extraction failed (confidence: {confidence:.2f}), using OCR...")
                text = self.extract_text_ocr(pdf_path, page_num, preprocess, dpi)
                page_result['method'] = 'ocr'
                page_result['confidence'] = 0.8  # Assume OCR is reasonably confident
                page_result['text'] = text
                print(f"  ✓ OCR extraction completed")
            
            # Extract images if requested
            if extract_images:
                images = self.extract_images_from_page(pdf_path, page_num, images_dir)
                page_result['images_count'] = len(images)
                results['images'].extend(images)
                if images:
                    print(f"  ✓ Extracted {len(images)} image(s)")
            
            results['pages'].append(page_result)
            full_text += f"\n{'='*50}\nPage {page_num + 1} ({page_result['method'].upper()})\n{'='*50}\n{text}\n"
        
        results['full_text'] = full_text
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"\nText saved to: {output_path}")
        
        return results


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Hybrid PDF text and image extractor')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('-o', '--output', help='Output text file path')
    parser.add_argument('-l', '--lang', default='ara+eng', 
                       help='OCR language (default: ara+eng)')
    parser.add_argument('--no-preprocess', action='store_true', 
                       help='Disable OCR preprocessing')
    parser.add_argument('--dpi', type=int, default=300, 
                       help='DPI for OCR (default: 300)')
    parser.add_argument('--extract-images', action='store_true',
                       help='Extract images from PDF')
    parser.add_argument('--images-dir', default='extracted_images',
                       help='Directory to save extracted images')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = HybridPDFExtractor(lang=args.lang)
    
    # Extract text and images
    results = extractor.extract_from_pdf(
        args.pdf_path,
        output_path=args.output,
        preprocess=not args.no_preprocess,
        dpi=args.dpi,
        extract_images=args.extract_images,
        images_dir=args.images_dir if args.extract_images else None
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"Total pages: {results['total_pages']}")
    
    direct_count = sum(1 for p in results['pages'] if p['method'] == 'direct')
    ocr_count = sum(1 for p in results['pages'] if p['method'] == 'ocr')
    
    print(f"Direct extraction: {direct_count} pages")
    print(f"OCR extraction: {ocr_count} pages")
    print(f"Total images extracted: {len(results['images'])}")
    
    print("\n" + "="*50)
    print("TEXT PREVIEW (first 500 characters)")
    print("="*50)
    print(results['full_text'][:500])
    print("...")


if __name__ == "__main__":
    main()
