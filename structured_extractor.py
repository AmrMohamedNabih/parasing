"""
Structured PDF Extractor
Extracts PDF content with block-level structure, column detection, and RTL awareness
"""

import os
import json
import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re


class StructuredPDFExtractor:
    """Extract PDF content with preserved structure"""
    
    def __init__(self, lang: str = 'ara+eng', confidence_threshold: float = 0.1):
        """
        Initialize Structured PDF Extractor
        
        Args:
            lang: Language(s) for OCR
            confidence_threshold: Minimum confidence for direct extraction
        """
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        
    def detect_rtl(self, text: str) -> Tuple[str, float]:
        """
        Detect if text is RTL (Arabic/Hebrew)
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (direction, rtl_ratio)
        """
        if not text:
            return "ltr", 0.0
        
        # Arabic: 0x0600-0x06FF, Hebrew: 0x0590-0x05FF
        rtl_chars = sum(1 for c in text if '\u0590' <= c <= '\u06FF')
        total_chars = len([c for c in text if c.strip()])
        
        if total_chars == 0:
            return "ltr", 0.0
        
        rtl_ratio = rtl_chars / total_chars
        direction = "rtl" if rtl_ratio > 0.3 else "ltr"
        
        return direction, rtl_ratio
    
    def detect_columns(self, blocks: List[Dict], page_width: float) -> int:
        """
        Detect number of columns based on block positions
        
        Args:
            blocks: List of text blocks with bbox
            page_width: Page width
            
        Returns:
            Number of columns detected
        """
        if not blocks:
            return 1
        
        # Get X-coordinates of block centers
        x_centers = []
        for block in blocks:
            if block['type'] == 'text' and block.get('bbox'):
                x0, y0, x1, y1 = block['bbox']
                x_center = (x0 + x1) / 2
                x_centers.append(x_center)
        
        if len(x_centers) < 2:
            return 1
        
        # Simple clustering: divide page into thirds
        left_third = page_width / 3
        right_third = 2 * page_width / 3
        
        left_blocks = sum(1 for x in x_centers if x < left_third)
        middle_blocks = sum(1 for x in x_centers if left_third <= x < right_third)
        right_blocks = sum(1 for x in x_centers if x >= right_third)
        
        # Determine columns
        if left_blocks > 0 and right_blocks > 0 and middle_blocks < 2:
            return 2
        elif left_blocks > 0 and middle_blocks > 0 and right_blocks > 0:
            return 3
        else:
            return 1
    
    def assign_column(self, bbox: List[float], page_width: float, num_columns: int) -> int:
        """
        Assign block to a column
        
        Args:
            bbox: Block bounding box [x0, y0, x1, y1]
            page_width: Page width
            num_columns: Number of columns
            
        Returns:
            Column number (1-indexed)
        """
        if num_columns == 1:
            return 1
        
        x_center = (bbox[0] + bbox[2]) / 2
        column_width = page_width / num_columns
        
        column = int(x_center / column_width) + 1
        return min(column, num_columns)
    
    def extract_text_block_direct(self, page, block_bbox: List[float]) -> Tuple[str, float]:
        """
        Extract text from a specific block using direct extraction
        
        Args:
            page: PyMuPDF page object
            block_bbox: Block bounding box
            
        Returns:
            Tuple of (text, confidence)
        """
        try:
            # Extract text from specific region
            rect = fitz.Rect(block_bbox)
            text = page.get_textbox(rect)
            
            if not text or len(text.strip()) < 3:
                return "", 0.0
            
            # Simple validation
            word_count = len(text.split())
            confidence = min(1.0, word_count / 10)
            
            return text.strip(), confidence
            
        except Exception as e:
            return "", 0.0
    
    def extract_text_block_ocr(self, pdf_path: str, page_num: int, 
                               block_bbox: List[float], dpi: int = 300) -> str:
        """
        Extract text from a block using OCR
        
        Args:
            pdf_path: Path to PDF
            page_num: Page number (0-indexed)
            block_bbox: Block bounding box [x0, y0, x1, y1]
            dpi: Resolution for conversion
            
        Returns:
            Extracted text
        """
        try:
            # Convert page to image
            images = convert_from_path(
                pdf_path, 
                dpi=dpi, 
                first_page=page_num + 1, 
                last_page=page_num + 1
            )
            
            if not images:
                return ""
            
            image = images[0]
            
            # Calculate crop coordinates (scale bbox to image size)
            page_width, page_height = image.size
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pdf_width = page.rect.width
            pdf_height = page.rect.height
            doc.close()
            
            scale_x = page_width / pdf_width
            scale_y = page_height / pdf_height
            
            x0 = int(block_bbox[0] * scale_x)
            y0 = int(block_bbox[1] * scale_y)
            x1 = int(block_bbox[2] * scale_x)
            y1 = int(block_bbox[3] * scale_y)
            
            # Crop to block
            cropped = image.crop((x0, y0, x1, y1))
            
            # Preprocess
            cropped = self.preprocess_image(cropped)
            
            # OCR
            text = pytesseract.image_to_string(cropped, lang=self.lang, config='--oem 3 --psm 6')
            
            return text.strip()
            
        except Exception as e:
            print(f"OCR error on block: {str(e)}")
            return ""
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for OCR"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(processed)
    
    def extract_page_structure(self, pdf_path: str, page_num: int, 
                               extract_images: bool = True, 
                               images_dir: str = None,
                               dpi: int = 300) -> Dict:
        """
        Extract structured content from a single page
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            extract_images: Whether to extract images
            images_dir: Directory to save images
            dpi: DPI for OCR
            
        Returns:
            Page structure dictionary
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        page_dict = {
            'page_number': page_num + 1,
            'width': page.rect.width,
            'height': page.rect.height,
            'blocks': []
        }
        
        # Extract text blocks
        text_dict = page.get_text("dict")
        block_id = 0
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                bbox = block.get("bbox", [0, 0, 0, 0])
                
                # Try direct extraction first
                text, confidence = self.extract_text_block_direct(page, bbox)
                extraction_method = "direct"
                
                # Fall back to OCR if confidence is low
                if confidence < self.confidence_threshold:
                    text = self.extract_text_block_ocr(pdf_path, page_num, bbox, dpi)
                    extraction_method = "ocr"
                    confidence = 0.8
                
                if not text:
                    continue
                
                # Detect RTL
                direction, rtl_ratio = self.detect_rtl(text)
                
                # Extract font info
                font_info = {}
                if block.get("lines"):
                    first_span = block["lines"][0].get("spans", [{}])[0]
                    font_info = {
                        "font": first_span.get("font", "Unknown"),
                        "size": first_span.get("size", 12)
                    }
                
                block_data = {
                    'block_id': f"p{page_num + 1}_b{block_id}",
                    'type': 'text',
                    'bbox': list(bbox),
                    'extraction_method': extraction_method,
                    'confidence': round(confidence, 2),
                    'text': text,
                    'direction': direction,
                    'rtl_ratio': round(rtl_ratio, 2),
                    **font_info
                }
                
                page_dict['blocks'].append(block_data)
                block_id += 1
            
            elif block.get("type") == 1 and extract_images:  # Image block
                bbox = block.get("bbox", [0, 0, 0, 0])
                
                # Extract image
                img_info = self.extract_image_block(doc, page, page_num, block_id, 
                                                    bbox, images_dir, pdf_path, dpi)
                
                if img_info:
                    page_dict['blocks'].append(img_info)
                    block_id += 1
        
        # Detect columns
        num_columns = self.detect_columns(page_dict['blocks'], page.rect.width)
        page_dict['columns'] = num_columns
        
        # Assign columns to blocks
        for block in page_dict['blocks']:
            if block.get('bbox'):
                block['column'] = self.assign_column(block['bbox'], page.rect.width, num_columns)
        
        doc.close()
        return page_dict
    
    def extract_image_block(self, doc, page, page_num: int, block_id: int,
                           bbox: List[float], images_dir: str, 
                           pdf_path: str, dpi: int) -> Optional[Dict]:
        """Extract and OCR an image block"""
        try:
            # Get images in this region
            image_list = page.get_images()
            
            if not image_list:
                return None
            
            # Extract first image (simplified)
            xref = image_list[0][0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image
            img_filename = None
            if images_dir:
                os.makedirs(images_dir, exist_ok=True)
                img_filename = f"page{page_num + 1}_img{block_id}.{image_ext}"
                img_path = os.path.join(images_dir, img_filename)
                
                with open(img_path, "wb") as img_file:
                    img_file.write(image_bytes)
            
            # Try OCR on image
            ocr_text = ""
            try:
                ocr_text = self.extract_text_block_ocr(pdf_path, page_num, bbox, dpi)
            except:
                pass
            
            return {
                'block_id': f"p{page_num + 1}_b{block_id}",
                'type': 'image',
                'bbox': list(bbox),
                'image_path': img_filename,
                'ocr_text': ocr_text,
                'confidence': 0.7 if ocr_text else 0.0
            }
            
        except Exception as e:
            print(f"Image extraction error: {str(e)}")
            return None
    
    def extract_structured(self, pdf_path: str, output_json_path: str = None,
                          extract_images: bool = True, images_dir: str = None,
                          dpi: int = 300) -> Dict:
        """
        Extract structured content from entire PDF
        
        Args:
            pdf_path: Path to PDF file
            output_json_path: Path to save JSON output
            extract_images: Whether to extract images
            images_dir: Directory to save images
            dpi: DPI for OCR
            
        Returns:
            Complete document structure
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"\nExtracting structured content from: {pdf_path}")
        print("-" * 50)
        
        doc = fitz.open(pdf_path)
        
        # Document metadata
        result = {
            'document': {
                'filename': os.path.basename(pdf_path),
                'total_pages': len(doc),
                'metadata': doc.metadata
            },
            'pages': []
        }
        
        doc.close()
        
        # Extract each page
        for page_num in range(result['document']['total_pages']):
            print(f"Processing page {page_num + 1}/{result['document']['total_pages']}...")
            
            page_structure = self.extract_page_structure(
                pdf_path, page_num, extract_images, images_dir, dpi
            )
            
            result['pages'].append(page_structure)
            
            # Print summary
            text_blocks = sum(1 for b in page_structure['blocks'] if b['type'] == 'text')
            image_blocks = sum(1 for b in page_structure['blocks'] if b['type'] == 'image')
            direct_blocks = sum(1 for b in page_structure['blocks'] 
                              if b.get('extraction_method') == 'direct')
            ocr_blocks = sum(1 for b in page_structure['blocks'] 
                           if b.get('extraction_method') == 'ocr')
            
            print(f"  Blocks: {text_blocks} text, {image_blocks} images")
            print(f"  Extraction: {direct_blocks} direct, {ocr_blocks} OCR")
            print(f"  Columns: {page_structure['columns']}")
        
        # Save JSON
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nStructured data saved to: {output_json_path}")
        
        return result


def main():
    """CLI for structured extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Structured PDF extractor')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--extract-images', action='store_true', help='Extract images')
    parser.add_argument('--images-dir', default='extracted_images', help='Images directory')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for OCR')
    
    args = parser.parse_args()
    
    extractor = StructuredPDFExtractor()
    
    result = extractor.extract_structured(
        args.pdf_path,
        output_json_path=args.output,
        extract_images=args.extract_images,
        images_dir=args.images_dir if args.extract_images else None,
        dpi=args.dpi
    )
    
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total pages: {result['document']['total_pages']}")
    print(f"Total blocks: {sum(len(p['blocks']) for p in result['pages'])}")


if __name__ == "__main__":
    main()
