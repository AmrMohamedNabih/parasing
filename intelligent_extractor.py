"""
Intelligent PDF Extractor - 7-Stage Pipeline
Combines layout-aware extraction with selective OCR for optimal accuracy and speed
"""

import os
import time
import json
import io
import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from difflib import SequenceMatcher

from pipeline_models import (
    TextBlock, ImageBlock, ExtractionStage, TextDirection,
    DirectExtractionResult, BlockOCRResult, FullPageOCRResult,
    GridOCRResult, ImageOCRResult, MergeResult, PostProcessingResult,
    PageResult, DocumentResult, StageResult
)


class IntelligentPDFExtractor:
    """
    7-Stage Intelligent PDF Extraction Pipeline
    
    Stages:
    1. Layout-Aware Direct Text Extraction (always)
    2. Block-Level OCR (selective)
    3. Full-Page OCR (safety net)
    4. Adaptive Grid OCR (scattered text)
    5. Image OCR (embedded images)
    6. Merge & De-duplication (always)
    7. Language-Aware Post-Processing (always)
    """
    
    def __init__(self, 
                 lang: str = 'ara+eng',
                 mode: str = 'balanced',
                 confidence_threshold: float = 0.3,
                 text_coverage_threshold: float = 20.0,
                 dpi: int = 300,
                 ocr_engine: str = 'easyocr'):
        """
        Initialize Intelligent PDF Extractor
        
        Args:
            lang: OCR language(s) - 'ara+eng', 'ara', 'eng'
            mode: Pipeline mode - 'fast', 'balanced', 'thorough'
            confidence_threshold: Minimum confidence for direct extraction
            text_coverage_threshold: Minimum text coverage % to skip full-page OCR
            dpi: Resolution for OCR
            ocr_engine: OCR engine to use - 'tesseract' or 'easyocr' (default: easyocr)
        """
        self.lang = lang
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.text_coverage_threshold = text_coverage_threshold
        self.dpi = dpi
        self.ocr_engine = ocr_engine
        
        # Initialize EasyOCR reader if selected
        self.easyocr_reader = None
        if ocr_engine == 'easyocr':
            self._init_easyocr()
        
        # Stage enablement based on mode
        self.stages_enabled = self._configure_stages(mode)
    
    def _init_easyocr(self):
        """Initialize EasyOCR reader"""
        try:
            import easyocr
            # Map language codes
            lang_map = {
                'ara+eng': ['ar', 'en'],
                'ara': ['ar'],
                'eng': ['en']
            }
            languages = lang_map.get(self.lang, ['ar', 'en'])
            print(f"Initializing EasyOCR with languages: {languages}")
            self.easyocr_reader = easyocr.Reader(languages, gpu=False, verbose=False)
            print("✓ EasyOCR initialized successfully")
        except ImportError:
            print("⚠ EasyOCR not installed. Install with: pip install easyocr")
            print("  Falling back to Tesseract...")
            self.ocr_engine = 'tesseract'
        except Exception as e:
            print(f"⚠ EasyOCR initialization failed: {e}")
            print("  Falling back to Tesseract...")
            self.ocr_engine = 'tesseract'
    
    def _configure_stages(self, mode: str) -> Dict[str, bool]:
        """Configure which stages are enabled based on mode"""
        if mode == 'fast':
            return {
                'direct': True,
                'block_ocr': True,
                'full_page_ocr': False,
                'grid_ocr': False,
                'image_ocr': True,
                'merge': True,
                'post_process': True
            }
        elif mode == 'thorough':
            return {
                'direct': True,
                'block_ocr': True,
                'full_page_ocr': True,
                'grid_ocr': True,
                'image_ocr': True,
                'merge': True,
                'post_process': True
            }
        else:  # balanced (default)
            return {
                'direct': True,
                'block_ocr': True,
                'full_page_ocr': True,
                'grid_ocr': False,
                'image_ocr': True,
                'merge': True,
                'post_process': True
            }
    
    # ==================== STAGE 1: LAYOUT-AWARE DIRECT EXTRACTION ====================
    
    def stage1_direct_extraction(self, pdf_path: str, page_num: int) -> DirectExtractionResult:
        """
        Stage 1: Extract embedded text directly from PDF
        ALWAYS RUNS - This is the baseline stage
        """
        start_time = time.time()
        result = DirectExtractionResult()
        
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            result.page_width = page.rect.width
            result.page_height = page.rect.height
            
            # Extract text with structure
            text_dict = page.get_text("dict")
            blocks = []
            block_id = 0
            
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    bbox = list(block.get("bbox", [0, 0, 0, 0]))
                    
                    # Extract text from block
                    text = self._extract_block_text(block)
                    if not text or len(text.strip()) < 2:
                        continue
                    
                    # Calculate confidence
                    confidence = self._calculate_direct_confidence(text, bbox)
                    
                    # Detect direction
                    direction, rtl_ratio = self._detect_text_direction(text)
                    
                    # Extract font info
                    font, size = self._extract_font_info(block)
                    
                    text_block = TextBlock(
                        block_id=f"p{page_num + 1}_b{block_id}",
                        bbox=bbox,
                        text=text,
                        confidence=confidence,
                        source_stage=ExtractionStage.DIRECT,
                        font=font,
                        size=size,
                        direction=direction,
                        rtl_ratio=rtl_ratio
                    )
                    
                    blocks.append(text_block)
                    block_id += 1
                    
                    if confidence >= self.confidence_threshold:
                        result.high_confidence_blocks += 1
                    else:
                        result.low_confidence_blocks += 1
            
            result.blocks = blocks
            result.total_blocks = len(blocks)
            doc.close()
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def _extract_block_text(self, block: Dict) -> str:
        """Extract text from a PyMuPDF block"""
        text_parts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text_parts.append(span.get("text", ""))
        return " ".join(text_parts).strip()
    
    def _calculate_direct_confidence(self, text: str, bbox: List[float]) -> float:
        """Calculate confidence score for directly extracted text"""
        if not text:
            return 0.0
        
        word_count = len(text.split())
        char_count = len(text.strip())
        
        # Low word count = low confidence
        if word_count < 3:
            return 0.2
        
        # Check for gibberish (too many single characters)
        single_chars = sum(1 for word in text.split() if len(word) == 1)
        if word_count > 0 and (single_chars / word_count) > 0.5:
            return 0.3
        
        # Calculate based on word and character count
        confidence = min(1.0, (word_count / 50) * 0.5 + (char_count / 500) * 0.5)
        return confidence
    
    def _detect_text_direction(self, text: str) -> Tuple[TextDirection, float]:
        """Detect if text is RTL (Arabic/Hebrew)"""
        if not text:
            return TextDirection.LTR, 0.0
        
        # Arabic: 0x0600-0x06FF, Hebrew: 0x0590-0x05FF
        rtl_chars = sum(1 for c in text if '\u0590' <= c <= '\u06FF')
        total_chars = len([c for c in text if c.strip()])
        
        if total_chars == 0:
            return TextDirection.LTR, 0.0
        
        rtl_ratio = rtl_chars / total_chars
        
        if rtl_ratio > 0.7:
            direction = TextDirection.RTL
        elif rtl_ratio > 0.3:
            direction = TextDirection.MIXED
        else:
            direction = TextDirection.LTR
        
        return direction, rtl_ratio
    
    def _extract_font_info(self, block: Dict) -> Tuple[Optional[str], Optional[float]]:
        """Extract font family and size from block"""
        try:
            first_line = block.get("lines", [{}])[0]
            first_span = first_line.get("spans", [{}])[0]
            font = first_span.get("font", "Unknown")
            size = first_span.get("size", 12)
            return font, size
        except:
            return None, None
    
    # ==================== STAGE 2: BLOCK-LEVEL OCR ====================
    
    def stage2_block_ocr(self, pdf_path: str, page_num: int, 
                        direct_result: DirectExtractionResult) -> BlockOCRResult:
        """
        Stage 2: Selective OCR on low-confidence blocks
        RUNS ONLY on blocks with: confidence < 0.3 OR word_count < 3 OR text == ""
        """
        start_time = time.time()
        result = BlockOCRResult()
        
        if not self.stages_enabled['block_ocr']:
            result.execution_time = time.time() - start_time
            return result
        
        try:
            improved_blocks = []
            
            for block in direct_result.blocks:
                # Check if block needs OCR
                needs_ocr = (
                    block.confidence < self.confidence_threshold or
                    block.word_count < 3 or
                    block.text.strip() == ""
                )
                
                if needs_ocr:
                    result.blocks_processed += 1
                    
                    # Run OCR on this specific block
                    ocr_text = self._ocr_block_region(pdf_path, page_num, block.bbox)
                    
                    if ocr_text and len(ocr_text.strip()) > len(block.text.strip()):
                        # OCR improved the text
                        direction, rtl_ratio = self._detect_text_direction(ocr_text)
                        
                        improved_block = TextBlock(
                            block_id=block.block_id + "_ocr",
                            bbox=block.bbox,
                            text=ocr_text,
                            confidence=0.8,  # OCR confidence
                            source_stage=ExtractionStage.BLOCK_OCR,
                            font=block.font,
                            size=block.size,
                            direction=direction,
                            rtl_ratio=rtl_ratio
                        )
                        improved_blocks.append(improved_block)
                        result.blocks_improved += 1
            
            result.blocks = improved_blocks
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def _ocr_block_region(self, pdf_path: str, page_num: int, bbox: List[float]) -> str:
        """Run OCR on a specific block region"""
        try:
            # Convert page to image
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=page_num + 1,
                last_page=page_num + 1
            )
            
            if not images:
                return ""
            
            image = images[0]
            
            # Calculate crop coordinates
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pdf_width = page.rect.width
            pdf_height = page.rect.height
            doc.close()
            
            page_width, page_height = image.size
            scale_x = page_width / pdf_width
            scale_y = page_height / pdf_height
            
            x0 = int(bbox[0] * scale_x)
            y0 = int(bbox[1] * scale_y)
            x1 = int(bbox[2] * scale_x)
            y1 = int(bbox[3] * scale_y)
            
            # Crop image
            cropped = image.crop((x0, y0, x1, y1))
            
            # Run OCR based on selected engine
            if self.ocr_engine == 'easyocr' and self.easyocr_reader:
                text = self._run_easyocr(cropped)
            else:
                # Preprocess for Tesseract
                cropped = self._preprocess_image(cropped)
                text = pytesseract.image_to_string(cropped, lang=self.lang, config='--oem 3 --psm 6')
            
            return text.strip()
            
        except Exception as e:
            print(f"Block OCR error: {str(e)}")
            return ""
    
    def _run_easyocr(self, image: Image.Image) -> str:
        """Run EasyOCR on an image"""
        try:
            if not self.easyocr_reader:
                return ""
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Run EasyOCR
            results = self.easyocr_reader.readtext(img_array, detail=1)
            
            # Combine text from all detections
            text_parts = []
            for detection in results:
                bbox, text, confidence = detection
                if confidence > 0.3:  # Only include confident detections
                    text_parts.append(text)
            
            return " ".join(text_parts)
        except Exception as e:
            print(f"EasyOCR error: {str(e)}")
            return ""
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR (Tesseract only)"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
        
        return Image.fromarray(processed)
    
    # ==================== STAGE 3: FULL-PAGE OCR ====================
    
    def stage3_full_page_ocr(self, pdf_path: str, page_num: int,
                            direct_result: DirectExtractionResult,
                            page_width: float, page_height: float) -> FullPageOCRResult:
        """
        Stage 3: Full-page OCR as safety net
        RUNS ONLY if: total_chars < 100 OR page_num == 0 OR text_coverage < 20%
        """
        start_time = time.time()
        result = FullPageOCRResult(page_image_dpi=self.dpi)
        
        if not self.stages_enabled['full_page_ocr']:
            result.execution_time = time.time() - start_time
            return result
        
        # Calculate text coverage
        total_chars = sum(block.char_count for block in direct_result.blocks)
        page_area = page_width * page_height
        text_area = sum(
            (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])
            for b in direct_result.blocks
        )
        coverage = (text_area / page_area * 100) if page_area > 0 else 0
        
        # Check if full-page OCR is needed
        needs_full_ocr = (
            total_chars < 100 or
            page_num == 0 or  # Always OCR cover page
            coverage < self.text_coverage_threshold
        )
        
        if not needs_full_ocr:
            result.execution_time = time.time() - start_time
            return result
        
        try:
            # Convert entire page to image
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=page_num + 1,
                last_page=page_num + 1
            )
            
            if not images:
                result.success = False
                result.execution_time = time.time() - start_time
                return result
            
            image = images[0]
            
            # Run OCR based on selected engine
            if self.ocr_engine == 'easyocr' and self.easyocr_reader:
                text = self._run_easyocr(image)
            else:
                # Tesseract with preprocessing
                image = self._preprocess_image(image)
                text = pytesseract.image_to_string(image, lang=self.lang, config='--oem 3 --psm 6')
            
            if text.strip():
                # Create a single block for full-page text
                direction, rtl_ratio = self._detect_text_direction(text)
                
                full_page_block = TextBlock(
                    block_id=f"p{page_num + 1}_full_ocr",
                    bbox=[0, 0, page_width, page_height],
                    text=text.strip(),
                    confidence=0.75,
                    source_stage=ExtractionStage.FULL_PAGE_OCR,
                    direction=direction,
                    rtl_ratio=rtl_ratio
                )
                result.blocks = [full_page_block]
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    # ==================== STAGE 4: ADAPTIVE GRID OCR ====================
    
    def stage4_grid_ocr(self, pdf_path: str, page_num: int,
                       page_width: float, page_height: float,
                       existing_blocks: List[TextBlock]) -> GridOCRResult:
        """
        Stage 4: Adaptive grid OCR for scattered text
        RUNS ONLY if previous stages are still insufficient
        """
        start_time = time.time()
        result = GridOCRResult()
        
        if not self.stages_enabled['grid_ocr']:
            result.execution_time = time.time() - start_time
            return result
        
        # Check if grid OCR is needed (text is still sparse)
        total_chars = sum(block.char_count for block in existing_blocks)
        if total_chars > 200:  # Sufficient text already
            result.execution_time = time.time() - start_time
            return result
        
        try:
            # Convert page to image
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=page_num + 1,
                last_page=page_num + 1
            )
            
            if not images:
                result.execution_time = time.time() - start_time
                return result
            
            image = images[0]
            img_width, img_height = image.size
            
            # Detect text-sparse regions (adaptive grid)
            regions = self._detect_sparse_regions(image, existing_blocks, page_width, page_height)
            result.grid_regions = len(regions)
            
            grid_blocks = []
            for i, region in enumerate(regions):
                x0, y0, x1, y1 = region
                cropped = image.crop((x0, y0, x1, y1))
                
                # Run OCR based on selected engine
                if self.ocr_engine == 'easyocr' and self.easyocr_reader:
                    text = self._run_easyocr(cropped)
                else:
                    cropped = self._preprocess_image(cropped)
                    text = pytesseract.image_to_string(cropped, lang=self.lang, config='--oem 3 --psm 6')
                
                if text.strip():
                    result.regions_with_text += 1
                    direction, rtl_ratio = self._detect_text_direction(text)
                    
                    # Convert image coordinates back to PDF coordinates
                    scale_x = page_width / img_width
                    scale_y = page_height / img_height
                    pdf_bbox = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
                    
                    grid_block = TextBlock(
                        block_id=f"p{page_num + 1}_grid{i}",
                        bbox=pdf_bbox,
                        text=text.strip(),
                        confidence=0.7,
                        source_stage=ExtractionStage.GRID_OCR,
                        direction=direction,
                        rtl_ratio=rtl_ratio
                    )
                    grid_blocks.append(grid_block)
            
            result.blocks = grid_blocks
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def _detect_sparse_regions(self, image: Image.Image, existing_blocks: List[TextBlock],
                               page_width: float, page_height: float) -> List[List[int]]:
        """Detect regions with sparse text coverage for adaptive grid OCR"""
        img_width, img_height = image.size
        
        # Divide into 3x3 grid
        grid_w = img_width // 3
        grid_h = img_height // 3
        
        sparse_regions = []
        
        for row in range(3):
            for col in range(3):
                x0 = col * grid_w
                y0 = row * grid_h
                x1 = x0 + grid_w
                y1 = y0 + grid_h
                
                # Convert to PDF coordinates to check overlap
                scale_x = page_width / img_width
                scale_y = page_height / img_height
                pdf_region = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
                
                # Check if this region has existing text
                has_text = any(
                    self._bbox_overlap(pdf_region, block.bbox) > 0.3
                    for block in existing_blocks
                )
                
                if not has_text:
                    sparse_regions.append([x0, y0, x1, y1])
        
        return sparse_regions
    
    def _bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU overlap between two bounding boxes"""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        
        x0_i = max(x0_1, x0_2)
        y0_i = max(y0_1, y0_2)
        x1_i = min(x1_1, x1_2)
        y1_i = min(y1_1, y1_2)
        
        if x1_i < x0_i or y1_i < y0_i:
            return 0.0
        
        intersection = (x1_i - x0_i) * (y1_i - y0_i)
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # ==================== STAGE 5: IMAGE OCR ====================
    
    def stage5_image_ocr(self, pdf_path: str, page_num: int,
                        images_dir: Optional[str] = None) -> ImageOCRResult:
        """
        Stage 5: Extract and OCR embedded images
        RUNS ONLY if page has embedded images
        """
        start_time = time.time()
        result = ImageOCRResult()
        
        if not self.stages_enabled['image_ocr']:
            result.execution_time = time.time() - start_time
            return result
        
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            image_list = page.get_images()
            
            if not image_list:
                doc.close()
                result.execution_time = time.time() - start_time
                return result
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Get image position (approximate)
                rects = page.get_image_rects(xref)
                bbox = list(rects[0]) if rects else [0, 0, 100, 100]
                
                # Save image if directory provided
                img_path = None
                if images_dir:
                    os.makedirs(images_dir, exist_ok=True)
                    img_filename = f"page{page_num + 1}_img{img_index}.{image_ext}"
                    img_path = os.path.join(images_dir, img_filename)
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                
                # Run OCR on image
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Run OCR based on selected engine
                    if self.ocr_engine == 'easyocr' and self.easyocr_reader:
                        ocr_text = self._run_easyocr(pil_image)
                    else:
                        pil_image = self._preprocess_image(pil_image)
                        ocr_text = pytesseract.image_to_string(pil_image, lang=self.lang, config='--oem 3 --psm 6')
                    
                    if ocr_text.strip():
                        result.images_with_text += 1
                except:
                    ocr_text = ""
                
                image_block = ImageBlock(
                    image_id=f"p{page_num + 1}_img{img_index}",
                    bbox=bbox,
                    image_path=img_path,
                    ocr_text=ocr_text.strip(),
                    confidence=0.7 if ocr_text.strip() else 0.0,
                    format=image_ext
                )
                result.images.append(image_block)
                
                # Also add as text block if has OCR text
                if ocr_text.strip():
                    direction, rtl_ratio = self._detect_text_direction(ocr_text)
                    text_block = TextBlock(
                        block_id=f"p{page_num + 1}_img{img_index}_text",
                        bbox=bbox,
                        text=ocr_text.strip(),
                        confidence=0.7,
                        source_stage=ExtractionStage.IMAGE_OCR,
                        direction=direction,
                        rtl_ratio=rtl_ratio
                    )
                    result.blocks.append(text_block)
            
            doc.close()
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    # ==================== STAGE 6: MERGE & DE-DUPLICATION ====================
    
    def stage6_merge_deduplication(self, all_stage_results: List[StageResult],
                                   page_num: int) -> MergeResult:
        """
        Stage 6: Merge outputs from all stages and remove duplicates
        ALWAYS RUNS - Critical for preventing duplicate text
        """
        start_time = time.time()
        result = MergeResult()
        
        try:
            # Collect all blocks from all stages
            all_blocks = []
            for stage_result in all_stage_results:
                all_blocks.extend(stage_result.blocks)
            
            result.total_input_blocks = len(all_blocks)
            
            if not all_blocks:
                result.execution_time = time.time() - start_time
                return result
            
            # Priority order: Direct > Block OCR > Full OCR > Grid OCR > Image OCR
            stage_priority = {
                ExtractionStage.DIRECT: 5,
                ExtractionStage.BLOCK_OCR: 4,
                ExtractionStage.FULL_PAGE_OCR: 3,
                ExtractionStage.GRID_OCR: 2,
                ExtractionStage.IMAGE_OCR: 1
            }
            
            # Remove duplicates
            unique_blocks = []
            removed_count = 0
            
            for block in all_blocks:
                is_duplicate = False
                
                for existing in unique_blocks:
                    # Check for overlap
                    if block.overlaps_with(existing, threshold=0.7):
                        # Check text similarity
                        similarity = self._text_similarity(block.text, existing.text)
                        
                        if similarity > 0.8:
                            # Duplicate found - keep higher priority
                            is_duplicate = True
                            
                            if stage_priority.get(block.source_stage, 0) > stage_priority.get(existing.source_stage, 0):
                                # Replace with higher priority block
                                unique_blocks.remove(existing)
                                unique_blocks.append(block)
                                result.blocks_merged += 1
                            
                            removed_count += 1
                            break
                
                if not is_duplicate:
                    unique_blocks.append(block)
            
            result.duplicates_removed = removed_count
            result.blocks = unique_blocks
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    # ==================== STAGE 7: LANGUAGE-AWARE POST-PROCESSING ====================
    
    def stage7_post_processing(self, blocks: List[TextBlock]) -> PostProcessingResult:
        """
        Stage 7: Language-aware cleanup and normalization
        ALWAYS RUNS - Final cleanup for better quality
        """
        start_time = time.time()
        result = PostProcessingResult()
        
        try:
            processed_blocks = []
            
            for block in blocks:
                # Apply post-processing
                processed_text = block.text
                
                # Arabic-specific processing
                if block.rtl_ratio > 0.3:
                    processed_text = self._process_arabic_text(processed_text)
                    result.arabic_blocks_reshaped += 1
                
                # Fix broken words
                processed_text = self._fix_broken_words(processed_text)
                
                # Remove OCR artifacts
                processed_text = self._remove_ocr_artifacts(processed_text)
                
                # Normalize whitespace
                processed_text = self._normalize_whitespace(processed_text)
                
                if processed_text != block.text:
                    result.words_fixed += 1
                
                # Create processed block
                processed_block = TextBlock(
                    block_id=block.block_id,
                    bbox=block.bbox,
                    text=processed_text,
                    confidence=block.confidence,
                    source_stage=ExtractionStage.POST_PROCESSED,
                    font=block.font,
                    size=block.size,
                    direction=block.direction,
                    rtl_ratio=block.rtl_ratio,
                    column=block.column
                )
                processed_blocks.append(processed_block)
            
            result.blocks = processed_blocks
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def _process_arabic_text(self, text: str) -> str:
        """Process Arabic text - reshape and reorder"""
        try:
            # Remove tatweel (Arabic kashida)
            text = text.replace('\u0640', '')
            
            # Normalize Arabic characters
            text = text.replace('\u0622', '\u0627')  # Alef with madda
            text = text.replace('\u0623', '\u0627')  # Alef with hamza above
            text = text.replace('\u0625', '\u0627')  # Alef with hamza below
            
            # Note: For full Arabic reshaping, you might want to use python-bidi or arabic-reshaper
            # For now, we'll do basic normalization
            
            return text
        except:
            return text
    
    def _fix_broken_words(self, text: str) -> str:
        """Fix common OCR word-breaking issues"""
        # Remove spaces between single characters that should be together
        text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', text)
        
        # Fix common broken words (can be expanded)
        text = text.replace('th e', 'the')
        text = text.replace('an d', 'and')
        text = text.replace('i s', 'is')
        
        return text
    
    def _remove_ocr_artifacts(self, text: str) -> str:
        """Remove common OCR artifacts"""
        # Remove standalone special characters
        text = re.sub(r'\s+[|~`]\s+', ' ', text)
        
        # Remove multiple consecutive special characters
        text = re.sub(r'[|~`]{2,}', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{3,}', '--', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        
        return text
    
    # ==================== MAIN EXTRACTION ORCHESTRATION ====================
    
    def extract_page(self, pdf_path: str, page_num: int,
                    extract_images: bool = True,
                    images_dir: Optional[str] = None) -> PageResult:
        """
        Extract content from a single page using the 7-stage pipeline
        """
        start_time = time.time()
        
        # Stage 1: Direct Extraction (ALWAYS)
        print(f"  Stage 1: Direct extraction...")
        stage1_result = self.stage1_direct_extraction(pdf_path, page_num)
        
        page_width = stage1_result.page_width
        page_height = stage1_result.page_height
        
        # Stage 2: Block-Level OCR (SELECTIVE)
        print(f"  Stage 2: Block-level OCR...")
        stage2_result = self.stage2_block_ocr(pdf_path, page_num, stage1_result)
        
        # Stage 3: Full-Page OCR (SAFETY NET)
        print(f"  Stage 3: Full-page OCR...")
        stage3_result = self.stage3_full_page_ocr(pdf_path, page_num, stage1_result,
                                                   page_width, page_height)
        
        # Collect blocks so far
        current_blocks = stage1_result.blocks + stage2_result.blocks + stage3_result.blocks
        
        # Stage 4: Adaptive Grid OCR (CONDITIONAL)
        print(f"  Stage 4: Adaptive grid OCR...")
        stage4_result = self.stage4_grid_ocr(pdf_path, page_num, page_width, page_height,
                                             current_blocks)
        
        # Stage 5: Image OCR (IF IMAGES EXIST)
        print(f"  Stage 5: Image OCR...")
        stage5_result = self.stage5_image_ocr(pdf_path, page_num, images_dir if extract_images else None)
        
        # Stage 6: Merge & De-duplication (ALWAYS)
        print(f"  Stage 6: Merge & de-duplication...")
        all_stage_results = [stage1_result, stage2_result, stage3_result, stage4_result, stage5_result]
        stage6_result = self.stage6_merge_deduplication(all_stage_results, page_num)
        
        # Stage 7: Post-Processing (ALWAYS)
        print(f"  Stage 7: Post-processing...")
        stage7_result = self.stage7_post_processing(stage6_result.blocks)
        
        # Detect columns
        columns = self._detect_columns(stage7_result.blocks, page_width)
        
        # Assign columns to blocks
        for block in stage7_result.blocks:
            block.column = self._assign_column(block.bbox, page_width, columns)
        
        # Create page result
        page_result = PageResult(
            page_number=page_num + 1,
            width=page_width,
            height=page_height,
            blocks=stage7_result.blocks,
            images=stage5_result.images,
            columns=columns,
            total_execution_time=time.time() - start_time
        )
        
        # Store stage results
        page_result.stage_results = {
            ExtractionStage.DIRECT: stage1_result,
            ExtractionStage.BLOCK_OCR: stage2_result,
            ExtractionStage.FULL_PAGE_OCR: stage3_result,
            ExtractionStage.GRID_OCR: stage4_result,
            ExtractionStage.IMAGE_OCR: stage5_result,
            ExtractionStage.MERGED: stage6_result,
            ExtractionStage.POST_PROCESSED: stage7_result
        }
        
        return page_result
    
    def _detect_columns(self, blocks: List[TextBlock], page_width: float) -> int:
        """Detect number of columns based on block positions"""
        if not blocks:
            return 1
        
        # Get X-coordinates of block centers
        x_centers = []
        for block in blocks:
            if block.bbox:
                x_center = (block.bbox[0] + block.bbox[2]) / 2
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
    
    def _assign_column(self, bbox: List[float], page_width: float, num_columns: int) -> int:
        """Assign block to a column"""
        if num_columns == 1:
            return 1
        
        x_center = (bbox[0] + bbox[2]) / 2
        column_width = page_width / num_columns
        
        column = int(x_center / column_width) + 1
        return min(column, num_columns)
    
    def extract_from_pdf(self, pdf_path: str,
                        output_json_path: Optional[str] = None,
                        extract_images: bool = True,
                        images_dir: Optional[str] = None) -> DocumentResult:
        """
        Extract content from entire PDF using the 7-stage intelligent pipeline
        
        Args:
            pdf_path: Path to PDF file
            output_json_path: Optional path to save JSON output
            extract_images: Whether to extract embedded images
            images_dir: Directory to save extracted images
            
        Returns:
            DocumentResult with complete extraction
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"\n{'='*60}")
        print(f"Intelligent PDF Extraction Pipeline")
        print(f"{'='*60}")
        print(f"File: {pdf_path}")
        print(f"Mode: {self.mode}")
        print(f"Language: {self.lang}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Get document info
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        metadata = doc.metadata
        doc.close()
        
        # Create document result
        doc_result = DocumentResult(
            filename=os.path.basename(pdf_path),
            total_pages=total_pages,
            metadata=metadata
        )
        
        # Process each page
        for page_num in range(total_pages):
            print(f"\nProcessing page {page_num + 1}/{total_pages}...")
            
            page_result = self.extract_page(pdf_path, page_num, extract_images, images_dir)
            doc_result.pages.append(page_result)
            
            # Print page summary
            print(f"  ✓ Extracted {len(page_result.blocks)} blocks")
            print(f"  ✓ Stage breakdown: {page_result.stage_stats}")
            print(f"  ✓ Confidence: {page_result.total_confidence:.2f}")
            print(f"  ✓ Time: {page_result.total_execution_time:.2f}s")
        
        doc_result.total_execution_time = time.time() - start_time
        
        # Save to JSON if requested
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(doc_result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\n✓ Saved to: {output_json_path}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total pages: {doc_result.total_pages}")
        print(f"Total blocks: {doc_result.total_blocks}")
        print(f"Total images: {doc_result.total_images}")
        print(f"Average confidence: {doc_result.avg_confidence:.2f}")
        print(f"Overall stage stats: {doc_result.overall_stage_stats}")
        print(f"Total time: {doc_result.total_execution_time:.2f}s")
        print(f"{'='*60}\n")
        
        return doc_result


def main():
    """CLI for intelligent PDF extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intelligent 7-Stage PDF Extractor')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('-l', '--lang', default='ara+eng', help='OCR language')
    parser.add_argument('-m', '--mode', default='balanced',
                       choices=['fast', 'balanced', 'thorough'],
                       help='Pipeline mode')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for OCR')
    parser.add_argument('--ocr-engine', default='easyocr',
                       choices=['tesseract', 'easyocr'],
                       help='OCR engine (default: easyocr for better accuracy)')
    parser.add_argument('--extract-images', action='store_true', help='Extract images')
    parser.add_argument('--images-dir', default='extracted_images', help='Images directory')
    
    args = parser.parse_args()
    
    extractor = IntelligentPDFExtractor(
        lang=args.lang,
        mode=args.mode,
        dpi=args.dpi,
        ocr_engine=args.ocr_engine
    )
    
    result = extractor.extract_from_pdf(
        args.pdf_path,
        output_json_path=args.output,
        extract_images=args.extract_images,
        images_dir=args.images_dir if args.extract_images else None
    )


if __name__ == "__main__":
    main()

