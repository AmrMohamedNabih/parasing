"""
Data Models for 7-Stage Intelligent PDF Extraction Pipeline
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ExtractionStage(Enum):
    """Enumeration of pipeline stages"""
    DIRECT = "direct"
    BLOCK_OCR = "block_ocr"
    FULL_PAGE_OCR = "full_page_ocr"
    GRID_OCR = "grid_ocr"
    IMAGE_OCR = "image_ocr"
    MERGED = "merged"
    POST_PROCESSED = "post_processed"


class TextDirection(Enum):
    """Text direction"""
    LTR = "ltr"
    RTL = "rtl"
    MIXED = "mixed"


@dataclass
class TextBlock:
    """Represents a single text block from any extraction stage"""
    block_id: str
    bbox: List[float]  # [x0, y0, x1, y1]
    text: str
    confidence: float
    source_stage: ExtractionStage
    font: Optional[str] = None
    size: Optional[float] = None
    direction: TextDirection = TextDirection.LTR
    rtl_ratio: float = 0.0
    word_count: int = 0
    char_count: int = 0
    column: Optional[int] = None
    
    def __post_init__(self):
        """Calculate word and char counts if not provided"""
        if self.word_count == 0:
            self.word_count = len(self.text.split())
        if self.char_count == 0:
            self.char_count = len(self.text.strip())
    
    def overlaps_with(self, other: 'TextBlock', threshold: float = 0.7) -> bool:
        """Check if this block overlaps with another block (IoU)"""
        x0_1, y0_1, x1_1, y1_1 = self.bbox
        x0_2, y0_2, x1_2, y1_2 = other.bbox
        
        # Calculate intersection
        x0_i = max(x0_1, x0_2)
        y0_i = max(y0_1, y0_2)
        x1_i = min(x1_1, x1_2)
        y1_i = min(y1_1, y1_2)
        
        if x1_i < x0_i or y1_i < y0_i:
            return False
        
        intersection = (x1_i - x0_i) * (y1_i - y0_i)
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou >= threshold


@dataclass
class ImageBlock:
    """Represents an extracted image with optional OCR text"""
    image_id: str
    bbox: List[float]
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    ocr_text: str = ""
    confidence: float = 0.0
    format: str = "png"


@dataclass
class StageResult:
    """Base class for stage results"""
    stage: ExtractionStage
    blocks: List[TextBlock] = field(default_factory=list)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def block_count(self) -> int:
        return len(self.blocks)
    
    @property
    def total_chars(self) -> int:
        return sum(block.char_count for block in self.blocks)
    
    @property
    def avg_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return sum(block.confidence for block in self.blocks) / len(self.blocks)


@dataclass
class DirectExtractionResult(StageResult):
    """Result from Stage 1: Direct Text Extraction"""
    stage: ExtractionStage = ExtractionStage.DIRECT
    page_width: float = 0.0
    page_height: float = 0.0
    total_blocks: int = 0
    high_confidence_blocks: int = 0
    low_confidence_blocks: int = 0


@dataclass
class BlockOCRResult(StageResult):
    """Result from Stage 2: Block-Level OCR"""
    stage: ExtractionStage = ExtractionStage.BLOCK_OCR
    blocks_processed: int = 0
    blocks_improved: int = 0


@dataclass
class FullPageOCRResult(StageResult):
    """Result from Stage 3: Full-Page OCR"""
    stage: ExtractionStage = ExtractionStage.FULL_PAGE_OCR
    page_image_dpi: int = 300
    preprocessing_applied: bool = True


@dataclass
class GridOCRResult(StageResult):
    """Result from Stage 4: Adaptive Grid OCR"""
    stage: ExtractionStage = ExtractionStage.GRID_OCR
    grid_regions: int = 0
    regions_with_text: int = 0


@dataclass
class ImageOCRResult(StageResult):
    """Result from Stage 5: Image OCR"""
    stage: ExtractionStage = ExtractionStage.IMAGE_OCR
    images: List[ImageBlock] = field(default_factory=list)
    images_with_text: int = 0


@dataclass
class MergeResult(StageResult):
    """Result from Stage 6: Merge & De-duplication"""
    stage: ExtractionStage = ExtractionStage.MERGED
    total_input_blocks: int = 0
    duplicates_removed: int = 0
    blocks_merged: int = 0


@dataclass
class PostProcessingResult(StageResult):
    """Result from Stage 7: Language-Aware Post-Processing"""
    stage: ExtractionStage = ExtractionStage.POST_PROCESSED
    arabic_blocks_reshaped: int = 0
    words_fixed: int = 0
    artifacts_removed: int = 0


@dataclass
class PageResult:
    """Complete result for a single page"""
    page_number: int
    width: float
    height: float
    blocks: List[TextBlock]
    images: List[ImageBlock] = field(default_factory=list)
    stage_results: Dict[ExtractionStage, StageResult] = field(default_factory=dict)
    columns: int = 1
    total_execution_time: float = 0.0
    
    @property
    def stage_stats(self) -> Dict[str, int]:
        """Count blocks by source stage"""
        stats = {}
        for block in self.blocks:
            stage_name = block.source_stage.value
            stats[stage_name] = stats.get(stage_name, 0) + 1
        return stats
    
    @property
    def total_confidence(self) -> float:
        """Average confidence across all blocks"""
        if not self.blocks:
            return 0.0
        return sum(block.confidence for block in self.blocks) / len(self.blocks)
    
    @property
    def text_coverage(self) -> float:
        """Percentage of page covered by text"""
        if self.width == 0 or self.height == 0:
            return 0.0
        
        page_area = self.width * self.height
        text_area = sum(
            (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1])
            for block in self.blocks
        )
        return (text_area / page_area) * 100 if page_area > 0 else 0.0


@dataclass
class DocumentResult:
    """Complete result for entire document"""
    filename: str
    total_pages: int
    pages: List[PageResult] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    total_execution_time: float = 0.0
    
    @property
    def total_blocks(self) -> int:
        return sum(len(page.blocks) for page in self.pages)
    
    @property
    def total_images(self) -> int:
        return sum(len(page.images) for page in self.pages)
    
    @property
    def overall_stage_stats(self) -> Dict[str, int]:
        """Aggregate stage statistics across all pages"""
        stats = {}
        for page in self.pages:
            for stage, count in page.stage_stats.items():
                stats[stage] = stats.get(stage, 0) + count
        return stats
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence across all pages"""
        if not self.pages:
            return 0.0
        confidences = [page.total_confidence for page in self.pages if page.total_confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'document': {
                'filename': self.filename,
                'total_pages': self.total_pages,
                'total_blocks': self.total_blocks,
                'total_images': self.total_images,
                'avg_confidence': round(self.avg_confidence, 2),
                'total_execution_time': round(self.total_execution_time, 2),
                'stage_stats': self.overall_stage_stats,
                'metadata': self.metadata
            },
            'pages': [
                {
                    'page_number': page.page_number,
                    'width': page.width,
                    'height': page.height,
                    'columns': page.columns,
                    'text_coverage': round(page.text_coverage, 2),
                    'total_confidence': round(page.total_confidence, 2),
                    'execution_time': round(page.total_execution_time, 2),
                    'stage_stats': page.stage_stats,
                    'blocks': [
                        {
                            'block_id': block.block_id,
                            'bbox': block.bbox,
                            'text': block.text,
                            'confidence': round(block.confidence, 2),
                            'source_stage': block.source_stage.value,
                            'font': block.font,
                            'size': block.size,
                            'direction': block.direction.value,
                            'rtl_ratio': round(block.rtl_ratio, 2),
                            'word_count': block.word_count,
                            'column': block.column
                        }
                        for block in page.blocks
                    ],
                    'images': [
                        {
                            'image_id': img.image_id,
                            'bbox': img.bbox,
                            'image_path': img.image_path,
                            'ocr_text': img.ocr_text,
                            'confidence': round(img.confidence, 2),
                            'format': img.format
                        }
                        for img in page.images
                    ]
                }
                for page in self.pages
            ]
        }
