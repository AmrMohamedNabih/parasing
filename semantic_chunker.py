import uuid
import tiktoken
from typing import List
from nltk.tokenize import sent_tokenize
from pipeline_models import TextBlock, SemanticChunk, ChunkingConfig, TextDirection

class SemanticChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _is_new_section(self, current: TextBlock, previous: TextBlock) -> bool:
        if not previous:
            return True
        # Detect large font size changes
        if current.size and previous.size:
            if current.size > previous.size + 2.0: # Assuming heading is larger
                return True
        return False

    def chunk_page(self, page_number: int, blocks: List[TextBlock]) -> List[SemanticChunk]:
        if not blocks:
            return []
        
        # Group blocks into sections
        sections = []
        current_section = []
        
        for i, block in enumerate(blocks):
            if i == 0 or self._is_new_section(block, blocks[i-1]):
                if current_section:
                    sections.append(current_section)
                current_section = [block]
            else:
                current_section.append(block)
        if current_section:
            sections.append(current_section)
            
        final_chunks = []
        for section_blocks in sections:
            merged = self._merge_small_blocks(section_blocks, page_number)
            for m_chunk in merged:
                if m_chunk.token_count > self.config.max_tokens:
                    split_chunks = self._split_large_chunk(m_chunk, page_number)
                    final_chunks.extend(split_chunks)
                else:
                    final_chunks.append(m_chunk)
                    
        return final_chunks

    def _merge_small_blocks(self, blocks: List[TextBlock], page_number: int) -> List[SemanticChunk]:
        if not blocks:
            return []
        
        # Initial conversion to SemanticChunk
        chunks = []
        for b in blocks:
            chunks.append(SemanticChunk(
                chunk_id=str(uuid.uuid4()),
                text=b.text,
                source_block_ids=[b.block_id],
                bbox=b.bbox.copy() if b.bbox else [0, 0, 0, 0],
                token_count=self._count_tokens(b.text),
                confidence=b.confidence,
                page_number=page_number,
                direction=b.direction
            ))
            
        # Merge < min_tokens into nearest neighbor iteratively
        while True:
            small_idx = -1
            for i, c in enumerate(chunks):
                if c.token_count < self.config.min_tokens:
                    small_idx = i
                    break
            
            if small_idx == -1:
                break # No more small chunks
            
            if len(chunks) == 1:
                break # Only one chunk left, can't merge
                
            # Find nearest neighbor
            left_idx = small_idx - 1
            right_idx = small_idx + 1
            
            merge_with = -1
            if left_idx >= 0 and right_idx < len(chunks):
                # Merge with the smaller of the two neighbors to balance sizes
                if chunks[left_idx].token_count < chunks[right_idx].token_count:
                    merge_with = left_idx
                else:
                    merge_with = right_idx
            elif left_idx >= 0:
                merge_with = left_idx
            else:
                merge_with = right_idx
                
            # Perform merge
            idx1, idx2 = min(small_idx, merge_with), max(small_idx, merge_with)
            c1, c2 = chunks[idx1], chunks[idx2]
            
            merged_bbox = [
                min(c1.bbox[0], c2.bbox[0]),
                min(c1.bbox[1], c2.bbox[1]),
                max(c1.bbox[2], c2.bbox[2]),
                max(c1.bbox[3], c2.bbox[3]),
            ]
            
            merged_text = c1.text + " \n " + c2.text
            merged_chunk = SemanticChunk(
                chunk_id=str(uuid.uuid4()),
                text=merged_text,
                source_block_ids=c1.source_block_ids + c2.source_block_ids,
                bbox=merged_bbox,
                token_count=self._count_tokens(merged_text),
                confidence=(c1.confidence + c2.confidence) / 2.0,
                page_number=page_number,
                direction=c1.direction
            )
            
            chunks.pop(idx2)
            chunks[idx1] = merged_chunk

        return chunks

    def _split_large_chunk(self, chunk: SemanticChunk, page_number: int) -> List[SemanticChunk]:
        sentences = sent_tokenize(chunk.text)
        
        result = []
        current_text = ""
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)
            
            if current_tokens + sent_tokens > self.config.max_tokens and current_text:
                result.append(SemanticChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=current_text.strip(),
                    source_block_ids=chunk.source_block_ids.copy(),
                    bbox=chunk.bbox.copy(),
                    token_count=current_tokens,
                    confidence=chunk.confidence,
                    page_number=page_number,
                    direction=chunk.direction
                ))
                current_text = sentence
                current_tokens = sent_tokens
            else:
                current_text += (" " if current_text else "") + sentence
                current_tokens = self._count_tokens(current_text)
                
        if current_text:
            result.append(SemanticChunk(
                chunk_id=str(uuid.uuid4()),
                text=current_text.strip(),
                source_block_ids=chunk.source_block_ids.copy(),
                bbox=chunk.bbox.copy(),
                token_count=current_tokens,
                confidence=chunk.confidence,
                page_number=page_number,
                direction=chunk.direction
            ))
            
        return result
