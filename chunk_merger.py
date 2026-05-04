import uuid
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from pipeline_models import SemanticChunk, ChunkingConfig

class NeuralChunkMerger:
    def __init__(self):
        self.model: SentenceTransformer | None = None

    def load(self):
        if self.model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def merge_chunks(self, chunks: List[SemanticChunk], config: ChunkingConfig) -> List[SemanticChunk]:
        if len(chunks) < 2:
            return chunks

        if self.model is None:
            self.load()
            
        current_chunks = chunks.copy()
        
        while True:
            if len(current_chunks) < 2:
                break
                
            texts = [c.text for c in current_chunks]
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            
            best_sim = -1.0
            best_pair = None
            
            # Check consecutive pairs within adjacency_window
            for i in range(len(current_chunks)):
                for j in range(i + 1, min(i + 1 + config.adjacency_window, len(current_chunks))):
                    # Check token limit
                    if current_chunks[i].token_count + current_chunks[j].token_count <= config.max_tokens:
                        # Compute cosine similarity
                        sim = np.dot(embeddings[i], embeddings[j])
                        if sim >= config.similarity_threshold and sim > best_sim:
                            best_sim = sim
                            best_pair = (i, j)
            
            if best_pair is None:
                break # No eligible pairs
                
            i, j = best_pair
            c1 = current_chunks[i]
            c2 = current_chunks[j]
            
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
                token_count=c1.token_count + c2.token_count,
                confidence=(c1.confidence + c2.confidence) / 2.0,
                page_number=c1.page_number,
                direction=c1.direction
            )
            
            # Since j > i, pop j first
            current_chunks.pop(j)
            current_chunks[i] = merged_chunk
            
        return current_chunks

# Global instance to be loaded at startup
neural_chunk_merger = NeuralChunkMerger()
