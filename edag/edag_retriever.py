"""
EDAG Retriever
--------------
Implements the 4-stage hierarchical beam search over a pre-built EDAG graph.

The retriever is loaded lazily on first query for a subject and cached in
memory. When edag.build.completed arrives for a subject_id, its cache entry
is invalidated so the next query loads the fresh graph.

EDAGScoredPoint duck-types Qdrant's ScoredPoint (.payload, .score) so it
flows through the existing search_service post-processing unchanged.
"""

import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Thread pool for non-blocking graph searches
_search_executor = ThreadPoolExecutor(max_workers=4)

# Fixed system parameters (must match builder)
TM       = 3     # top meta nodes to expand
DELTA    = 0.10  # beam margin at parent stage
TAU_LAT  = 0.70  # lateral edge threshold
W_LEAF   = 0.7   # composite score weight: leaf similarity
W_PARENT = 0.2   # composite score weight: parent similarity
W_META   = 0.1   # composite score weight: meta similarity

EDAG_GRAPHS_DIR = Path(__file__).parent.parent / "edag_graphs"


# ── Duck-type for Qdrant ScoredPoint ─────────────────────────────────────────

@dataclass
class EDAGScoredPoint:
    """
    Mimics qdrant_client.models.ScoredPoint so it passes through
    generation_service.build_sources() and search_service post-processing
    without any special casing.
    """
    score: float
    payload: dict = field(default_factory=dict)


# ── Retriever ─────────────────────────────────────────────────────────────────

class EDAGRetriever:
    def __init__(self, subject_id: str, graph_dir: Path) -> None:
        self.subject_id = subject_id
        self._graph_dir = graph_dir
        self._loaded = False

        # Populated on load
        self._leaf_matrix: Optional[np.ndarray] = None      # (N, D) float16
        self._parent_centroids: Optional[np.ndarray] = None  # (KP, D) float16
        self._meta_centroids: Optional[np.ndarray] = None    # (KM, D) float16
        self._parent_of_leaf: Optional[list[int]] = None     # len N
        self._meta_of_parent: Optional[list[int]] = None     # len KP
        self._lateral_edges: dict[int, list[int]] = {}
        self._leaf_edges: dict[int, list[int]] = {}
        self._leaf_meta: list[dict] = []
        self._children_of_parent: dict[int, list[int]] = {}  # parent_id → [leaf indices]

    def _load(self) -> None:
        if self._loaded:
            return

        logger.info("Loading EDAG graph for subject=%s from %s", self.subject_id, self._graph_dir)

        self._leaf_matrix      = np.load(str(self._graph_dir / "leaves.npy"))  # float16
        self._parent_centroids = np.load(str(self._graph_dir / "parents.npy")).astype(np.float16)
        self._meta_centroids   = np.load(str(self._graph_dir / "metas.npy")).astype(np.float16)

        with open(self._graph_dir / "graph.json", "r", encoding="utf-8") as f:
            graph = json.load(f)

        self._parent_of_leaf  = graph["parent_of_leaf"]
        self._meta_of_parent  = graph["meta_of_parent"]
        self._leaf_meta       = graph["leaf_meta"]

        # Convert string-keyed edges back to int-keyed
        self._lateral_edges = {
            int(k): v for k, v in graph.get("lateral_edges", {}).items()
        }
        self._leaf_edges = {
            int(k): v for k, v in graph.get("leaf_edges", {}).items()
        }

        # Build children_of_parent index
        self._children_of_parent = {}
        for leaf_idx, parent_id in enumerate(self._parent_of_leaf):
            self._children_of_parent.setdefault(parent_id, []).append(leaf_idx)

        self._loaded = True
        logger.info(
            "EDAG graph loaded — leaves=%d parents=%d metas=%d",
            len(self._leaf_meta), len(self._parent_centroids), len(self._meta_centroids)
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[EDAGScoredPoint]:
        """
        4-stage hierarchical beam search.

        Args:
            query_embedding: unit-norm float32 vector, shape (D,)
            top_k: number of results to return

        Returns:
            List of EDAGScoredPoint sorted by composite score descending.
        """
        self._load()

        q = query_embedding.astype(np.float32)

        # ── Stage 1: Meta scoring ─────────────────────────────────────────────
        meta_scores: np.ndarray = self._meta_centroids @ q  # shape (KM,)
        top_meta_ids: list[int] = np.argsort(meta_scores)[::-1][:TM].tolist()

        candidate_parent_ids: set[int] = set()
        for m in top_meta_ids:
            candidate_parent_ids |= {
                j for j, lbl in enumerate(self._meta_of_parent) if lbl == m
            }

        if not candidate_parent_ids:
            logger.warning("EDAG Stage 1 returned no candidate parents for subject=%s", self.subject_id)
            return []

        # ── Stage 2: Parent beam search ───────────────────────────────────────
        p_scores: dict[int, float] = {
            j: float(self._parent_centroids[j] @ q)
            for j in candidate_parent_ids
        }
        best_p = max(p_scores.values())
        beam: set[int] = {j for j, s in p_scores.items() if s >= best_p - DELTA}
        beam_initial = list(beam)

        # ── Stage 3: Lateral expansion ────────────────────────────────────────
        to_add: set[int] = set()
        for j in beam:
            for neighbor in self._lateral_edges.get(j, []):
                if neighbor not in beam:
                    score = float(self._parent_centroids[neighbor] @ q)
                    if score >= TAU_LAT:
                        to_add.add(neighbor)
                        p_scores[neighbor] = score
        beam |= to_add

        # ── Stage 4: Weighted leaf scoring + leaf-edge crawl ──────────────────
        candidate_leaves: list[int] = []
        for j in beam:
            candidate_leaves += self._children_of_parent.get(j, [])

        results: list[tuple[int, float]] = []
        visited: set[int] = set()

        for i in candidate_leaves:
            if i in visited:
                continue
            visited.add(i)

            leaf_vec = self._leaf_matrix[i].astype(np.float32)
            leaf_sim = float(leaf_vec @ q)
            parent_id = self._parent_of_leaf[i]
            meta_id   = self._meta_of_parent[parent_id]

            score = (
                W_LEAF   * leaf_sim
                + W_PARENT * p_scores.get(parent_id, 0.0)
                + W_META   * float(meta_scores[meta_id])
            )
            results.append((i, score))

            # Crawl leaf edges for boundary recovery
            for neighbor in self._leaf_edges.get(i, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_vec = self._leaf_matrix[neighbor].astype(np.float32)
                    n_sim = float(n_vec @ q)
                    n_parent_id = self._parent_of_leaf[neighbor]
                    n_meta_id   = self._meta_of_parent[n_parent_id]
                    n_score = (
                        W_LEAF   * n_sim
                        + W_PARENT * p_scores.get(n_parent_id, 0.0)
                        + W_META   * float(meta_scores[n_meta_id])
                    )
                    results.append((neighbor, n_score))

        # Sort descending by composite score, take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]

        # Build EDAGScoredPoint list (duck-types Qdrant ScoredPoint)
        output: list[EDAGScoredPoint] = []
        for leaf_idx, score in top:
            meta = self._leaf_meta[leaf_idx]
            output.append(EDAGScoredPoint(
                score=round(score, 6),
                payload={
                    "text_block_id": meta.get("chunk_id", ""),
                    "document_id":   meta.get("document_id", ""),
                    "page_number":   meta.get("page", 0),
                    "direction":     meta.get("direction", "ltr"),
                    "bbox":          meta.get("bbox", []),
                    "filename":      meta.get("filename", ""),
                    "text":          meta.get("text", ""),
                },
            ))

        logger.info(
            "EDAG search — subject=%s beam_parents=%d candidates=%d returned=%d",
            self.subject_id, len(beam), len(candidate_leaves), len(output)
        )
        
        trace = {
            "meta_ids": top_meta_ids,
            "beam_parent_ids": beam_initial,
            "lateral_parent_ids": list(to_add),
        }
        return output, trace

    async def search_async(self, query_embedding: np.ndarray, top_k: int = 10) -> tuple[list[EDAGScoredPoint], dict]:
        """Run search in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _search_executor, 
            lambda: self.search(query_embedding, top_k)
        )


# ── LRU Cache for Retrievers ──────────────────────────────────────────────────

class RetrieverCache:
    def __init__(self, max_size: int = 5) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, EDAGRetriever] = OrderedDict()

    def get(self, subject_id: str) -> EDAGRetriever:
        if subject_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(subject_id)
            return self._cache[subject_id]
        
        # Load new
        graph_dir = EDAG_GRAPHS_DIR / subject_id
        retriever = EDAGRetriever(subject_id, graph_dir)
        
        self._cache[subject_id] = retriever
        if len(self._cache) > self.max_size:
            # Remove oldest (least recently used)
            oldest_id, _ = self._cache.popitem(last=False)
            logger.info("Evicted subject=%s from EDAG cache", oldest_id)
            
        return retriever

    def invalidate(self, subject_id: str) -> None:
        if subject_id in self._cache:
            del self._cache[subject_id]
            logger.info("Invalidated EDAG cache for subject=%s", subject_id)


_retriever_cache = RetrieverCache(max_size=5)


def get_or_load_edag_retriever(subject_id: str) -> EDAGRetriever:
    return _retriever_cache.get(subject_id)


def invalidate_edag_cache(subject_id: str) -> None:
    _retriever_cache.invalidate(subject_id)

