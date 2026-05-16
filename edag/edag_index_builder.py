"""
EDAG Index Builder
------------------
Standalone Kafka consumer that builds the Edge-Directed Acyclic Graph (EDAG)
hierarchical index for a subject when triggered.

Run with:
    python -m edag.edag_index_builder

Design decisions:
- Vectors are fetched from Qdrant (scroll with_vectors=True) instead of
  re-embedding from scratch. This guarantees exact alignment with the flat
  Qdrant search index and avoids expensive model inference.
- The builder is idempotent: re-running for the same subject_id overwrites
  old graph files and resets edag_status to 'ready'.
- Always publishes to edag.build.completed — never leaves the topic silent.
- Uses asyncpg directly (same pattern as embedding_worker.py) for DB updates.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg
import numpy as np
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────

DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://rag_user:rag_password@localhost:5432/rag_db"
)
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
KAFKA_BOOTSTRAP: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9094")
COLLECTION_NAME = "rag_text_blocks"
EDAG_GRAPHS_DIR = Path(__file__).parent.parent / "edag_graphs"

KAFKA_BUILD_TOPIC = "edag.build.requested"
KAFKA_COMPLETED_TOPIC = "edag.build.completed"

# Fixed system parameters (do not change)
KP = 100         # number of parent clusters
KM = 10          # number of meta clusters
TAU_LAT = 0.70   # lateral edge threshold (parent ↔ parent)
TAU_LEAF = 0.75  # leaf-to-leaf edge threshold
MIN_CHUNKS = 100  # minimum chunks before build is triggered

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("edag_builder")

_shutdown = False


def _db_url_to_asyncpg(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


# ── Vector fetching from Qdrant ───────────────────────────────────────────────

def _fetch_vectors_from_qdrant(
    qdrant: QdrantClient, subject_id: str
) -> tuple[np.ndarray, list[dict]]:
    """
    Scroll all stored vectors for a subject from Qdrant.
    Returns (leaf_matrix float32, leaf_meta list).
    Vectors come directly from Qdrant — no re-embedding needed.
    """
    vectors = []
    leaf_meta = []
    offset = None

    logger.info("Scrolling Qdrant for subject_id=%s ...", subject_id)

    while True:
        result, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="subject_id", match=MatchValue(value=subject_id))]
            ),
            limit=256,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )

        for point in result:
            vec = point.vector
            if vec is None:
                continue
            vectors.append(vec)
            p = point.payload or {}
            leaf_meta.append({
                "chunk_id":    p.get("text_block_id", str(point.id)),
                "page":        p.get("page_number", 0),
                "bbox":        p.get("bbox", []),
                "direction":   p.get("direction", "ltr"),
                "document_id": p.get("document_id", ""),
                "filename":    p.get("filename", ""),
                "text":        p.get("text", ""),
            })

        if next_offset is None:
            break
        offset = next_offset

    if not vectors:
        return np.empty((0, 1024), dtype=np.float32), []

    leaf_matrix = np.array(vectors, dtype=np.float32)
    logger.info("Fetched %d vectors from Qdrant for subject=%s", len(vectors), subject_id)
    return leaf_matrix, leaf_meta


# ── Core build logic ──────────────────────────────────────────────────────────

def _build_edag_sync(subject_id: str) -> dict:
    """
    Synchronous build wrapper. Decides between full and incremental build.
    """
    qdrant = QdrantClient(url=QDRANT_URL, timeout=120)

    # 1. Fetch current vectors from Qdrant
    leaf_matrix_f32, leaf_meta = _fetch_vectors_from_qdrant(qdrant, subject_id)
    N = len(leaf_meta)
    graph_dir = EDAG_GRAPHS_DIR / subject_id
    graph_file = graph_dir / "graph.json"
    has_graph = graph_file.exists()

    if has_graph:
        if N < MIN_CHUNKS:
            logger.warning(
                "Subject %s chunk count dropped to %d (< %d). Deleting existing graph.",
                subject_id, N, MIN_CHUNKS
            )
            shutil.rmtree(graph_dir, ignore_errors=True)
            return {"status": "removed_insufficient_chunks", "leaf_count": N}
        
        logger.info("Subject %s already has a graph. Performing INCREMENTAL update.", subject_id)
        return _incremental_build_logic(subject_id, leaf_matrix_f32, leaf_meta)
    else:
        if N < MIN_CHUNKS:
            logger.warning(
                "Subject %s has only %d chunks (< %d). Skipping initial build.",
                subject_id, N, MIN_CHUNKS
            )
            return {"status": "skipped_insufficient_chunks", "leaf_count": N}
        
        logger.info("Subject %s is new or graph missing. Performing FULL rebuild.", subject_id)
        return _full_build_logic(subject_id, leaf_matrix_f32, leaf_meta)


def _full_build_logic(subject_id: str, leaf_matrix_f32: np.ndarray, leaf_meta: list[dict]) -> dict:
    """Original full build logic using KMeans."""
    N = len(leaf_meta)
    leaf_matrix_f16 = leaf_matrix_f32.astype(np.float16)

    # 1. Leaf-to-leaf edges
    logger.info("Full Build: Computing leaf-to-leaf edges (N=%d)...", N)
    leaf_edges: dict[int, list[int]] = {}
    block_size = 256
    for i in range(N):
        vi = leaf_matrix_f32[i]
        for j_start in range(i + 1, N, block_size):
            j_end = min(j_start + block_size, N)
            block = leaf_matrix_f32[j_start:j_end]
            sims = block @ vi
            for k, sim in enumerate(sims):
                j = j_start + k
                if float(sim) >= TAU_LEAF:
                    leaf_edges.setdefault(i, []).append(j)
                    leaf_edges.setdefault(j, []).append(i)

    # 2. Cluster into parents
    actual_kp = min(KP, N)
    logger.info("Full Build: Clustering %d parents...", actual_kp)
    kmeans_p = MiniBatchKMeans(n_clusters=actual_kp, random_state=42, n_init=3)
    parent_labels = kmeans_p.fit_predict(leaf_matrix_f32).tolist()
    parent_centroids = normalize(kmeans_p.cluster_centers_.astype(np.float32), norm="l2").astype(np.float16)

    # 3. Lateral edges
    parent_sims = parent_centroids.astype(np.float32) @ parent_centroids.astype(np.float32).T
    lateral_edges = {}
    for i in range(actual_kp):
        for j in range(i + 1, actual_kp):
            if float(parent_sims[i, j]) >= TAU_LAT:
                lateral_edges.setdefault(i, []).append(j)
                lateral_edges.setdefault(j, []).append(i)

    # 4. Cluster into meta nodes
    actual_km = min(KM, actual_kp)
    logger.info("Full Build: Clustering %d meta nodes...", actual_km)
    kmeans_m = MiniBatchKMeans(n_clusters=actual_km, random_state=42, n_init=3)
    meta_labels = kmeans_m.fit_predict(parent_centroids.astype(np.float32)).tolist()
    meta_centroids = normalize(kmeans_m.cluster_centers_.astype(np.float32), norm="l2").astype(np.float16)

    # Save
    graph_dir = EDAG_GRAPHS_DIR / subject_id
    graph_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(graph_dir / "leaves.npy"), leaf_matrix_f16)
    np.save(str(graph_dir / "parents.npy"), parent_centroids)
    np.save(str(graph_dir / "metas.npy"), meta_centroids)

    graph_data = {
        "parent_of_leaf": parent_labels,
        "meta_of_parent": meta_labels,
        "lateral_edges":  {str(k): v for k, v in lateral_edges.items()},
        "leaf_edges":     {str(k): v for k, v in leaf_edges.items()},
        "leaf_meta":      leaf_meta,
    }
    with open(graph_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False)

    return {"status": "success", "leaf_count": N}


def _incremental_build_logic(subject_id: str, leaf_matrix_f32: np.ndarray, leaf_meta: list[dict]) -> dict:
    """
    Incremental build logic: assigns new chunks to nearest parents and 
    updates centroids without full re-clustering.
    """
    graph_dir = EDAG_GRAPHS_DIR / subject_id
    with open(graph_dir / "graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)
    
    parent_centroids = np.load(str(graph_dir / "parents.npy")).astype(np.float32)
    meta_centroids = np.load(str(graph_dir / "metas.npy")).astype(np.float32)
    
    existing_ids = {m["chunk_id"] for m in graph["leaf_meta"]}
    new_indices = [i for i, m in enumerate(leaf_meta) if m["chunk_id"] not in existing_ids]
    
    if not new_indices:
        logger.info("Incremental: No new chunks found for %s.", subject_id)
        return {"status": "success", "leaf_count": len(leaf_meta), "note": "no new chunks"}

    logger.info("Incremental: Adding %d new chunks to graph %s", len(new_indices), subject_id)

    parent_of_leaf = graph["parent_of_leaf"]
    meta_of_parent = graph["meta_of_parent"]
    
    # 1. Assign new chunks to nearest existing parents
    for i in new_indices:
        vec = leaf_matrix_f32[i]
        # Cosine similarity to all parent centroids
        scores = parent_centroids @ vec
        p_id = int(np.argmax(scores))
        
        # Incremental centroid update (weighted mean)
        count = parent_of_leaf.count(p_id)
        parent_centroids[p_id] = (parent_centroids[p_id] * count + vec) / (count + 1)
        parent_centroids[p_id] /= (np.linalg.norm(parent_centroids[p_id]) + 1e-9)
        
        parent_of_leaf.append(p_id)

    # 2. Re-calculate meta centroids from updated parent centroids
    for m_id in range(len(meta_centroids)):
        child_parents = [j for j, m in enumerate(meta_of_parent) if m == m_id]
        if child_parents:
            new_meta_vec = np.mean(parent_centroids[child_parents], axis=0)
            meta_centroids[m_id] = new_meta_vec / (np.linalg.norm(new_meta_vec) + 1e-9)

    # 3. Re-calculate lateral edges (parent similarity changed)
    parent_sims = parent_centroids @ parent_centroids.T
    lateral_edges = {}
    actual_kp = len(parent_centroids)
    for i in range(actual_kp):
        for j in range(i + 1, actual_kp):
            if float(parent_sims[i, j]) >= TAU_LAT:
                lateral_edges.setdefault(i, []).append(j)
                lateral_edges.setdefault(j, []).append(i)

    # 4. Update leaf edges (only for new leaves + their neighbors)
    leaf_edges = {int(k): v for k, v in graph.get("leaf_edges", {}).items()}
    for i in new_indices:
        vi = leaf_matrix_f32[i]
        sims = leaf_matrix_f32 @ vi
        for j, sim in enumerate(sims):
            if i != j and float(sim) >= TAU_LEAF:
                leaf_edges.setdefault(i, []).append(j)
                leaf_edges.setdefault(j, []).append(i)

    # Save
    np.save(str(graph_dir / "leaves.npy"), leaf_matrix_f32.astype(np.float16))
    np.save(str(graph_dir / "parents.npy"), parent_centroids.astype(np.float16))
    np.save(str(graph_dir / "metas.npy"), meta_centroids.astype(np.float16))
    
    new_graph_data = {
        "parent_of_leaf": parent_of_leaf,
        "meta_of_parent": meta_of_parent,
        "lateral_edges": {str(k): v for k, v in lateral_edges.items()},
        "leaf_edges": {str(k): v for k, v in leaf_edges.items()},
        "leaf_meta": leaf_meta,
    }
    with open(graph_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump(new_graph_data, f, ensure_ascii=False)
        
    return {"status": "success", "leaf_count": len(leaf_meta), "incremental": True}


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _update_subject_status(
    pool: asyncpg.Pool,
    subject_id: str,
    status: str,
    leaf_count: Optional[int] = None,
) -> None:
    if leaf_count is not None:
        await pool.execute(
            """
            UPDATE subjects
            SET edag_status    = $1,
                edag_built_at  = now(),
                edag_leaf_count = $2
            WHERE id = $3::uuid
            """,
            status, leaf_count, subject_id,
        )
    else:
        await pool.execute(
            "UPDATE subjects SET edag_status = $1 WHERE id = $2::uuid",
            status, subject_id,
        )
    logger.info("subjects.edag_status \u2192 '%s' for %s", status, subject_id)


# ── Kafka publish helper ──────────────────────────────────────────────────────

async def _publish_completed(
    producer: AIOKafkaProducer,
    subject_id: str,
    status: str,
    leaf_count: int = 0,
    error: Optional[str] = None,
) -> None:
    payload = {
        "subject_id": subject_id,
        "leaf_count": leaf_count,
        "built_at":   datetime.utcnow().isoformat(),
        "status":     status,
        "error":      error,
    }
    try:
        await producer.send(KAFKA_COMPLETED_TOPIC, payload)
        logger.info("Published to %s: %s", KAFKA_COMPLETED_TOPIC, payload)
    except Exception as exc:
        logger.error("Failed to publish to %s: %s", KAFKA_COMPLETED_TOPIC, exc)


# ── Main consumer loop ────────────────────────────────────────────────────────

async def run_builder() -> None:
    global _shutdown

    pool = await asyncpg.create_pool(
        dsn=_db_url_to_asyncpg(DATABASE_URL), min_size=1, max_size=3
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    await producer.start()

    consumer = AIOKafkaConsumer(
        KAFKA_BUILD_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="edag-builder",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode()),
    )

    for attempt in range(5):
        try:
            await consumer.start()
            logger.info("EDAG builder listening on '%s'", KAFKA_BUILD_TOPIC)
            break
        except Exception as exc:
            logger.warning("Kafka connect attempt %d failed: %s", attempt + 1, exc)
            await asyncio.sleep(5)
    else:
        logger.error("Could not connect to Kafka after 5 attempts. Exiting.")
        await producer.stop()
        await pool.close()
        return

    loop = asyncio.get_running_loop()

    try:
        async for msg in consumer:
            if _shutdown:
                break

            data = msg.value
            subject_id = data.get("subject_id", "")
            user_id = data.get("user_id", "")
            logger.info(
                "Build requested \u2014 subject=%s user=%s", subject_id, user_id
            )

            try:
                # Run CPU-bound build in a thread executor
                result = await loop.run_in_executor(
                    None, _build_edag_sync, subject_id
                )
                status = result["status"]
                leaf_count = result.get("leaf_count", 0)

                if status == "success":
                    await _update_subject_status(pool, subject_id, "ready", leaf_count)
                elif status in ("skipped_insufficient_chunks", "removed_insufficient_chunks"):
                    await _update_subject_status(pool, subject_id, "none", leaf_count)
                else:
                    await _update_subject_status(pool, subject_id, "failed")

                await _publish_completed(
                    producer, subject_id, status, leaf_count
                )

            except Exception as exc:
                logger.exception(
                    "Build failed for subject=%s: %s", subject_id, exc
                )
                try:
                    await _update_subject_status(pool, subject_id, "failed")
                except Exception:
                    pass
                await _publish_completed(
                    producer, subject_id, "failed", error=str(exc)
                )

    finally:
        await consumer.stop()
        await producer.stop()
        await pool.close()
        logger.info("EDAG builder shut down.")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    global _shutdown

    def _handle_signal(signum, frame):
        global _shutdown
        logger.info("Signal %s received \u2014 shutting down.", signum)
        _shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    await run_builder()


if __name__ == "__main__":
    asyncio.run(main())
