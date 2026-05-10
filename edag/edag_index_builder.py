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
    Synchronous build — runs in a thread executor to keep the consumer
    event loop free. Returns a result dict with status and leaf_count.
    """
    qdrant = QdrantClient(url=QDRANT_URL, timeout=120)

    # Step 1-2: Fetch vectors directly from Qdrant (no re-embedding)
    leaf_matrix_f32, leaf_meta = _fetch_vectors_from_qdrant(qdrant, subject_id)
    N = len(leaf_meta)

    if N < MIN_CHUNKS:
        logger.warning(
            "Subject %s has only %d chunks (< %d). Skipping build.",
            subject_id, N, MIN_CHUNKS
        )
        return {"status": "skipped_insufficient_chunks", "leaf_count": N}

    # Step 3: Leaf matrix — float16 for storage, float32 for computation
    leaf_matrix_f16 = leaf_matrix_f32.astype(np.float16)

    # Step 4: Leaf-to-leaf edges (use float32 dot products even though storage is f16)
    logger.info("Computing leaf-to-leaf edges (N=%d, TAU_LEAF=%.2f)...", N, TAU_LEAF)
    leaf_edges: dict[int, list[int]] = {}
    # Compute in blocks to avoid O(N²) memory for large N
    block_size = 256
    for i in range(N):
        vi = leaf_matrix_f32[i]
        for j_start in range(i + 1, N, block_size):
            j_end = min(j_start + block_size, N)
            block = leaf_matrix_f32[j_start:j_end]
            sims = block @ vi  # shape (block_size,)
            for k, sim in enumerate(sims):
                j = j_start + k
                if float(sim) >= TAU_LEAF:
                    leaf_edges.setdefault(i, []).append(j)
                    leaf_edges.setdefault(j, []).append(i)

    # Step 5: Cluster into KP parent clusters
    actual_kp = min(KP, N)
    logger.info("Running MiniBatchKMeans(n_clusters=%d) for parents...", actual_kp)
    try:
        kmeans_p = MiniBatchKMeans(
            n_clusters=actual_kp, random_state=42, max_iter=300, n_init=3
        )
        parent_labels: np.ndarray = kmeans_p.fit_predict(leaf_matrix_f32)
    except Exception:
        logger.warning("KMeans (parents) failed on first attempt, retrying max_iter=500...")
        kmeans_p = MiniBatchKMeans(
            n_clusters=actual_kp, random_state=42, max_iter=500, n_init=5
        )
        parent_labels = kmeans_p.fit_predict(leaf_matrix_f32)

    # L2-normalize parent centroids
    parent_centroids: np.ndarray = normalize(
        kmeans_p.cluster_centers_.astype(np.float32), norm="l2"
    ).astype(np.float16)


    # Step 6: Lateral edges between parents
    logger.info("Computing lateral edges between parents (TAU_LAT=%.2f)...", TAU_LAT)
    parent_sims = parent_centroids @ parent_centroids.T  # (KP, KP)
    lateral_edges: dict[int, list[int]] = {}
    for i in range(actual_kp):
        for j in range(i + 1, actual_kp):
            if float(parent_sims[i, j]) >= TAU_LAT:
                lateral_edges.setdefault(i, []).append(j)
                lateral_edges.setdefault(j, []).append(i)

    # Step 7: Cluster into KM meta clusters
    actual_km = min(KM, actual_kp)
    logger.info("Running MiniBatchKMeans(n_clusters=%d) for meta nodes...", actual_km)
    try:
        kmeans_m = MiniBatchKMeans(
            n_clusters=actual_km, random_state=42, max_iter=300, n_init=3
        )
        meta_labels: np.ndarray = kmeans_m.fit_predict(parent_centroids)
    except Exception:
        logger.warning("KMeans (meta) failed on first attempt, retrying max_iter=500...")
        kmeans_m = MiniBatchKMeans(
            n_clusters=actual_km, random_state=42, max_iter=500, n_init=5
        )
        meta_labels = kmeans_m.fit_predict(parent_centroids)

    # L2-normalize meta centroids
    meta_centroids: np.ndarray = normalize(
        kmeans_m.cluster_centers_.astype(np.float32), norm="l2"
    ).astype(np.float16)


    # Step 8: Serialize to disk
    graph_dir = EDAG_GRAPHS_DIR / subject_id
    graph_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(graph_dir / "leaves.npy"), leaf_matrix_f16)
    np.save(str(graph_dir / "parents.npy"), parent_centroids)
    np.save(str(graph_dir / "metas.npy"), meta_centroids)

    graph_data = {
        "parent_of_leaf":  parent_labels.tolist(),
        "meta_of_parent":  meta_labels.tolist(),
        "lateral_edges":   {str(k): v for k, v in lateral_edges.items()},
        "leaf_edges":      {str(k): v for k, v in leaf_edges.items()},
        "leaf_meta":       leaf_meta,
    }
    with open(graph_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False)

    logger.info(
        "EDAG graph written to %s | leaves=%d parents=%d metas=%d",
        graph_dir, N, actual_kp, actual_km
    )
    return {"status": "success", "leaf_count": N}


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
    logger.info("subjects.edag_status → '%s' for %s", status, subject_id)


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
                "Build requested — subject=%s user=%s", subject_id, user_id
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
                elif status == "skipped_insufficient_chunks":
                    await _update_subject_status(pool, subject_id, "none")
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
        logger.info("Signal %s received — shutting down.", signum)
        _shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    await run_builder()


if __name__ == "__main__":
    asyncio.run(main())
