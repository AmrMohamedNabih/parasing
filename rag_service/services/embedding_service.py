"""
Embedding Service
-----------------
Singleton wrapper around intfloat/multilingual-e5-small.
encode_query() runs in a ThreadPoolExecutor to avoid blocking the event loop.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed_svc")


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    async def load(self) -> None:
        logger.info("Loading SentenceTransformer: %s", _MODEL_NAME)
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            _executor, lambda: SentenceTransformer(_MODEL_NAME)
        )
        logger.info("Embedding model ready.")

    async def encode_query(self, text: str) -> list[float]:
        """
        Encode a user question using the web_search_query prompt.
        Returns a normalised float list ready for Qdrant.
        """
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(
            _executor,
            partial(
                self._model.encode,
                text,
                prompt_name="web_search_query",
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
        )
        return vec.tolist()


embedding_service = EmbeddingService()
