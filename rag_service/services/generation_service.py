"""
Generation Service
------------------
Streams answers from Gemini using a thread-bridge queue pattern.

The entire Gemini SDK iteration (including each next() call on the stream)
runs inside a daemon thread. Text chunks are forwarded to the async caller
via asyncio.run_coroutine_threadsafe → asyncio.Queue, keeping the event
loop completely unblocked between chunks.

Key decisions:
- majority_rtl overrides the language hint to "ar" when >50% of retrieved
  passages have direction='rtl'.
- ResourceExhausted is retried once after 10 s.
- Each chunk.text is accessed with getattr guard to handle finish-reason
  chunks that lack a .text attribute.
- The async generator is always consumed under asyncio.wait_for(timeout=120)
  in the Kafka consumer so a stalled Gemini stream never holds a semaphore slot.
"""

import asyncio
import logging
import threading
import time
from typing import AsyncGenerator

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from rag_service.config import settings
from rag_service.schemas import SourceBlock

logger = logging.getLogger(__name__)

_GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.2,
    max_output_tokens=1024,
)


def _system_prompt(language: str) -> str:
    base = (
        "You are a helpful assistant answering questions based ONLY on the "
        "provided context passages. If the answer cannot be found in the "
        "context, say so clearly. Do not hallucinate. "
        "Cite passage numbers when relevant."
    )
    lang_suffix = {
        "ar":   " Respond in Arabic.",
        "en":   " Respond in English.",
        "auto": " Respond in the same language as the question.",
    }
    return base + lang_suffix.get(language, lang_suffix["auto"])


def _user_prompt(question: str, scored_points: list) -> str:
    passages = []
    for i, pt in enumerate(scored_points, 1):
        p = pt.payload
        passages.append(
            f"[PASSAGE {i}] (page {p.get('page_number', '?')}, "
            f"score {pt.score:.2f})\n{p.get('text', '')}"
        )
    return (
        "Context passages:\n\n"
        + "\n\n".join(passages)
        + f"\n\n---\nQuestion: {question}\n\nAnswer:"
    )


class GenerationService:
    def __init__(self) -> None:
        self._configured = False

    def configure(self) -> None:
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._configured = True
            logger.info("Gemini configured — model: %s", settings.GEMINI_MODEL)
        else:
            logger.warning("GEMINI_API_KEY is empty — generation will not work.")

    def build_sources(self, scored_points: list) -> list[SourceBlock]:
        return [
            SourceBlock(
                text_block_id=p.payload.get("text_block_id", ""),
                document_id=p.payload.get("document_id", ""),
                page_number=p.payload.get("page_number", 0),
                text_snippet=p.payload.get("text", "")[:120],
                score=round(p.score, 4),
            )
            for p in scored_points
        ]

    async def stream_answer(
        self,
        question: str,
        scored_points: list,
        language: str,
        majority_rtl: bool,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator yielding text fragments from Gemini.

        Thread-bridge pattern: the full SDK iteration runs in a daemon thread;
        chunks arrive in an asyncio.Queue read by this async generator.
        """
        if majority_rtl:
            language = "ar"

        prompt = (
            _system_prompt(language)
            + "\n\n"
            + _user_prompt(question, scored_points)
        )

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run_stream() -> None:
            model = genai.GenerativeModel(settings.GEMINI_MODEL)

            def _stream_once() -> None:
                response = model.generate_content(
                    prompt, stream=True, generation_config=_GENERATION_CONFIG
                )
                for chunk in response:
                    text = getattr(chunk, "text", None)   # guard: finish-reason chunks
                    if text:
                        asyncio.run_coroutine_threadsafe(queue.put(text), loop)

            try:
                _stream_once()
            except ResourceExhausted as exc:
                logger.warning("Gemini ResourceExhausted — retrying in 10 s: %s", exc)
                time.sleep(10)
                try:
                    _stream_once()
                except Exception as retry_exc:
                    logger.error("Gemini retry failed: %s", retry_exc)
                    asyncio.run_coroutine_threadsafe(
                        queue.put(f"\n[Generation error: {retry_exc}]"), loop
                    )
            except Exception as exc:
                logger.error("Gemini generation error: %s", exc)
                asyncio.run_coroutine_threadsafe(
                    queue.put(f"\n[Generation error: {exc}]"), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

        thread = threading.Thread(target=_run_stream, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk


generation_service = GenerationService()
