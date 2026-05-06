"""
Generation Service
------------------
Streams answers from Gemini or OpenAI using a thread-bridge queue pattern.
"""

import asyncio
import logging
import threading
import time
from typing import AsyncGenerator

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from mistralai.client import Mistral
from openai import OpenAI

from rag_service.config import settings
from rag_service.schemas import SourceBlock

logger = logging.getLogger(__name__)

_GENERATION_CONFIG_GEMINI = genai.types.GenerationConfig(
    temperature=0.2,
    max_output_tokens=1024,
)

def _system_prompt(language: str, deep_analysis: bool = False, task_plan: bool = False, has_context: bool = True, mindmap_mode: bool = False) -> str:
    if deep_analysis:
        base = (
            "You are a highly analytical research assistant. Your task is to provide "
            "a deep, comprehensive analysis and summary of the provided context passages. "
            "Focus on key findings, methodologies, conclusions, and any significant takeaways. "
            "Structure your response logically with headers and bullet points where appropriate. "
            "Base your answer ONLY on the provided context."
        )
    elif task_plan:
        base = (
            "You are an expert Study and Project Planner. Your task is to analyze the "
            "provided context and create a detailed execution plan (tasks). "
            "You MUST provide two parts in your response:\n"
            "1. A natural language summary of the plan for the user.\n"
            "2. A structured JSON list of tasks wrapped inside <task_plan> tags.\n\n"
            "Example format for the JSON part:\n"
            "<task_plan>\n"
            "[\n"
            "  {\"title\": \"Read Chapter 1\", \"description\": \"Focus on pages 1-10\", \"dueDate\": \"2024-05-10T10:00:00\", \"priority\": \"HIGH\"},\n"
            "  {\"title\": \"Draft Summary\", \"description\": \"Summarize key findings\", \"dueDate\": \"2024-05-11T15:00:00\", \"priority\": \"MEDIUM\"}\n"
            "]\n"
            "</task_plan>\n\n"
            "Available Priorities: HIGH, MEDIUM, LOW.\n"
            "Use ISO 8601 format for dueDate. If no specific date is mentioned, spread them out starting from tomorrow."
        )
    elif mindmap_mode:
        base = (
            "You are an expert at creating Excalidraw whiteboards. "
            "Based on the provided context, generate a valid Excalidraw JSON structure "
            "representing a mindmap or flowchart of the concepts discussed. "
            "Your output MUST be raw JSON only, without any markdown formatting like ```json. "
            "The JSON must have the following structure: "
            '{"type": "excalidraw", "version": 2, "source": "rag", "elements": [...], "appState": {...}} '
            "You can use shapes like 'rectangle', 'diamond', 'ellipse', 'text', and 'arrow' to connect them. "
            "Ensure the elements are visually spaced out so they don't overlap."
        )
    else:
        if has_context:
            base = (
                "You are a helpful assistant answering questions based ONLY on the "
                "provided context passages. If the answer cannot be found in the "
                "context, say so clearly. Do not hallucinate. "
                "Do NOT include a 'Relevant passages:' section or header at the end."
            )
        else:
            base = (
                "You are a helpful research assistant. You are currently in a general chat mode "
                "without specific document context. Answer the user's question to the best of your "
                "knowledge using the conversation history if available."
            )
    
    summary_instruction = ""
    if not mindmap_mode:
        summary_instruction = (
            "\n\nAt the end of your response, you MUST provide a single bullet point "
            "summarizing this specific interaction (max 20 words). Wrap it in <summary_point> tags. "
            "Example: <summary_point>- User asked about X and AI explained Y.</summary_point>"
        )

    lang_suffix = {
        "ar":   " Respond in Arabic.",
        "en":   " Respond in English.",
        "auto": " Respond in the same language as the question.",
    }
    return base + summary_instruction + lang_suffix.get(language, lang_suffix["auto"])


def _user_prompt(question: str, scored_points: list, deep_analysis: bool = False, summary: str = None) -> str:
    summary_context = f"\n\n[Background Context: Summary of previous points in this conversation]\n{summary}\n[End of Background Context]" if summary else ""
    
    passages = []
    for i, pt in enumerate(scored_points, 1):
        p = pt.payload
        passages.append(
            f"[PASSAGE {i}] (Document: {p.get('filename', 'Unknown')}, page {p.get('page_number', '?')}, "
            f"score {pt.score:.2f})\n{p.get('text', '')}"
        )
    
    analysis_context = "\n\n(Note: This is a deep analysis request. Please provide a thorough breakdown.)" if deep_analysis else ""
    
    return (
        "Context passages:\n\n"
        + "\n\n".join(passages)
        + f"\n\n---\n{summary_context}\nQuestion: {question}{analysis_context}\n\nAnswer:"
    )


class GenerationService:
    def __init__(self) -> None:
        self._configured = False
        self._openai_client = None
        self._mistral_client = None

    def configure(self) -> None:
        if settings.LLM_PROVIDER == "gemini":
            if settings.GEMINI_API_KEY:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._configured = True
                logger.info("Gemini configured — model: %s", settings.GEMINI_MODEL)
            else:
                logger.warning("GEMINI_API_KEY is empty.")
        elif settings.LLM_PROVIDER == "openai":
            if settings.OPENAI_API_KEY:
                self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                self._configured = True
                logger.info("OpenAI configured — model: %s", settings.OPENAI_MODEL)
            else:
                logger.warning("OPENAI_API_KEY is empty.")
        elif settings.LLM_PROVIDER == "mistral":
            if settings.MISTRAL_API_KEY:
                self._mistral_client = Mistral(api_key=settings.MISTRAL_API_KEY)
                self._configured = True
                logger.info("Mistral configured — model: %s", settings.MISTRAL_MODEL)
            else:
                logger.warning("MISTRAL_API_KEY is empty.")

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
        deep_analysis: bool = False,
        task_plan: bool = False,
        summary: str = None,
        mindmap_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        if majority_rtl:
            language = "ar"

        has_context = len(scored_points) > 0
        system_msg = _system_prompt(language, deep_analysis, task_plan, has_context, mindmap_mode)
        user_msg = _user_prompt(question, scored_points, deep_analysis, summary)
        
        # ── INTERACTION LOGGING ──────────────────────────────────────────────
        interaction_log = (
            "\n" + "="*80 + "\n"
            "PROMPT SENT TO LLM\n"
            "-"*80 + "\n"
            f"SYSTEM: {system_msg}\n\n"
            f"USER: {user_msg}\n"
            "="*80 + "\n"
        )
        logger.info(interaction_log)
        # ────────────────────────────────────────────────────────────────────

        if mindmap_mode:
            max_tokens = 8192
        else:
            max_tokens = 2048 if deep_analysis else 1024

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run_stream() -> None:
            if settings.LLM_PROVIDER == "gemini":
                self._run_gemini_stream(system_msg + "\n\n" + user_msg, queue, loop, max_tokens, mindmap_mode)
            elif settings.LLM_PROVIDER == "mistral":
                self._run_mistral_stream(system_msg, user_msg, queue, loop, max_tokens, mindmap_mode)
            else:
                self._run_openai_stream(system_msg, user_msg, queue, loop, max_tokens, mindmap_mode)

        thread = threading.Thread(target=_run_stream, daemon=True)
        thread.start()

        full_answer = []
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            full_answer.append(chunk)
            yield chunk

        # Log final answer
        logger.info("\n" + "-"*80 + "\nAI ANSWER:\n" + "".join(full_answer) + "\n" + "="*80 + "\n")

    async def generate_updated_summary(self, history_text: str) -> str:
        """
        Generates an updated bullet-point summary of the conversation.
        """
        if not self._configured:
            return None

        prompt = (
            "You are a helpful assistant that maintains a concise conversation history. "
            "Based on the following recent exchange, provide an updated "
            "summary in bullet points. Keep it to the most important points only (max 5-7 points).\n\n"
            f"Conversation History:\n{history_text}\n\n"
            "Updated Summary (Bullet points):"
        )

        logger.info("Generating updated summary...")
        try:
            if settings.LLM_PROVIDER == "gemini":
                model = genai.GenerativeModel(settings.GEMINI_MODEL)
                response = model.generate_content(prompt)
                logger.info("Gemini raw response: %s", response.text if hasattr(response, 'text') else "NO TEXT")
                new_summary = response.text.strip()
                logger.info("New summary generated: %s", new_summary[:100] + "...")
                return new_summary
            elif settings.LLM_PROVIDER == "openai":
                response = self._openai_client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512
                )
                new_summary = response.choices[0].message.content.strip()
                logger.info("New summary generated: %s", new_summary[:100] + "...")
                return new_summary
            elif settings.LLM_PROVIDER == "mistral":
                response = self._mistral_client.chat.complete(
                    model=settings.MISTRAL_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512
                )
                new_summary = response.choices[0].message.content.strip()
                logger.info("New summary generated: %s", new_summary[:100] + "...")
                return new_summary
            return None
        except Exception as exc:
            logger.error("Summary generation failed: %s", exc)
            return None

    def _run_gemini_stream(self, prompt, queue, loop, max_tokens=1024, is_json=False):
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        kwargs = {
            "temperature": 0.2,
            "max_output_tokens": max_tokens,
        }
        if is_json:
            kwargs["response_mime_type"] = "application/json"
            
        config = genai.types.GenerationConfig(**kwargs)
        try:
            response = model.generate_content(
                prompt, stream=True, generation_config=config
            )
            for chunk in response:
                text = getattr(chunk, "text", None)
                if text:
                    asyncio.run_coroutine_threadsafe(queue.put(text), loop)
        except ResourceExhausted:
            logger.warning("Gemini Quota Exceeded")
            asyncio.run_coroutine_threadsafe(queue.put("\n[Gemini Quota Exceeded. Please try again later.]"), loop)
        except Exception as exc:
            logger.error("Gemini error: %s", exc)
            asyncio.run_coroutine_threadsafe(queue.put(f"\n[Generation error: {exc}]"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    def _run_mistral_stream(self, system_msg, user_msg, queue, loop, max_tokens=1024, is_json=False):
        try:
            response = self._mistral_client.chat.stream(
                model=settings.MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            for chunk in response:
                if chunk.data.choices and chunk.data.choices[0].delta.content:
                    text = chunk.data.choices[0].delta.content
                    if text:
                        asyncio.run_coroutine_threadsafe(queue.put(text), loop)
        except Exception as exc:
            logger.error("Mistral error: %s", exc)
            asyncio.run_coroutine_threadsafe(queue.put(f"\n[Generation error: {exc}]"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    def _run_openai_stream(self, system_msg, user_msg, queue, loop, max_tokens=1024, is_json=False):
        try:
            response = self._openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                stream=True,
                temperature=0.2,
                max_tokens=max_tokens,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    asyncio.run_coroutine_threadsafe(queue.put(text), loop)
        except Exception as exc:
            logger.error("OpenAI error: %s", exc)
            asyncio.run_coroutine_threadsafe(queue.put(f"\n[Generation error: {exc}]"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)


generation_service = GenerationService()
