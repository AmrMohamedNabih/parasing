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
    max_output_tokens=2048,
)

def _system_prompt(language: str, deep_analysis: bool = False, task_plan: bool = False, has_context: bool = True, mindmap_mode: bool = False, notebook_mode: bool = False) -> str:
    if deep_analysis:
        base = (
            "You are a highly analytical research expert. Your task is to provide "
            "a deep, comprehensive analysis of the provided context. \n\n"
            "CRITICAL GUIDELINES:\n"
            "1. DO NOT refer to passages by their labels (e.g., 'Passage 1' or 'Context Block'). "
            "2. Focus on key findings, methodologies, and conclusions. "
            "3. Structure your response logically with professional headers, bullet points, and "
            "markdown tables or structured diagrams/comparisons where appropriate to organize and describe details more clearly.\n"
            "4. Reason through the entire context step-by-step to ensure no detail is missed."
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
            "You are an expert at creating Excalidraw whiteboards.\n"
            "Based on the provided context, generate a valid Excalidraw JSON structure representing a mindmap of the concepts discussed.\n"
            "Your output MUST be raw JSON only, without any markdown formatting like ```json.\n"
            "The JSON must have the following structure:\n"
            '{"type": "excalidraw", "version": 2, "source": "rag", "elements": [...], "appState": {"viewBackgroundColor": "transparent", "theme": "light"}}\n\n'
            "CRITICAL DESIGN AND LAYOUT RULES:\n"
            "1. Box Sizing: For every rectangle/ellipse containing text, the box width and height must match the text length. "
            "Calculate width = (character_count * 9.5) + 40 (minimum 160px). Calculate height = (line_count * 20) + 30 (minimum 60px).\n"
            "2. Text Centering: The corresponding text element MUST have its center coordinates aligned exactly with the shape's center:\n"
            "   text.x = shape.x + (shape.width - text.width) / 2\n"
            "   text.y = shape.y + (shape.height - text.height) / 2\n"
            "3. Overlap Prevention: Space elements out widely. Spacing must be at least 300px horizontally and 200px vertically between shapes. "
            "Place the central concept at (100, 100) and branch out radially or in a hierarchical tree.\n"
            "4. Arrow Connections: Arrows connecting shapes must start and end at the exact boundaries of the shapes, not the centers. "
            "For Box A (x1, y1) and Box B (x2, y2), an arrow from A to B must start at A's edge and end at B's edge. "
            "Use elements of type 'arrow' with defined start/end bindings or matching coordinates, and keep lines clean and straight."
        )
    elif notebook_mode:
        base = (
            "You are an Expert academic note-taker. "
            "Your goal is to create comprehensive, well-structured, and highly readable study notes "
            "based on the provided document context and the user's specific request.\n\n"
            "Requirements:\n"
            "1. Focus the notes on the themes and topics requested by the user.\n"
            "2. Use clear headings and sub-headings (Markdown format).\n"
            "3. Use bullet points for key concepts, definitions, and important details.\n"
            "4. Ensure the notes are logical and easy to study from.\n"
            "5. Do NOT include phrases like 'Based on the context' or 'The document says'.\n"
            "6. Incorporate specific details and examples from the provided context blocks.\n"
            "7. Focus only on the educational content."
        )
    else:
        if has_context:
            base = (
                "You are an intelligent and professional research assistant. "
                "Your task is to provide accurate, natural-sounding answers based ONLY on the "
                "provided context passages. \n\n"
                "CRITICAL GUIDELINES:\n"
                "1. DO NOT mention labels like '[CONTEXT {i}]', 'Passage 1', or 'Source 1' in your response. "
                "2. DO NOT say things like 'According to Passage 1...' or 'The context mentions...'. "
                "3. Answer the question directly and naturally as if you are the expert providing the information. "
                "4. If the answer cannot be found in the context, say so clearly. Do not hallucinate.\n"
                "5. Analyze all provided context carefully and reason through the information before responding."
            )
        else:
            base = (
                "You are a helpful research assistant. You are currently in a general chat mode "
                "without specific document context. Answer the user's question to the best of your "
                "knowledge using the conversation history if available."
            )
    
    summary_instruction = ""
    if not mindmap_mode and not notebook_mode:
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
    from datetime import datetime
    summary_context = f"\n\n[Background Context: Summary of previous points in this conversation]\n{summary}\n[End of Background Context]" if summary else ""
    
    passages = []
    for i, pt in enumerate(scored_points, 1):
        p = pt.payload
        passages.append(
            f"[CONTEXT BLOCK {i}] (Document: {p.get('filename', 'Unknown')}, page {p.get('page_number', '?')})\n"
            f"{p.get('text', '')}"
        )
    
    analysis_context = "\n\n(Note: This is a deep analysis request. Please provide a thorough breakdown.)" if deep_analysis else ""
    current_time = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    time_context = f"\n[Current Time: {current_time}]"
    
    return (
        "Context passages:\n\n"
        + "\n\n".join(passages)
        + f"\n\n---\n{summary_context}{time_context}\nQuestion: {question}{analysis_context}\n\nAnswer:"
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
        notebook_mode: bool = False,
    ) -> AsyncGenerator[str, None]:
        if majority_rtl:
            language = "ar"

        has_context = len(scored_points) > 0
        system_msg = _system_prompt(language, deep_analysis, task_plan, has_context, mindmap_mode, notebook_mode)
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

        if mindmap_mode or notebook_mode or task_plan:
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
