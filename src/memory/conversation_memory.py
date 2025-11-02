"""Utilities for persisting and retrieving long-term conversation memory.

This module implements a lightweight persistence layer that stores
conversation history for the CLI agent.  It provides three core
capabilities required by the user request:

*  **Persistence** – conversations are written to disk so that they can be
   recalled in later CLI sessions.
*  **Retrieval** – when a new objective is issued we surface the most
   relevant historical records to the agent.
*  **Compression** – historical messages are summarised and trimmed to keep
   the on-disk footprint manageable while still preserving important
   context.

The implementation deliberately avoids introducing new dependencies and can
operate without an LLM-backed summariser.  If an LLM is available the
summaries will be richer, otherwise we fall back to heuristic-based
compression.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

try:  # Optional dependency – only used when available.
    from src.utils.llm_manager import get_llm_manager  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    get_llm_manager = None  # type: ignore


@dataclass
class MemoryRecord:
    """A single persisted conversation record."""

    objective: str
    created_at: str
    summary: str
    highlights: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """In-memory view of the retrieved long-term memory."""

    summary: str = ""
    records: List[MemoryRecord] = field(default_factory=list)
    project_state: Dict[str, Any] = field(default_factory=dict)


class ConversationMemoryStore:
    """Disk-backed store for long-term conversation history."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_turns: int = 18,
        summary_max_chars: int = 1200,
        thread_summary_max_chars: int = 4000,
        retrieval_top_k: int = 3,
    ) -> None:
        self._storage_path = Path(storage_path or Path("data/memory/conversations.json"))
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_turns = max(6, max_turns)
        self._summary_max_chars = max(400, summary_max_chars)
        self._thread_summary_max_chars = max(1000, thread_summary_max_chars)
        self._retrieval_top_k = max(1, retrieval_top_k)
        self._lock = threading.Lock()

        self._data: Dict[str, Any] = {"threads": {}}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_context(self, thread_id: str, objective: str) -> MemoryContext:
        """Load the long-term memory context for a thread/objective pair."""

        thread_data = self._data["threads"].get(
            thread_id, {"history": [], "thread_summary": "", "project_state": {}}
        )
        summary = thread_data.get("thread_summary", "")
        history: List[Dict[str, Any]] = thread_data.get("history", [])
        relevant = self._retrieve_relevant(history, objective, self._retrieval_top_k)
        records = [self._dict_to_record(item) for item in relevant]
        project_state = dict(thread_data.get("project_state", {}))
        return MemoryContext(summary=summary, records=records, project_state=project_state)

    def build_context_messages(self, context: MemoryContext) -> List[SystemMessage]:
        """Create system messages that expose memory context to the agent."""

        messages: List[SystemMessage] = []
        if context.summary:
            messages.append(
                SystemMessage(
                    content=(
                        "长期记忆摘要：\n"
                        f"{context.summary.strip()}"
                    )
                )
            )

        for idx, record in enumerate(context.records, 1):
            record_lines = [
                f"历史任务 #{idx}",
                f"时间: {record.created_at}",
                f"目标: {record.objective}",
            ]
            if record.summary:
                record_lines.append(f"摘要: {record.summary}")
            if record.highlights:
                record_lines.append(f"重点: {record.highlights}")
            messages.append(SystemMessage(content="\n".join(record_lines)))

        return messages

    def store_conversation(
        self,
        thread_id: str,
        objective: str,
        messages: Iterable[BaseMessage],
        result_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Persist the conversation history for future sessions."""

        normalised_messages = self._normalise_messages(list(messages))
        trimmed_messages, summary_addition = self._compress_messages(normalised_messages)
        highlights = self._build_highlights(normalised_messages, result_text)
        record = MemoryRecord(
            objective=objective,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
            summary=summary_addition,
            highlights=highlights,
            messages=trimmed_messages,
            metadata=metadata or {},
        )

        metadata = metadata or {}

        with self._lock:
            thread_data = self._data["threads"].setdefault(
                thread_id,
                {"history": [], "thread_summary": "", "project_state": {}},
            )
            thread_data["history"].append(self._record_to_dict(record))
            if summary_addition:
                thread_data["thread_summary"] = self._merge_thread_summary(
                    thread_data.get("thread_summary", ""), summary_addition
                )
            thread_data["thread_summary"] = self._trim_text(
                thread_data.get("thread_summary", ""), self._thread_summary_max_chars
            )

            project_state_update = metadata.get("project_state")
            if project_state_update:
                merged = self._merge_project_state(
                    thread_data.get("project_state", {}), project_state_update
                )
                thread_data["project_state"] = merged

            self._save()

        return record

    # ------------------------------------------------------------------
    # Project state helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_project_state(
        existing: Dict[str, Any], update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two nested dictionaries describing project artefacts."""

        if not existing:
            existing = {}

        merged = json.loads(json.dumps(existing)) if existing else {}

        def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in source.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    target[key] = _deep_merge(dict(target[key]), value)
                else:
                    target[key] = value
            return target

        if update:
            merged = _deep_merge(dict(merged), update)
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._storage_path.exists():
            return

        try:
            with self._storage_path.open("r", encoding="utf-8") as file:
                self._data = json.load(file)
        except Exception:
            # A corrupted store should not prevent the agent from working.
            self._data = {"threads": {}}

    def _save(self) -> None:
        tmp_path = self._storage_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(self._data, file, ensure_ascii=False, indent=2)
        tmp_path.replace(self._storage_path)

    @staticmethod
    def _record_to_dict(record: MemoryRecord) -> Dict[str, Any]:
        return {
            "objective": record.objective,
            "created_at": record.created_at,
            "summary": record.summary,
            "highlights": record.highlights,
            "messages": record.messages,
            "metadata": record.metadata,
        }

    @staticmethod
    def _dict_to_record(record_dict: Dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            objective=record_dict.get("objective", ""),
            created_at=record_dict.get("created_at", ""),
            summary=record_dict.get("summary", ""),
            highlights=record_dict.get("highlights", ""),
            messages=list(record_dict.get("messages", [])),
            metadata=dict(record_dict.get("metadata", {})),
        )

    def _normalise_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        normalised: List[Dict[str, Any]] = []
        for message in messages:
            entry: Dict[str, Any] = {
                "type": getattr(message, "type", message.__class__.__name__.lower()),
                "content": getattr(message, "content", ""),
            }
            name = getattr(message, "name", None)
            if name:
                entry["name"] = name
            additional = getattr(message, "additional_kwargs", None)
            if additional:
                entry["additional_kwargs"] = additional
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            normalised.append(entry)
        return normalised

    def _compress_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], str]:
        if len(messages) <= self._max_turns:
            return messages, ""

        older_messages = messages[:-self._max_turns]
        summary = self._build_summary(older_messages)
        trimmed = messages[-self._max_turns :]
        return trimmed, summary

    def _build_summary(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""

        # First attempt: use an available LLM to create a concise summary.
        if get_llm_manager is not None:
            try:
                llm = get_llm_manager().get_llm()
                prompt_parts = [
                    "请根据以下多轮对话内容生成一个不超过 180 字的摘要。",
                    "摘要需覆盖用户目标、关键工具调用及最终结论。",
                ]
                conversation_text = []
                for msg in messages:
                    role = msg.get("type", "assistant")
                    role_prefix = {
                        "human": "用户",
                        "ai": "助手",
                        "system": "系统",
                        "tool": "工具",
                    }.get(role, role)
                    content = msg.get("content", "")
                    conversation_text.append(f"{role_prefix}: {content}")
                summary_prompt = "\n".join(prompt_parts + conversation_text)
                summary_response = llm.invoke(summary_prompt)
                if isinstance(summary_response, (AIMessage, SystemMessage, HumanMessage)):
                    summary_text = getattr(summary_response, "content", "")
                else:
                    summary_text = str(summary_response)
                summary_text = summary_text.strip()
                if summary_text:
                    return self._trim_text(summary_text, self._summary_max_chars)
            except Exception:
                # Fall back to heuristic summarisation on any error.
                pass

        # Heuristic fallback: capture the first and last user/assistant turns.
        user_snippets = [
            self._truncate(msg.get("content", ""))
            for msg in messages
            if msg.get("type") == "human"
        ]
        assistant_snippets = [
            self._truncate(msg.get("content", ""))
            for msg in messages
            if msg.get("type") in {"ai", "assistant"}
        ]

        summary_lines: List[str] = []
        if user_snippets:
            summary_lines.append(f"用户意图: {user_snippets[0]}")
            if len(user_snippets) > 1:
                summary_lines.append(f"最新提问: {user_snippets[-1]}")
        if assistant_snippets:
            summary_lines.append(f"助手回应: {assistant_snippets[-1]}")

        joined = "\n".join(summary_lines)
        return self._trim_text(joined, self._summary_max_chars)

    def _merge_thread_summary(self, existing: str, addition: str) -> str:
        if not existing:
            return self._trim_text(addition, self._thread_summary_max_chars)
        merged = f"{existing.strip()}\n---\n{addition.strip()}"
        return self._trim_text(merged, self._thread_summary_max_chars)

    def _build_highlights(
        self, messages: List[Dict[str, Any]], result_text: Optional[str]
    ) -> str:
        user_messages = [m for m in messages if m.get("type") == "human"]
        assistant_messages = [m for m in messages if m.get("type") in {"ai", "assistant"}]

        highlights: List[str] = []
        if user_messages:
            highlights.append(f"最终用户需求: {self._truncate(user_messages[-1].get('content', ''))}")
        if result_text:
            highlights.append(f"最终答复: {self._truncate(result_text)}")
        elif assistant_messages:
            highlights.append(f"最新助手输出: {self._truncate(assistant_messages[-1].get('content', ''))}")

        return "\n".join(highlights)

    def _retrieve_relevant(
        self,
        history: List[Dict[str, Any]],
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not history:
            return []

        if not query:
            return history[-top_k:]

        query_lower = query.lower()
        scored: List[tuple[float, Dict[str, Any]]] = []
        for record in history:
            haystack = " ".join(
                [
                    record.get("objective", ""),
                    record.get("summary", ""),
                    record.get("highlights", ""),
                ]
            ).lower()
            score = SequenceMatcher(None, query_lower, haystack).ratio()
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_records = [record for _, record in scored[:top_k]]
        return top_records

    @staticmethod
    def _truncate(text: str, limit: int = 160) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _trim_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        # Retain the most recent information by keeping the tail of the text.
        return text[-limit:]


__all__ = [
    "ConversationMemoryStore",
    "MemoryContext",
    "MemoryRecord",
]

