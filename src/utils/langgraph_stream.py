"""Utilities for streaming LangGraph agent events with a unified schema."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.agent.agent_new import (
    AgentState,
    build_graph,
    build_project_state_message,
    conversation_memory,
    create_initial_state,
    serialise_project_state,
)

EventHandler = Callable[[Dict[str, Any]], Awaitable[None]]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalised: List[Any] = []
        for item in content:
            if isinstance(item, BaseMessage):
                normalised.append(serialize_message(item))
            else:
                normalised.append(_normalise_content(item))
        return normalised
    if isinstance(content, dict):
        return {key: _normalise_content(value) for key, value in content.items()}
    return content


def serialize_message(message: BaseMessage) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": message.type,
        "content": _normalise_content(getattr(message, "content", "")),
    }

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        payload["additional_kwargs"] = _normalise_content(additional_kwargs)

    metadata = getattr(message, "metadata", None)
    if metadata:
        payload["metadata"] = _normalise_content(metadata)

    name = getattr(message, "name", None)
    if name:
        payload["name"] = name

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = _normalise_content(tool_calls)

    return payload


def _plan_default(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return serialize_message(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _plan_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plan_default(item) for item in value]
    return str(value)


def _serialise_plan(plan: Iterable[Any]) -> Any:
    try:
        return json.loads(json.dumps(list(plan), ensure_ascii=False, default=_plan_default))
    except TypeError:
        return [_plan_default(item) for item in plan]


def _fingerprint(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_plan_default)
    except TypeError:
        return str(value)


def _extract_final_ai_message(messages: Iterable[BaseMessage]) -> Optional[BaseMessage]:
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            return message
    return None


class LangGraphEventAdapter:
    """Transforms LangGraph state deltas into structured event payloads."""

    def __init__(self, run_id: str, thread_id: str) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self._plan_fingerprint: Optional[str] = None
        self._message_count: int = 0
        self.final_message: Optional[BaseMessage] = None
        self.error_info: Optional[Any] = None
        self.last_node: str = ""

    def _event(self, event_type: str, node: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": event_type,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "node": node,
            "payload": payload,
            "ts": _iso_now(),
        }

    def build_start_event(self, objective: str, input_files: List[str]) -> Dict[str, Any]:
        return self._event(
            "start",
            "START",
            {
                "objective": objective,
                "input_files": input_files,
            },
        )

    def build_error_event(self, node: str, detail: Any) -> Dict[str, Any]:
        if isinstance(detail, dict):
            payload = detail
        else:
            payload = {"detail": detail}
        return self._event("error", node, payload)

    def register_runtime_error(self, error: Exception) -> None:
        self.error_info = {"detail": str(error)}

    def process_node_event(self, node: str, state: AgentState) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        self.last_node = node
        execution_status = state.get("execution_status")
        events.append(
            self._event(
                "node_enter",
                node,
                {
                    "execution_status": execution_status,
                    "next_step": state.get("next_step"),
                },
            )
        )

        plan = state.get("plan") or []
        plan_payload = _serialise_plan(plan)
        plan_fingerprint = _fingerprint(plan_payload)
        if plan_payload and plan_fingerprint != self._plan_fingerprint:
            self._plan_fingerprint = plan_fingerprint
            events.append(
                self._event(
                    "plan_update",
                    node,
                    {
                        "plan": plan_payload,
                    },
                )
            )

        messages: List[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
        new_messages = messages[self._message_count :]
        self._message_count = len(messages)

        for message in new_messages:
            serialised = serialize_message(message)
            if isinstance(message, AIMessage) and message.tool_calls:
                events.append(self._event("tool_call", node, {"message": serialised}))
            elif isinstance(message, ToolMessage):
                events.append(self._event("tool_result", node, {"message": serialised}))

        if node == "response" and messages:
            final_ai = _extract_final_ai_message(messages)
            if final_ai is not None:
                self.final_message = final_ai

        if node == "error_info":
            self.error_info = state  # type: ignore[assignment]
            events.append(self.build_error_event(node, state))
        elif execution_status == "failed" and self.error_info is None:
            self.error_info = {"execution_status": execution_status}
            events.append(self.build_error_event(node, self.error_info))

        return events

    def build_end_event(self, final_state: Optional[AgentState]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if final_state is not None:
            payload["execution_status"] = final_state.get("execution_status")
        if self.final_message is not None:
            payload["response"] = serialize_message(self.final_message)
        if self.error_info is not None:
            payload["error"] = self.error_info
        node = self.last_node or ("response" if self.error_info is None else "error_info")
        return self._event("end", node, payload)


def _store_conversation(
    objective: str,
    input_files: Optional[List[str]],
    thread_id: str,
    final_state: Optional[AgentState],
    final_message: Optional[BaseMessage],
    error_info: Optional[Any],
) -> None:
    if final_state is None:
        return

    messages: Iterable[BaseMessage] = final_state.get("messages", [])  # type: ignore[assignment]
    result_text: Optional[str] = None
    if final_message is not None:
        result_text = getattr(final_message, "content", None)
    elif error_info is not None:
        result_text = str(error_info)

    conversation_memory.store_conversation(
        thread_id=thread_id,
        objective=objective,
        messages=messages,
        result_text=result_text,
        metadata={
            "input_files": input_files or [],
            "project_state": serialise_project_state(final_state.get("project_state", {})),
        },
    )


async def run_agent_stream(
    objective: str,
    input_files: Optional[List[str]],
    thread_id: Optional[str],
    run_id: Optional[str],
    event_handler: EventHandler,
) -> Tuple[Optional[BaseMessage], Optional[Any]]:
    initial_state, resolved_thread_id = create_initial_state(objective, input_files, thread_id)
    resolved_run_id = run_id or str(uuid4())
    adapter = LangGraphEventAdapter(resolved_run_id, resolved_thread_id)

    await event_handler(adapter.build_start_event(objective, initial_state.get("input_files", [])))

    graph = build_graph()
    final_state: Optional[AgentState] = None

    try:
        async for event in graph.astream(
            initial_state,
            config={"recursion_limit": 50, "configurable": {"thread_id": resolved_thread_id}},
        ):
            for node, state in event.items():
                if node == "__end__":
                    continue
                final_state = state
                for payload in adapter.process_node_event(node, state):
                    await event_handler(payload)
    except Exception as exc:  # pragma: no cover - defensive
        adapter.register_runtime_error(exc)
        await event_handler(adapter.build_error_event("runtime", {"detail": str(exc)}))
    finally:
        end_event = adapter.build_end_event(final_state)
        await event_handler(end_event)
        _store_conversation(
            objective,
            input_files,
            adapter.thread_id,
            final_state,
            adapter.final_message,
            adapter.error_info,
        )

    return adapter.final_message, adapter.error_info
