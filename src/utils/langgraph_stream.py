"""Utilities for streaming LangGraph agent events with a unified schema.

Supports multiple streaming modes:
- values: Full state after each step
- updates: State updates after each step (default)
- messages: LLM token-by-token streaming
- custom: Custom user-defined events
- debug: Maximum verbosity
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.agent.state import AgentState
from src.agent.graph import (
    get_agent_graph,
    create_initial_state,
    build_project_state_message,
    conversation_memory,
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


def as_state_dict(state: Any) -> Dict[str, Any]:
    """Coerce LangGraph StateSnapshot-like objects into plain dicts."""
    if state is None:
        return {}
    if isinstance(state, dict):
        return state
    for attr in ("values", "data"):
        if hasattr(state, attr):
            value = getattr(state, attr)
            if isinstance(value, dict):
                return value
    try:
        return dict(state)
    except Exception:
        return {"_state": state}


class LangGraphEventAdapter:
    """Transforms LangGraph state deltas into structured event payloads.

    Enhanced to support native LangGraph streaming modes:
    - messages mode: Captures LLM token-by-token output
    - updates mode: Captures state changes per node
    - custom mode: Captures custom user events
    """

    def __init__(self, run_id: str, thread_id: str) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self._plan_fingerprint: Optional[str] = None
        self._message_count: int = 0
        self.final_message: Optional[BaseMessage] = None
        self.error_info: Optional[Any] = None
        self.last_node: str = ""

        # Token streaming state
        self._accumulated_tokens: Dict[str, str] = {}  # node_id -> accumulated content
        self._last_token_time: float = 0.0

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
                "message": "🚀 Starting analysis...",
            },
        )

    def build_progress_event(self, node: str, progress: float, message: str) -> Dict[str, Any]:
        """Build a progress update event for UI feedback."""
        return self._event(
            "progress",
            node,
            {
                "progress": progress,
                "message": message,
            },
        )

    def build_token_event(
        self,
        node: str,
        token: str,
        is_complete: bool = False,
    ) -> Dict[str, Any]:
        """Build a token streaming event for LLM output.

        Args:
            node: The node generating the token
            token: The token content
            is_complete: Whether this is the final token
        """
        return self._event(
            "token",
            node,
            {
                "token": token,
                "is_complete": is_complete,
            },
        )

    def build_node_enter_event(self, node: str) -> Dict[str, Any]:
        """Build a lightweight node-enter event for token streaming mode."""
        return self._event(
            "node_enter",
            node,
            {
                "next_step": node,
                "message": f"⚙️ Executing: {node}",
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
        """Process node execution event and emit structured updates.

        Returns a list of events to stream to the client.
        """
        events: List[Dict[str, Any]] = []
        self.last_node = node
        state_dict = as_state_dict(state)
        execution_status = state_dict.get("execution_status")

        # Node entry event with progress
        events.append(
            self._event(
                "node_enter",
                node,
                {
                    "execution_status": execution_status,
                    "next_step": state_dict.get("next_step"),
                    "message": f"⚙️ Executing: {node}",
                },
            )
        )

        # Plan update if changed
        plan = state_dict.get("plan") or []
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
                        "message": f"📋 Plan updated: {len(plan)} steps",
                    },
                )
            )

        # Process messages
        messages: List[BaseMessage] = state_dict.get("messages", [])  # type: ignore[assignment]
        new_messages = messages[self._message_count :]
        self._message_count = len(messages)

        for message in new_messages:
            serialised = serialize_message(message)
            if isinstance(message, AIMessage) and message.tool_calls:
                events.append(
                    self._event(
                        "tool_call",
                        node,
                        {
                            "message": serialised,
                            "tool_count": len(message.tool_calls),
                            "message": f"🔧 Calling {len(message.tool_calls)} tool(s)",
                        },
                    )
                )
            elif isinstance(message, ToolMessage):
                events.append(
                    self._event(
                        "tool_result",
                        node,
                        {
                            "message": serialised, # This should be the tool's actual output
                        },
                    )
                )

        # Capture final response message
        if node == "response" and messages:
            final_ai = _extract_final_ai_message(messages)
            if final_ai is not None:
                self.final_message = final_ai

        # Error handling
        if node == "error_info":
            self.error_info = state_dict
            events.append(self.build_error_event(node, state_dict))
        elif execution_status == "failed" and self.error_info is None:
            self.error_info = state_dict.get("error_info") or {"execution_status": execution_status}
            events.append(self.build_error_event(node, self.error_info))

        return events

    def build_end_event(self, final_state: Optional[AgentState]) -> Dict[str, Any]:
        """Build the end event with summary."""
        payload: Dict[str, Any] = {
            "message": "✅ Analysis complete!",
        }
        if final_state is not None:
            final_state_dict = as_state_dict(final_state)
            payload["execution_status"] = final_state_dict.get("execution_status")
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

    final_state_dict = as_state_dict(final_state)
    messages: Iterable[BaseMessage] = final_state_dict.get("messages", [])  # type: ignore[assignment]
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
            "project_state": serialise_project_state(final_state_dict.get("project_state", {})),
        },
    )


async def run_agent_stream(
    objective: str,
    input_files: Optional[List[str]],
    thread_id: Optional[str],
    run_id: Optional[str],
    event_handler: EventHandler,
    stream_mode: str = "updates",
) -> Tuple[Optional[BaseMessage], Optional[Any]]:
    """Run agent with streaming support.

    Args:
        objective: User's analysis objective
        input_files: Optional list of input file paths
        thread_id: Conversation thread identifier
        run_id: Execution run identifier
        event_handler: Async callback for streaming events
        stream_mode: LangGraph streaming mode
            - "updates": Stream state updates (default, best for progress tracking)
            - "values": Stream full state (verbose)
            - "messages": Stream LLM tokens (best for real-time text output)
            - "custom": Stream custom events only
            - "debug": Maximum verbosity

    Returns:
        Tuple of (final_message, error_info)
    """
    initial_state = create_initial_state(
        objective,
        input_files,
        thread_id,
        run_id=run_id,
    )
    resolved_thread_id = initial_state.get("thread_id") or thread_id or str(uuid4())
    resolved_run_id = run_id or initial_state.get("run_id") or str(uuid4())
    adapter = LangGraphEventAdapter(resolved_run_id, resolved_thread_id)

    # Send start event
    await event_handler(adapter.build_start_event(objective, initial_state.get("input_files", [])))

    graph = get_agent_graph()
    final_state: Optional[AgentState] = None

    try:
        # Use native LangGraph streaming
        if stream_mode == "messages":
            # Token-level streaming
            last_node: Optional[str] = None
            async for message_chunk, metadata in graph.astream(
                initial_state,
                config={
                    "recursion_limit": 50,
                    "configurable": {"thread_id": resolved_thread_id},
                },
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "unknown")
                if node != last_node:
                    last_node = node
                    adapter.last_node = node
                    await event_handler(adapter.build_node_enter_event(node))

                if message_chunk.content:
                    await event_handler(
                        adapter.build_token_event(node, message_chunk.content)
                    )

            # After token streaming, get final state
            final_state = graph.get_state(
                config={"configurable": {"thread_id": resolved_thread_id}}
            )

        elif stream_mode == "updates":
            # Original node-level streaming
            async for event in graph.astream(
                initial_state,
                config={
                    "recursion_limit": 50,
                    "configurable": {"thread_id": resolved_thread_id},
                },
                stream_mode="updates",
            ):
                for node, state in event.items():
                    if node == "__end__":
                        final_state = state
                        continue
                    final_state = state
                    for payload in adapter.process_node_event(node, state):
                        await event_handler(payload)

        else:
            # Other modes (values, debug, custom)
            async for chunk in graph.astream(
                initial_state,
                config={
                    "recursion_limit": 50,
                    "configurable": {"thread_id": resolved_thread_id},
                },
                stream_mode=stream_mode,
            ):
                await event_handler(
                    {
                        "type": "raw",
                        "run_id": resolved_run_id,
                        "thread_id": resolved_thread_id,
                        "stream_mode": stream_mode,
                        "payload": chunk,
                        "ts": _iso_now(),
                    }
                )
            # Get final state
            final_state = graph.get_state(
                config={"configurable": {"thread_id": resolved_thread_id}}
            )

    except Exception as exc:  # pragma: no cover - defensive
        import traceback
        traceback.print_exc()  # Print full traceback
        adapter.register_runtime_error(exc)
        await event_handler(adapter.build_error_event("runtime", {"detail": str(exc)}))
    finally:
        # Extract final message from state
        if final_state:
            final_state_dict = as_state_dict(final_state)
            messages = final_state_dict.get("messages", [])
            if messages:
                final_ai = _extract_final_ai_message(messages)
                if final_ai:
                    adapter.final_message = final_ai

        # Send end event
        end_event = adapter.build_end_event(final_state)
        await event_handler(end_event)

        # Store conversation
        _store_conversation(
            objective,
            input_files,
            adapter.thread_id,
            final_state,
            adapter.final_message,
            adapter.error_info,
        )

    return adapter.final_message, adapter.error_info
