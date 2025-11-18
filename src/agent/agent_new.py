import re
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, List, Tuple, Union, Dict, Any, Optional
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from typing import Literal
import logging
import structlog
from uuid import uuid4

from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.exceptions import OutputParserException
from langgraph.graph import END
from langgraph.graph import StateGraph, START 
from langgraph.checkpoint.memory import MemorySaver


from src.agent.tool_registry import TOOLS
from config.setting import settings
from src.tools.dataset_qa import retrieve_bio_context
from src.utils.llm_manager import get_llm_manager
from src.utils.path_manager import path_manager, extract_paths_from_objective, validate_h5ad_file, create_analysis_work_dir
from src.memory.conversation_memory import ConversationMemoryStore, MemoryContext


MEMORY_QUERY_PATTERNS = [
    r"what\s+did\s+i\s+ask\s+(you\s+)?(before|last\s+time)",
    r"remember\s+(our|the)\s+(last|previous)\s+(conversation|task)",
    r"what\s+was\s+my\s+previous\s+(request|question|mission)",
    r"recall\s+(?:the\s+)?(earlier|previous)\s+(?:instructions|task)",
]

STATUS_QUERY_PATTERNS = [
    r"how\s+is\s+(the\s+)?(analysis|task|job|mission)\s+(going|progressing)",
    r"what\s+is\s+the\s+status\s+of",
    r"give\s+me\s+an\s+update",
    r"have\s+you\s+finished",
]

INTENT_SYNONYMS: Dict[str, str] = {
    "memory": "memory_query",
    "memory lookup": "memory_query",
    "memory_query": "memory_query",
    "memory retrieval": "memory_query",
    "remember": "memory_query",
    "status": "status_check",
    "status_check": "status_check",
    "progress": "status_check",
    "update": "status_check",
    "direct": "direct_response",
    "direct_response": "direct_response",
    "chat": "chitchat",
    "chitchat": "chitchat",
    "greeting": "greeting",
}

ALLOWED_INTENT_TYPES: Tuple[str, ...] = (
    "cell_annotation",
    "clustering_analysis",
    "marker_gene_analysis",
    "pathway_analysis",
    "regulatory_network",
    "differential_expression",
    "trajectory_analysis",
    "quality_control",
    "data_visualization",
    "generic",
    "memory_query",
    "status_check",
    "direct_response",
    "clarification",
    "greeting",
    "chitchat",
)

NON_TASK_INTENTS = {"memory_query", "status_check", "direct_response", "clarification", "greeting", "chitchat"}

MIN_TASK_CONFIDENCE = 0.55

DEFAULT_INTENT_DESCRIPTIONS: Dict[str, str] = {
    "memory_query": "Retrieve prior conversation context or missions from long-term memory.",
    "status_check": "Report the current status or progress of the active task.",
    "direct_response": "Answer the user directly without executing analysis tools.",
    "clarification": "Ask clarifying questions to better understand the request.",
    "greeting": "Handle conversational greetings without analysis.",
    "chitchat": "Engage in small talk without triggering tools.",
}

TOOL_STEP_LABELS: Dict[str, str] = {
    "load_h5ad_data": "加载与预处理数据",
    "extract_embeddings_with_scgpt": "提取 scGPT 嵌入",
    "cluster_and_diff": "聚类与差异分析",
    "annotate_with_markers": "细胞类型注释",
    "run_ssgsea_enrichment": "ssGSEA 富集分析",
    "dataset_bio_qa": "知识库问答总结",
}


@dataclass
class FailureInjectionConfig:
    tool_names: List[str] = field(default_factory=lambda: ["load_h5ad_data"])
    rate: float = 0.0
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_names": list(self.tool_names),
            "rate": float(self.rate),
            "seed": self.seed,
        }


@dataclass
class AgentRuntimeConfig:
    planner_mode: Literal["multi_agent", "linear", "disabled"] = "multi_agent"
    enable_replanner: bool = True
    enable_memory: bool = True
    enable_rag: bool = True
    enable_dataset_qa: bool = True
    allow_tool_execution: bool = True
    track_metrics: bool = True
    max_replan_attempts: int = 4
    failure_injection: Optional[FailureInjectionConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "planner_mode": self.planner_mode,
            "enable_replanner": self.enable_replanner,
            "enable_memory": self.enable_memory,
            "enable_rag": self.enable_rag,
            "enable_dataset_qa": self.enable_dataset_qa,
            "allow_tool_execution": self.allow_tool_execution,
            "track_metrics": self.track_metrics,
            "max_replan_attempts": self.max_replan_attempts,
        }
        if self.failure_injection is not None:
            payload["failure_injection"] = self.failure_injection.to_dict()
        return payload


def _default_linear_plan(include_dataset_qa: bool = True) -> List[str]:
    steps = [
        "Use load_h5ad_data to preprocess the uploaded AnnData file.",
        "Call extract_embeddings_with_scgpt to compute latent embeddings.",
        "Execute cluster_and_diff for clustering and differential expression.",
        "Invoke annotate_with_markers to label cell types.",
    ]
    if include_dataset_qa:
        steps.append("Leverage dataset_bio_qa for knowledge-base grounded reporting.")
    steps.append("Run run_ssgsea_enrichment to compute pathway enrichment scores.")
    return steps


def _get_runtime_config(state: "AgentState") -> Dict[str, Any]:
    return dict(state.get("runtime_config", {}) or {})


def _config_flag(state: "AgentState", key: str, default: bool) -> bool:
    config = _get_runtime_config(state)
    return bool(config.get(key, default))


def _config_value(state: "AgentState", key: str, default: Any) -> Any:
    config = _get_runtime_config(state)
    return config.get(key, default)


def _ensure_replanner_route(state: "AgentState") -> None:
    if _config_flag(state, "enable_replanner", True):
        return
    state["messages"].append(
        AIMessage(
            content="<REPLAN_SKIPPED>Intelligent replanner disabled by configuration. Returning response directly.</REPLAN_SKIPPED>"
        )
    )
    state["next_step"] = "response"
    if state.get("execution_status") != "completed":
        state["execution_status"] = "failed"


def _looks_like_memory_query(text: str) -> bool:
    """Heuristic detection for questions that should use memory retrieval."""

    if not text:
        return False

    lowered = text.lower()
    for pattern in MEMORY_QUERY_PATTERNS:
        if re.search(pattern, lowered):
            return True

    # Simple fallbacks for shorter prompts
    keywords = {"remember", "memory", "recall", "previous", "last time"}
    return any(keyword in lowered for keyword in keywords)


def _looks_like_status_query(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower()
    for pattern in STATUS_QUERY_PATTERNS:
        if re.search(pattern, lowered):
            return True

    keywords = {"status", "progress", "update", "finished"}
    return any(keyword in lowered for keyword in keywords)


def _normalize_intent_label(raw_label: Optional[str]) -> str:
    if not raw_label:
        return "generic"

    cleaned = raw_label.strip().lower()
    cleaned = INTENT_SYNONYMS.get(cleaned, cleaned)

    if cleaned not in ALLOWED_INTENT_TYPES:
        return "generic"

    return cleaned


def _is_task_intent(intent: "Intent") -> bool:
    normalized = _normalize_intent_label(getattr(intent, "intent", None))
    if normalized in NON_TASK_INTENTS:
        return False
    return intent.confidence >= MIN_TASK_CONFIDENCE


def _sanitize_intent(intent: "Intent") -> "Intent":
    intent_dict = intent.model_dump()
    normalized_label = _normalize_intent_label(intent_dict.get("intent"))
    description = intent_dict.get("description") or DEFAULT_INTENT_DESCRIPTIONS.get(normalized_label, "")
    justification = intent_dict.get("justification") or description or ""
    dependencies = intent_dict.get("dependencies") or []

    sanitized = Intent(
        intent=normalized_label,
        description=description,
        confidence=intent_dict.get("confidence", 0.0),
        dependencies=list(dependencies),
        justification=justification,
    )
    return sanitized


def _postprocess_intent_response(response: "IntentResponse") -> "IntentResponse":
    sanitized_intents: List[Intent] = []
    for intent in response.intents:
        try:
            sanitized_intents.append(_sanitize_intent(intent))
        except ValidationError as exc:
            logger.warning("[Intent Recognition] Dropped malformed intent: %s", exc)

    if not sanitized_intents:
        sanitized_intents.append(
            Intent(
                intent="direct_response",
                description=DEFAULT_INTENT_DESCRIPTIONS["direct_response"],
                confidence=0.0,
                dependencies=[],
                justification="No actionable intents remained after validation.",
            )
        )

    response.intents = sanitized_intents
    task_intents = [intent for intent in sanitized_intents if _is_task_intent(intent)]
    response.is_task = bool(task_intents)
    return response


def _fallback_direct_response(reason: str) -> "IntentResponse":
    logger.info("[Intent Recognition] Falling back to direct response: %s", reason)
    fallback_intent = Intent(
        intent="direct_response",
        description=DEFAULT_INTENT_DESCRIPTIONS["direct_response"],
        confidence=0.0,
        dependencies=[],
        justification=f"Fallback because structured intent parsing failed: {reason}",
    )
    return IntentResponse(intents=[fallback_intent], is_task=False)


def _rule_based_intent(objective: str) -> Optional["IntentResponse"]:
    if _looks_like_memory_query(objective):
        intent = Intent(
            intent="memory_query",
            description=DEFAULT_INTENT_DESCRIPTIONS["memory_query"],
            confidence=1.0,
            dependencies=[],
            justification="Detected memory retrieval phrasing in the request.",
        )
        return IntentResponse(intents=[intent], is_task=False)

    if _looks_like_status_query(objective):
        intent = Intent(
            intent="status_check",
            description=DEFAULT_INTENT_DESCRIPTIONS["status_check"],
            confidence=0.9,
            dependencies=[],
            justification="Detected progress inquiry phrasing in the request.",
        )
        return IntentResponse(intents=[intent], is_task=False)

    return None


def _append_unique_human_message(state: "AgentState", content: str) -> None:
    if not content:
        return

    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage) and last_message.content == content:
            return

    state["messages"].append(HumanMessage(content=content))


def _apply_intent_result(
    state: "AgentState",
    result: "IntentResponse",
    *,
    source: str,
    rationale: str,
    raw_response: Optional[Any] = None,
) -> None:
    task_intents = [intent for intent in result.intents if _is_task_intent(intent)]

    if result.is_task and not task_intents:
        # Guard against inconsistencies by forcing direct response routing.
        result.is_task = False

    state["intents"] = task_intents if result.is_task else []
    state["recognized_intents"] = [intent.model_dump() for intent in result.intents]

    planner_mode = _config_value(state, "planner_mode", "multi_agent")
    if not result.is_task or planner_mode == "disabled":
        state["next_step"] = "response"
    else:
        state["next_step"] = "planner"
    trace_payload = {
        "objective": state.get("objective"),
        "source": source,
        "rationale": rationale,
        "is_task": result.is_task,
        "task_intents": [intent.model_dump() for intent in task_intents],
        "classified_intents": [intent.model_dump() for intent in result.intents],
        "raw_response": raw_response.model_dump() if hasattr(raw_response, "model_dump") else raw_response,
    }
    state["intent_trace"] = trace_payload
    logger.info("[Intent Recognition] Final decision: %s", trace_payload)


def _get_recognized_intent_labels(state: "AgentState") -> List[str]:
    """Return the normalized intent labels detected for the current turn."""

    raw_intents = state.get("recognized_intents", []) or []
    labels: List[str] = []

    for item in raw_intents:
        if isinstance(item, Intent):
            labels.append(item.intent)
        elif isinstance(item, dict):
            labels.append(_normalize_intent_label(item.get("intent")))
        elif hasattr(item, "intent"):
            labels.append(_normalize_intent_label(getattr(item, "intent", "")))

    return labels


def _format_memory_response(state: "AgentState") -> str:
    """Build a conversational reply describing the stored long-term memory."""

    summary = (state.get("memory_summary") or "").strip()
    records = state.get("memory_records") or []

    if not summary and not records:
        return (
            "当前会话没有可供检索的长期记忆记录。如需我在未来记住关键信息，"
            "请明确告知我哪些结论或文件需要保存。"
        )

    lines: List[str] = ["### 长期记忆检索结果"]

    if summary:
        lines.append("")
        lines.append("**会话摘要**")
        lines.append(summary)

    if records:
        lines.append("")
        lines.append("**历史任务记录**")
        for idx, record in enumerate(records, 1):
            if isinstance(record, dict):
                record_dict = record
            else:
                record_dict = getattr(record, "__dict__", {})

            objective = record_dict.get("objective") or "（未记录目标）"
            created_at = record_dict.get("created_at") or "时间未知"
            record_summary = record_dict.get("summary") or ""
            highlights = record_dict.get("highlights") or ""

            lines.append(f"{idx}. {objective}（{created_at}）")
            if record_summary:
                lines.append(f"   - 摘要：{record_summary}")
            if highlights:
                lines.append(f"   - 重点：{highlights}")

    return "\n".join(lines).strip()


def _format_status_response(state: "AgentState") -> str:
    """Render a structured status update for project progress inquiries."""

    status_text = (state.get("execution_status") or "未开始").strip() or "未开始"
    lines: List[str] = ["### 项目状态更新", f"- 当前执行状态：{status_text}"]

    remaining_plan = [step for step in state.get("plan", []) if isinstance(step, str) and step.strip()]
    if remaining_plan:
        lines.append("- 待执行步骤：")
        for idx, step in enumerate(remaining_plan, 1):
            lines.append(f"  {idx}. {step.strip()}")

    tool_history = state.get("tool_history") or []
    if tool_history:
        lines.append("- 最近的工具调用：")
        for entry in tool_history[-3:]:
            description_parts: List[str] = []
            tool_name = ""
            if isinstance(entry, dict):
                tool_name = entry.get("tool") or entry.get("tool_name") or "未知工具"
                outcome = entry.get("status") or entry.get("result_status") or entry.get("outcome")
                if outcome:
                    description_parts.append(str(outcome))
                error = entry.get("error") or entry.get("message")
                if error:
                    description_parts.append(str(error))
            else:
                tool_name = str(getattr(entry, "tool", getattr(entry, "tool_name", "未知工具")))
            detail = "；".join(part for part in description_parts if part)
            if detail:
                lines.append(f"  • {tool_name}：{detail}")
            else:
                lines.append(f"  • {tool_name}")

    analysis_notes = state.get("analysis_notes") or {}
    completed_steps = analysis_notes.get("completed_steps")
    if isinstance(completed_steps, list) and completed_steps:
        lines.append("- 已完成步骤：")
        for idx, step in enumerate(completed_steps, 1):
            lines.append(f"  {idx}. {step}")

    recent_note = analysis_notes.get("last_note")
    if isinstance(recent_note, str) and recent_note.strip():
        lines.append(f"- 最新备注：{recent_note.strip()}")

    if len(lines) == 2:
        lines.append("- 尚未记录详细的执行历史。")

    return "\n".join(lines).strip()


def _safe_json_loads(payload: Any) -> Optional[Dict[str, Any]]:
    """Best-effort JSON decoding utility used for tool outputs."""

    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _ensure_project_state(state: "AgentState") -> Dict[str, Any]:
    project_state = state.setdefault("project_state", {})
    project_state.setdefault("datasets", {})
    return project_state


def _get_active_dataset_entry(
    state: "AgentState",
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    project_state = state.get("project_state") or {}
    datasets = project_state.get("datasets") or {}
    dataset_id = project_state.get("active_dataset") or project_state.get("last_dataset")
    entry: Optional[Dict[str, Any]] = None
    if dataset_id and isinstance(datasets.get(dataset_id), dict):
        entry = datasets[dataset_id]
    return dataset_id, entry


def _ensure_dataset_entry(
    state: "AgentState", work_dir: Optional[str] = None, dataset_hint: Optional[str] = None
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    project_state = _ensure_project_state(state)
    datasets: Dict[str, Any] = project_state.setdefault("datasets", {})

    dataset_id: Optional[str] = None
    if work_dir:
        dataset_id = Path(work_dir).name
    elif dataset_hint:
        dataset_id = dataset_hint
    elif state.get("work_dir"):
        dataset_id = Path(state["work_dir"]).name
    else:
        dataset_id = project_state.get("active_dataset") or project_state.get("last_dataset")
        if dataset_id is None and datasets:
            dataset_id = next(iter(datasets))

    if dataset_id is None and work_dir is None:
        return None, None

    if dataset_id is None and work_dir:
        dataset_id = Path(work_dir).name

    entry = datasets.setdefault(dataset_id, {})
    entry.setdefault("completed_steps", [])

    if work_dir:
        entry["work_dir"] = work_dir
        state["work_dir"] = work_dir

    if entry.get("work_dir") and not state.get("work_dir"):
        state["work_dir"] = entry["work_dir"]

    project_state["active_dataset"] = dataset_id
    project_state["last_dataset"] = dataset_id

    return dataset_id, entry


def _mark_completed(entry: Dict[str, Any], tool_name: str) -> None:
    completed = entry.setdefault("completed_steps", [])
    if tool_name not in completed:
        completed.append(tool_name)


def _update_project_state_from_tool(
    state: "AgentState", tool_name: str, tool_args: Dict[str, Any], tool_result: Any
) -> None:
    payload = _safe_json_loads(tool_result)
    project_state = _ensure_project_state(state)

    work_dir = tool_args.get("work_dir") if isinstance(tool_args, dict) else None
    dataset_hint = None

    if isinstance(tool_args, dict):
        input_file = tool_args.get("file_path") or tool_args.get("input_path")
    else:
        input_file = None

    if payload and isinstance(payload, dict):
        work_dir = payload.get("work_dir") or work_dir or payload.get("work_path")
        dataset_hint = payload.get("dataset_id")

    dataset_id, dataset_entry = _ensure_dataset_entry(state, work_dir, dataset_hint)
    if dataset_entry is None:
        return

    if input_file:
        dataset_entry.setdefault("input_files", set())
        if isinstance(dataset_entry["input_files"], set):
            dataset_entry["input_files"].add(str(input_file))
        else:
            files = set(dataset_entry.get("input_files", []))
            files.add(str(input_file))
            dataset_entry["input_files"] = files

    if tool_name == "load_h5ad_data" and payload:
        dataset_entry["preprocessed_path"] = payload.get("preproc_path") or dataset_entry.get("preprocessed_path")
        if payload.get("work_dir"):
            dataset_entry["work_dir"] = payload["work_dir"]
            state["work_dir"] = payload["work_dir"]
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "extract_embeddings_with_scgpt" and payload:
        dataset_entry["embeddings_path"] = payload.get("embeddings_path") or dataset_entry.get("embeddings_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "cluster_and_diff" and payload:
        dataset_entry["clustered_path"] = payload.get("clustered_path") or dataset_entry.get("clustered_path")
        dataset_entry["diff_gene_path"] = payload.get("diff_gene_path") or dataset_entry.get("diff_gene_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "annotate_with_markers" and payload:
        dataset_entry["annotated_path"] = (
            payload.get("annoted_Path")
            or payload.get("annotated_path")
            or dataset_entry.get("annotated_path")
        )
        dataset_entry.setdefault("annotation", {})
        dataset_entry["annotation"].update(
            {
                "candidate_path": payload.get("anno_candidate"),
                "result_path": payload.get("anno_result"),
            }
        )
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "run_ssgsea_enrichment" and payload:
        enrichment = dataset_entry.setdefault("enrichment", {})
        enrichment["ssgsea"] = payload
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "dataset_bio_qa" and payload:
        dataset_entry.setdefault("qa", payload)

    project_state.setdefault("datasets", {})[dataset_id] = dataset_entry

    if isinstance(dataset_entry.get("input_files"), set):
        dataset_entry["input_files"] = sorted(str(item) for item in dataset_entry["input_files"])

    completed_notes = state.setdefault("analysis_notes", {}).setdefault("completed_steps", [])
    human_label = TOOL_STEP_LABELS.get(tool_name, tool_name)
    if human_label not in completed_notes:
        completed_notes.append(human_label)


def _is_tool_already_completed(dataset_entry: Dict[str, Any], tool_name: str) -> bool:
    if not dataset_entry:
        return False

    if tool_name == "load_h5ad_data":
        return bool(dataset_entry.get("preprocessed_path"))
    if tool_name == "extract_embeddings_with_scgpt":
        return bool(dataset_entry.get("embeddings_path"))
    if tool_name == "cluster_and_diff":
        return bool(dataset_entry.get("clustered_path") and dataset_entry.get("diff_gene_path"))
    if tool_name == "annotate_with_markers":
        return bool(
            dataset_entry.get("annotated_path") or dataset_entry.get("annotation", {}).get("result_path")
        )
    if tool_name == "run_ssgsea_enrichment":
        enrichment = dataset_entry.get("enrichment", {}).get("ssgsea", {})
        return bool(enrichment.get("result_paths") or enrichment.get("status"))
    return False


def _identify_tool_from_plan_step(step: str) -> Optional[str]:
    lowered = step.lower()
    for tool_name in (
        "load_h5ad_data",
        "extract_embeddings_with_scgpt",
        "cluster_and_diff",
        "annotate_with_markers",
        "dataset_bio_qa",
        "run_ssgsea_enrichment",
    ):
        if tool_name in lowered:
            return tool_name
    return None


def _prune_completed_plan(state: "AgentState") -> None:
    dataset_id, dataset_entry = _get_active_dataset_entry(state)

    if not dataset_entry:
        return

    plan = state.get("plan") or []
    if not plan:
        return

    filtered_steps: List[str] = []
    pruned = False
    for step in plan:
        if not isinstance(step, str):
            filtered_steps.append(step)
            continue
        tool_name = _identify_tool_from_plan_step(step)
        if tool_name and _is_tool_already_completed(dataset_entry, tool_name):
            logger.info(
                "[General Planner] Skipping completed step '%s' for dataset '%s'",
                tool_name,
                dataset_id,
            )
            pruned = True
            continue
        filtered_steps.append(step)

    if pruned:
        state["plan"] = filtered_steps


def build_project_state_message(project_state: Dict[str, Any]) -> Optional[str]:
    if not project_state:
        return None

    datasets = project_state.get("datasets") or {}
    if not datasets:
        return None

    active_dataset = project_state.get("active_dataset") or project_state.get("last_dataset")
    lines = ["### 已保存的项目状态"]

    for dataset_id, entry in datasets.items():
        prefix = "(当前) " if active_dataset and dataset_id == active_dataset else ""
        line = f"- {prefix}数据集 `{dataset_id}`"
        work_dir = entry.get("work_dir")
        if work_dir:
            line += f"，工作目录：`{work_dir}`"
        lines.append(line)

        completed = entry.get("completed_steps") or []
        if completed:
            human_steps = ", ".join(TOOL_STEP_LABELS.get(step, step) for step in completed)
            lines.append(f"  • 已完成步骤：{human_steps}")

        if entry.get("embeddings_path"):
            lines.append(f"  • 嵌入文件：`{entry['embeddings_path']}`")
        if entry.get("annotated_path"):
            lines.append(f"  • 细胞注释：`{entry['annotated_path']}`")
        enrichment = entry.get("enrichment", {}).get("ssgsea", {})
        if enrichment.get("result_paths"):
            lines.append("  • 已计算 ssGSEA 富集分析结果，可直接复用。")

    return "\n".join(lines)


def serialise_project_state(project_state: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _convert(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, set):
            return sorted(str(item) for item in value)
        return value

    return _convert(project_state or {})


def _extract_latest_user_question(messages: List[BaseMessage], fallback: str = "") -> str:
    """Return the most recent human utterance that looks like a question."""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = (message.content or "").strip()
            if not content:
                continue
            if content.lower().startswith("user dataset available at:"):
                continue
            return content
    return fallback


def _build_dataset_followup_message(dataset_result: Dict[str, Any], work_dir: Optional[str]) -> str:
    """Render a follow-up message after dataset analysis is completed."""

    question = (dataset_result.get("question") or "").strip()
    answer = (dataset_result.get("answer") or "").strip()
    dataset_report_path = dataset_result.get("dataset_report_path") or dataset_result.get("dataset_report")
    celltype_report_path = dataset_result.get("celltype_report_path") or dataset_result.get("celltype_report")
    qa_history_path = dataset_result.get("qa_history_path")
    local_refs = dataset_result.get("local_reference_count", dataset_result.get("local_hits"))
    pubmed_refs = dataset_result.get("pubmed_reference_count", dataset_result.get("pubmed_hits"))

    summary_lines = [
        "### 数据集分析完成",
        "我已基于上传的数据集自动完成质量控制、解读并生成报告。",
    ]

    if work_dir:
        summary_lines.append(f"- 工作目录：`{work_dir}`")
    if dataset_report_path:
        summary_lines.append(f"- 数据集报告：`{dataset_report_path}`")
    if celltype_report_path:
        summary_lines.append(f"- 细胞类型报告：`{celltype_report_path}`")
    if qa_history_path:
        summary_lines.append(f"- 问答历史：`{qa_history_path}`")

    if question:
        summary_lines.append("")
        summary_lines.append(f"**原始问题**：{question}")
    if answer:
        summary_lines.append("**回答摘要**：")
        summary_lines.append(answer)

    if isinstance(local_refs, int) or isinstance(pubmed_refs, int):
        summary_lines.append("")
        summary_lines.append("参考来源统计：")
        summary_lines.append(f"- 本地知识库引用：{local_refs if isinstance(local_refs, int) else len(local_refs or [])} 条")
        summary_lines.append(f"- PubMed 摘要引用：{pubmed_refs if isinstance(pubmed_refs, int) else len(pubmed_refs or [])} 条")

    summary_lines.append("")
    summary_lines.append("如果你还有其他生物信息学问题，请继续提问，我会结合知识库和对话历史继续解答。")
    return "\n".join(summary_lines)


def _generate_bio_rag_answer(question: str, state: "AgentState") -> Tuple[str, Dict[str, Any]]:
    """Generate a conversational answer enhanced with RAG references."""

    try:
        top_k_local = int(getattr(settings, "RETRIVE_TOP_K", 3))
    except (TypeError, ValueError):
        top_k_local = 3
    top_k_local = max(1, top_k_local)

    work_dir = state.get("work_dir")

    try:
        references = retrieve_bio_context(
            question,
            work_dir=work_dir,
            top_k_local=top_k_local,
            top_k_pubmed=3,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("[Response] Failed to retrieve RAG references: %s", exc)
        references = {
            "local_docs": [],
            "pubmed_docs": [],
            "context_sections": [],
            "local_reference_count": 0,
            "pubmed_reference_count": 0,
        }

    context_sections = list(references.get("context_sections", []))
    dataset_result = state.get("analysis_notes", {}).get("dataset_bio_qa")
    if dataset_result:
        dataset_summary: List[str] = []
        dataset_question = dataset_result.get("question")
        dataset_answer = dataset_result.get("answer")
        if dataset_question:
            dataset_summary.append(f"- 初始数据集问题：{dataset_question}")
        if dataset_answer:
            trimmed_answer = dataset_answer.strip()
            if len(trimmed_answer) > 800:
                trimmed_answer = trimmed_answer[:800] + "…"
            dataset_summary.append(f"- 数据集解读摘要：{trimmed_answer}")
        dataset_report = dataset_result.get("dataset_report_path")
        if dataset_report:
            dataset_summary.append(f"- 报告路径：`{dataset_report}`")
        if dataset_summary:
            context_sections.insert(0, "【数据集背景】\n" + "\n".join(dataset_summary))

    context_text = "\n\n".join(section.strip() for section in context_sections if section).strip()
    if not context_text:
        context_text = "（未检索到额外参考资料，也请基于专业知识完成解答。）"

    system_message = SystemMessage(
        content=(
            "You are a senior biomedical scientist. Always provide precise, well-structured "
            "answers in Chinese unless the user explicitly requests another language."
        )
    )
    human_message = HumanMessage(
        content=(
            "请参考以下资料回答用户的问题，并在无法检索到外部知识时结合专业知识推理：\n\n"
            f"{context_text}\n\n"
            f"问题：{question}\n\n"
            "请按照以下结构回答：\n"
            "## 解答\n- 给出直接、严谨的回答\n"
            "## 证据\n- 本地知识库：...（如无写'无'）\n- PubMed：...（如无写'无'）\n"
            "## 建议\n- 如需进一步实验或数据分析，请给出建议\n"
        )
    )

    response_message = llm.invoke([system_message, human_message])
    answer_text = getattr(response_message, "content", str(response_message))
    if isinstance(answer_text, list):
        answer_text = "\n".join(str(part) for part in answer_text)

    metadata = {
        "question": question,
        "local_reference_count": references.get("local_reference_count", 0),
        "pubmed_reference_count": references.get("pubmed_reference_count", 0),
    }
    return answer_text, metadata


def _clean_replanner_output(content: str) -> str:
    """Remove common wrappers the LLM may add around JSON output."""

    if not content:
        return ""

    cleaned = content.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()


def _parse_replanner_json(raw_content: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Attempt to parse the replanner response into JSON with fallbacks."""

    candidates: List[str] = []

    if raw_content and raw_content.strip():
        candidates.append(raw_content.strip())

    cleaned = _clean_replanner_output(raw_content)
    if cleaned and cleaned not in candidates:
        candidates.append(cleaned)

    brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0).strip()
        if candidate not in candidates:
            candidates.append(candidate)

    last_error: Optional[str] = None
    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as err:
            last_error = str(err)

    return None, last_error


structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

memory = MemorySaver()
conversation_memory = ConversationMemoryStore()

tools = TOOLS
tools_by_name = {tool.name: tool for tool in tools}

llm_manager = get_llm_manager()
llm = llm_manager.get_llm()
llm_with_tools = llm_manager.get_llm_with_tools(tools)

class Intent(BaseModel):
    """Intent recognition schema"""

    model_config = ConfigDict(extra="ignore")

    intent: str = Field(description="Normalized intent label from the allowed catalogue")
    description: str = Field(default="", description="Detailed task description")
    confidence: float = Field(default=0.8, ge=0, le=1, description="Recognition confidence level (0-1)")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Prerequisite intention of dependency"
    )
    justification: str = Field(default="", description="Rationale explaining why this label was predicted")

    @field_validator("intent", mode="before")
    def normalize_intent(cls, v):
        return _normalize_intent_label(v)

    @field_validator("confidence", mode="before")
    def clamp_confidence(cls, v):
        try:
            value = float(v)
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, min(1.0, value))

    @field_validator('confidence')
    def round_confidence(cls, v):
        """Confidence level to two decimal places"""
        return round(v, 2)

    @field_validator("dependencies", mode="before")
    def ensure_dependencies(cls, v):
        if not v:
            return []
        if isinstance(v, list):
            return [str(item) for item in v]
        return [str(v)]

    @field_validator("justification", mode="before")
    def ensure_justification(cls, v, values):
        if v:
            return str(v)
        description = values.get("description", "") if values else ""
        return description or ""

class IntentResponse(BaseModel):
    """Intent recognition response"""
    model_config = ConfigDict(extra="ignore")
    intents: List[Intent] = Field(description="List of Identified Intentions")
    is_task: bool = Field(default=True, description="Does the user intend to generate a plan and call tools to complete it")

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class AgentState(TypedDict):
    """Intelligent agent state, used to support complex tasks and dynamic programming"""
    messages: list[BaseMessage]
    objective: str
    input_files: List[str]
    intents: List[Intent]
    plan: List[str]
    next_step: Optional[str]
    memory_summary: str
    memory_records: List[Dict[str, Any]]
    thread_id: str
    replan_attempts: int
    max_replan_attempts: int
    execution_status: str
    intent_trace: Dict[str, Any]
    work_dir: Optional[str]
    tool_history: List[Dict[str, Any]]
    analysis_notes: Dict[str, Any]
    recognized_intents: NotRequired[List[Dict[str, Any]]]
    project_state: Dict[str, Any]

# User intent recognition
async def intent_recognition(state: AgentState) -> AgentState:
    """
    Universal intent recognition: using LLM to dynamically recognize user intent, capable of analyzing multiple user intentions and potential user intentions
    """
    objective = state["objective"]
    input_files = state.get("input_files", [])
    if input_files:
        dataset_message = f"User dataset available at: {input_files[0]}"
        if not state["messages"] or not (
            isinstance(state["messages"][0], HumanMessage)
            and state["messages"][0].content == dataset_message
        ):
            state["messages"].insert(0, HumanMessage(content=dataset_message))

    _append_unique_human_message(state, objective)

    rule_based = _rule_based_intent(objective)
    if rule_based:
        rationale = rule_based.intents[0].justification if rule_based.intents else "Rule-based classification"
        _apply_intent_result(state, rule_based, source="rule", rationale=rationale, raw_response=rule_based)
        return state

    allowed_labels_text = ", ".join(ALLOWED_INTENT_TYPES)
    non_task_text = ", ".join(sorted(NON_TASK_INTENTS))
    intent_prompt = f"""
You are a helpful biological data analysis assistant responsible for analyzing user intent.
Classify the latest user instruction and decide whether it requires executing bioinformatics analysis tools.

Return a JSON object that conforms to this schema:
{{
  "intents": [
    {{
      "intent": "<one of: {allowed_labels_text}>",
      "description": "Short natural language description of the user's goal",
      "confidence": 0.0-1.0,
      "dependencies": ["names of prerequisite intents"],
      "justification": "Why this label was selected"
    }}
  ],
  "is_task": true | false
}}

Guidelines:
- Use intents from {{{non_task_text}}} for conversational, clarification, status, or memory retrieval requests.
- Set "is_task" to false when every recognized intent belongs to {{{non_task_text}}}.
- Only mark "is_task" as true if at least one intent clearly requires running computational analysis.
- If uncertain, choose the safest non-tool intent and set "is_task" to false.
"""

    try:
        intent_llm = llm.with_structured_output(IntentResponse)
        messages = [SystemMessage(content=intent_prompt)] + state["messages"]
        raw_result = intent_llm.invoke(messages)
        logger.info("[Intent Recognition] Raw classifier output: %s", raw_result)

        processed_result = _postprocess_intent_response(raw_result)
        _apply_intent_result(
            state,
            processed_result,
            source="llm",
            rationale="Structured intent classification",
            raw_response=raw_result,
        )

    except (ValidationError, OutputParserException, ValueError) as err:
        logger.error("[Intent Recognition] Intent parsing failed: %s", err)
        fallback = _fallback_direct_response(str(err))
        _apply_intent_result(state, fallback, source="fallback", rationale=str(err), raw_response=fallback)

    except Exception as err:
        logger.exception("[Intent Recognition] Unexpected failure: %s", err)
        fallback = _fallback_direct_response(str(err))
        _apply_intent_result(state, fallback, source="fallback", rationale=str(err), raw_response=fallback)

    return state

async def response(state: AgentState):
    """Final response generation with dataset follow-up and RAG support."""

    analysis_notes = state.setdefault("analysis_notes", {})
    final_message: Optional[str] = None

    try:
        recognized_labels = _get_recognized_intent_labels(state)
        dataset_result = analysis_notes.get("dataset_bio_qa")

        if "memory_query" in recognized_labels:
            final_message = _format_memory_response(state)
        elif "status_check" in recognized_labels:
            final_message = _format_status_response(state)
        elif dataset_result and _config_flag(state, "enable_dataset_qa", True):
            work_dir = dataset_result.get("work_dir") or state.get("work_dir")
            final_message = _build_dataset_followup_message(dataset_result, work_dir)
        else:
            question = _extract_latest_user_question(
                state.get("messages", []),
                state.get("objective", ""),
            )
            if question and _config_flag(state, "enable_rag", True):
                answer_text, rag_metadata = _generate_bio_rag_answer(question, state)
                analysis_notes["last_rag"] = {
                    **rag_metadata,
                    "answer": answer_text,
                }
                final_message = answer_text.strip()
            else:
                logger.warning("[Response] No user question detected; falling back to default prompt.")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("[Response] Structured answer generation failed: %s", exc)
        final_message = None

    if not final_message:
        response_prompt = (
            "You are a helpful biomedical assistant who can view the user's Q&A history and give professional answers to the "
            "user's questions. If the latest messages contain execution errors, propose concrete next steps."
        )
        conversation_history = list(state.get("messages", []))
        if not conversation_history or conversation_history[-1].type != "human":
            objective = state.get("objective")
            if objective:
                conversation_history.append(HumanMessage(content=objective))

        fallback_messages = [SystemMessage(content=response_prompt)] + conversation_history
        fallback_response = llm.invoke(fallback_messages)
        final_message = getattr(fallback_response, "content", str(fallback_response))
        if isinstance(final_message, list):
            final_message = "\n".join(str(part) for part in final_message)

    if analysis_notes.get("dataset_bio_qa"):
        if "继续提问" not in final_message:
            final_message = (
                final_message.rstrip()
                + "\n\n如果你还有其他生物信息学问题，请继续提问，我会结合知识库和对话历史继续解答。"
            )
    else:
        final_message = final_message.rstrip() + "\n\n如需进一步探讨，请继续提问。"

    state["messages"].append(AIMessage(content=final_message))
    state["next_step"] = "end"

    logger.info("[Response] Response content: %s", final_message)
    return state


async def general_planner(state: AgentState) -> AgentState:
    """
    Dynamically generate execution plans based on intent and available tools
    """
    logger.info("[General Planner] Start planning based on user intent and available tools")

    track_metrics = _config_flag(state, "track_metrics", True)
    if track_metrics:
        metrics = state.setdefault("metrics", {})
        metrics["planner_invocations"] = metrics.get("planner_invocations", 0) + 1

    planner_mode = _config_value(state, "planner_mode", "multi_agent")
    if planner_mode == "disabled":
        state["messages"].append(
            AIMessage(
                content="<PLAN_SKIPPED>Planner disabled by runtime configuration. Responding directly.</PLAN_SKIPPED>"
            )
        )
        state["plan"] = []
        state["next_step"] = "response"
        return state

    if planner_mode == "linear":
        include_dataset_qa = _config_flag(state, "enable_dataset_qa", True)
        plan_steps = _default_linear_plan(include_dataset_qa)
        state["plan"] = plan_steps
        state["messages"].append(
            AIMessage(
                content="<PLAN_GENERATED>\n"
                + json.dumps({"steps": plan_steps}, ensure_ascii=False, indent=2)
                + "\n</PLAN_GENERATED>"
            )
        )
        state["next_step"] = "general_executor" if plan_steps else "response"
        return state

    input_files = state.get("input_files", [])
    
    # Create a working directory
    if input_files and "input_file_info" in state:
        print(state["input_file_info"])
        first_file_info = state["input_file_info"][0]
        base_name = first_file_info["file_name"].replace('.h5ad', '')
        work_dir = create_analysis_work_dir(base_name)
        state["work_dir"] = str(work_dir)
        logger.info(f"[General Planner] Create a working directory {work_dir}")
    def tools_to_string(tools):
        result = ["Available tools:\n"]
        for i, tool in enumerate(tools, 1):
            
            name = tool.name
            description = getattr(tool, 'description', 'No description')
            
            lines = [
                f"{i}. Name: {name}",
                f"   Description: {description}"
            ]
            result.append("\n".join(lines))
        return "\n\n" + "\n".join(result) + "\n"
    
    tools_info = tools_to_string(tools)

    plan_prompt_template = """

For the previously analyzed user intent, come up with a simple step by step plan. 
Users may have multiple intentions, and a plan should be generated for each intention, taking into account the dependencies between different intentions. The result of the previous step may be the input for the next step
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Ensure that each task can be completed at the minimum task granularity, and the description of each step should be as detailed as possible - do not skip steps.

{tools_info}

To achieve this goal, you will be able to view the conversation history between users and yourself, as well as information on available tools.
IMPORTANT:
- You MUST generate a non-empty list of steps.
- Each step should map to one of the available tools.
- Steps must be in execution order.
- Do not skip any essential step.
"""
    max_attempts = 3
    plan_generation_success = False

    previous_errors_feedback = ""

    for attempt in range(max_attempts):
        try:
            plan_llm = llm.with_structured_output(Plan)

            plan_prompt =plan_prompt_template.format(
                tools_info=tools_info
            )

            # print(f"plan_prompt{plan_prompt} type:{type(plan_prompt)}")

            logger.info(f"[General Planner] Attempting to generate plan (Attempt {attempt + 1}/{max_attempts})")
            if attempt > 0 and previous_errors_feedback:
                feedback_prompt = f"""
IMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT (Attempt {attempt}):
The plan generated in the previous attempt had the following issues:
{previous_errors_feedback}

Please carefully review the feedback above and regenerate the plan, ensuring it adheres strictly to the required JSON format and content guidelines.
"""
                plan_prompt = plan_prompt + "\n\n" + feedback_prompt 
            
            state["messages"].append(HumanMessage(content=plan_prompt))

            plan_messages = state["messages"]
            plan = plan_llm.invoke(plan_messages)
            logger.info(f"[General Planner] Plan generation result: {plan}")

            if plan and isinstance(plan.steps, list) and len(plan.steps) > 0:
                # Check if steps are non-empty
                non_empty_steps = [step for step in plan.steps if step.strip()]
                if non_empty_steps:
                    plan_generation_success = True
                    state["plan"] = non_empty_steps
                    _prune_completed_plan(state)
                    current_plan = state.get("plan", [])
                    if not current_plan:
                        logger.info(
                            "[General Planner] All proposed steps already completed; skipping execution."
                        )
                        state["messages"].append(
                            AIMessage(
                                content=(
                                    "<PLAN_GENERATED>\n{\n  \"steps\": []\n}\n</PLAN_GENERATED>"
                                )
                            )
                        )
                        state["next_step"] = "response"
                        return state
                    logger.info("[General Planner] Plan generated successfully.")
                    previous_errors_feedback = ""
                    break
                else:
                    error_msg = "Plan generation produced an object but steps list is invalid or empty."
                    logger.warning(f"[General Planner] Plan generated but all steps are empty (Attempt {attempt + 1}).")
                    previous_errors_feedback = error_msg
            else:
                error_msg = "Plan generation produced an object but steps list is invalid or empty."
                logger.warning(f"[General Planner] Plan generation failed or produced empty plan (Attempt {attempt + 1}).")      
                previous_errors_feedback = error_msg
        except ValidationError as e: 
            error_details = str(e)
            logger.error(f"[General Planner] Pydantic validation error (Attempt {attempt + 1}): {error_details}")
            
            feedback_msg = f"ValidationError: The output JSON structure was incorrect. Details: {error_details}. "
            feedback_msg += "Please ensure your response is a valid JSON object with a 'steps' key that is an array of non-empty strings, matching the provided Pydantic model exactly."
            previous_errors_feedback = feedback_msg
            
        except OutputParserException as e:
            error_details = str(e)
            logger.error(f"[General Planner] Output parsing error (Attempt {attempt + 1}): {error_details}")
            
            feedback_msg = f"OutputParserException: Failed to parse the LLM output. Details: {error_details}. "
            feedback_msg += "Please make sure your response is a clean JSON object that strictly follows the specified format, with no extra text, markdown, or explanations."
            previous_errors_feedback = feedback_msg

        except Exception as e:
            error_type = type(e).__name__
            error_details = str(e)
            logger.error(f"[General Planner] Unexpected error (Attempt {attempt + 1}): {error_type}: {error_details}")
            
            feedback_msg = f"Unexpected Error ({error_type}): {error_details}. "
            if "timeout" in error_type.lower() or "timeout" in error_details.lower():
                feedback_msg += "The previous attempt timed out. Please try generating a more concise plan or break down complex steps further."
            else:
                feedback_msg += "An unexpected error occurred. Please try regenerating the plan, ensuring it's a valid JSON object."
            previous_errors_feedback = feedback_msg

    if plan_generation_success and state.get("plan"):
        plan_model = Plan(steps=state["plan"])
        plan_json_str = plan_model.model_dump_json(indent=2)
        plan_message_content = f"<PLAN_GENERATED>\n{plan_json_str}\n</PLAN_GENERATED>"
        state["messages"].append(AIMessage(content=plan_message_content))
        state["next_step"] = "general_executor"

    else:
        error_msg = f"Failed to generate a valid plan after {max_attempts} attempts. Final error feedback: {previous_errors_feedback}"
        logger.error(f"[General Planner] {error_msg}")
        state["messages"].append(AIMessage(content=f"<PLAN_ERROR>{error_msg}</PLAN_ERROR>"))
        state["next_step"] = "response" 
    
    return state

async def _handle_tool_calls(state: AgentState, decision_message: AIMessage, current_step: str) -> AgentState:
    """Handles the logic for executing one or more tool calls. Modifies state in place."""

    tool_outputs = []
    tool_call_state = False
    track_metrics = _config_flag(state, "track_metrics", True)
    metrics = state.setdefault("metrics", {}) if track_metrics else {}
    allow_tools = _config_flag(state, "allow_tool_execution", True)
    failure_cfg = _config_value(state, "failure_injection", None)
    rng: Optional[random.Random] = None
    if isinstance(failure_cfg, dict) and failure_cfg.get("rate"):
        seed = failure_cfg.get("seed")
        rng = state.get("_failure_rng")
        if rng is None:
            rng = random.Random(seed)
            state["_failure_rng"] = rng

    for tool_call in decision_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"Attempting to call tool: {tool_name} with args: {tool_args}")

        if not allow_tools:
            warning = f"Tool execution disabled by configuration. Skipping call to {tool_name}."
            tool_outputs.append(
                ToolMessage(
                    content=f"Error: {warning}",
                    name=tool_name or "unknown_tool",
                    tool_call_id=tool_call_id
                )
            )
            continue

        if tool_name not in tools_by_name:
            error_msg = f"Tool '{tool_name}' not found in available tools."
            print(error_msg)

            tool_outputs.append(
                ToolMessage(
                    content=f"Error: {error_msg}",
                    name = tool_name,
                    tool_call_id=tool_call_id
                )
            )
        else:
            try:
                if tool_name == "dataset_bio_qa" and not _config_flag(state, "enable_dataset_qa", True):
                    skip_msg = "Dataset knowledge retrieval disabled by configuration."
                    tool_outputs.append(
                        ToolMessage(
                            content=f"Error: {skip_msg}",
                            name=tool_name,
                            tool_call_id=tool_call_id,
                        )
                    )
                    continue

                if rng and tool_name in set(failure_cfg.get("tool_names", [])):
                    if rng.random() < float(failure_cfg.get("rate", 0.0)):
                        raise RuntimeError("Injected tool failure for robustness experiment")

                tool_to_call: BaseTool = tools_by_name[tool_name]

                if hasattr(tool_to_call, 'ainvoke'):
                    tool_result = await tool_to_call.ainvoke(tool_args)
                else:
                    tool_result = tool_to_call.invoke(tool_args)

                if track_metrics:
                    metrics["tool_calls"] = metrics.get("tool_calls", 0) + 1

                state.setdefault("tool_history", []).append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                    }
                )

                if tool_name == "dataset_bio_qa":
                    dataset_payload = _safe_json_loads(tool_result)
                    if dataset_payload:
                        state.setdefault("analysis_notes", {})["dataset_bio_qa"] = dataset_payload
                        if "work_dir" not in dataset_payload and state.get("work_dir"):
                            dataset_payload["work_dir"] = state.get("work_dir")

                _update_project_state_from_tool(state, tool_name, tool_args, tool_result)

                if isinstance(tool_result, (dict, list)):
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                else:
                    tool_result_str = str(tool_result)

                result_str = f"<excute>Tool {tool_name} call result: {tool_result_str}</excute>"
                print(f"Tool '{tool_name}' executed successfully.")

                if state.get("plan"):
                    state["plan"] = state["plan"][1:]
                else:
                    print("Warning: Attempted to advance plan but no steps remain.")

                tool_outputs.append(
                    ToolMessage(
                        content=result_str,
                        name=tool_name,
                        tool_call_id=tool_call_id
                    )
                )
                tool_call_state = True

            except Exception as e:
                error_msg = f"Error calling tool '{tool_name}': {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                if track_metrics:
                    metrics["tool_errors"] = metrics.get("tool_errors", 0) + 1
                tool_outputs.append(
                    ToolMessage(
                        content=f"Error: {error_msg}",
                        name=tool_name,
                        tool_call_id=tool_call_id
                    )
                )

    state["messages"].extend(tool_outputs)

    if tool_call_state :
        remaining_plan = state.get("plan", [])
        if remaining_plan:
            state["next_step"] = "general_executor"
        else:
            state["next_step"] = "response"    
    else:
        state["next_step"] = "response"
        
    return state

            
async def general_executor(state: AgentState) -> AgentState:
    """
    Universal executor: executes the current planned step, supports tool invocation and error handling
    """
    if _config_flag(state, "track_metrics", True):
        metrics = state.setdefault("metrics", {})
        metrics["executor_invocations"] = metrics.get("executor_invocations", 0) + 1
    plan = state.get("plan", [])
    if not plan:
        print("[General Executor] No remaining plan steps to execute.")
        state["messages"].append(AIMessage(content="<observation>No remaining plan steps to execute.</observation>"))
        state["next_step"] = "response"
        return state
    current_step = plan[0]

    dataset_id, dataset_entry = _get_active_dataset_entry(state)
    tool_name = _identify_tool_from_plan_step(str(current_step))
    if tool_name and dataset_entry and _is_tool_already_completed(dataset_entry, tool_name):
        logger.info(
            "[General Executor] Skipping completed step '%s' for dataset '%s'", tool_name, dataset_id
        )
        observation = (
            f"<observation>检测到步骤 '{tool_name}' 已在当前数据集上完成，本轮将自动跳过。</observation>"
        )
        state["messages"].append(AIMessage(content=observation))
        state["plan"] = plan[1:]
        if state["plan"]:
            state["next_step"] = "general_executor"
        else:
            state["execution_status"] = "completed"
            state["next_step"] = "response"
        return state

    messages = state.get("messages", [])

    formatted_history = "\n"

    if messages:
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        for i, msg in enumerate(recent_messages):
            content_preview = (msg.content[:500]+ "...")if msg.content and len(msg.content) > 500 else (msg.content or "")
            formatted_history += f" {i+1}. [{msg.type}]{content_preview}\n "
    else:
        formatted_history += " (No previous conversation history)\n"

    task_formatted = f"""
You are a task execution agent responsible for completing a specific step in a larger plan.
You can observe the contextual environment of this step, and you need to extract necessary parameters from the previous conversation history to complete the current task.
You cannot fabricate parameters for tool calls. Parameters must be included in the conversation history, otherwise you will be punished.
If there are no tools in the conversation history to call the required parameters, make a request to the user.

You need to complete this task in the current step : {current_step}

Here is the recent conversation history: This provides a complete background of the interactions to date.Read carefully to understand:
*What has already been done.
*What data or variables may already exist in the execution environment (from the previous'<execute>'block).
*Any specific instructions or feedback given by the user or observed from previous executions.

{formatted_history}

"""
    try:
        decision_message = llm_with_tools.invoke(task_formatted)
        print(f"LLM Decision Message: {decision_message}")

        if decision_message.tool_calls:
            concise_tool_call_message = AIMessage(
                content = json.dumps(decision_message.tool_calls, indent = 2),
                tool_calls = decision_message.tool_calls
            )

            state["messages"].append(concise_tool_call_message)
            state = await _handle_tool_calls(state, decision_message,current_step)

        else:
            response_content = decision_message.content.strip()

            if not response_content:
                error_msg = "LLM did not call any tool and provided no reponse content.You need to reanalyze the conversation history to complete the task"
                print(error_msg)
                state["messages"].append(AIMessage(content=f"<EXECUTION_ERROR>{error_msg}</EXECUTION_ERROR>"))
                state["next_step"] = "general_executor"

            print(f"LLM provided a direct response or requesr for input: {response_content}")

            observation_content = f"<observation>LLM response to task '{current_step}':{response_content}</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            state["next_step"] = "replanner"
            _ensure_replanner_route(state)

    except Exception as e:
        error_msg = f"An unexpected error occurred in execute_step: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        state["messages"].append(AIMessage(content=f"<EXECUTION_ERROR>{error_msg}</EXECUTION_ERROR>"))
        state["next_step"] = "replanner"
        _ensure_replanner_route(state)

    return state



async def intelligent_replanner(state: AgentState) -> AgentState:
    """
    Intelligent replanning: Analyze execution status, dynamically adjust plans or provide responses
    """
    logger.info("[Intelligent Replanner] Start re planning and analyzing")

    if not _config_flag(state, "enable_replanner", True):
        _ensure_replanner_route(state)
        return state

    if _config_flag(state, "track_metrics", True):
        metrics = state.setdefault("metrics", {})
        metrics["replanner_invocations"] = metrics.get("replanner_invocations", 0) + 1

    messages: List[BaseMessage] = state.get("messages", [])
    current_plan: List[str] = state.get("plan", [])
    current_step = current_plan[0] if current_plan else "Unknown or finished step"

    attempts = state.get("replan_attempts", 0)
    max_attempts = max(1, state.get("max_replan_attempts", 4))
    if attempts >= max_attempts:
        error_msg = (
            "Replanner exceeded maximum retry attempts. Escalating to direct response for user guidance."
        )
        logger.error("[Intelligent Replanner] %s", error_msg)
        state["messages"].append(AIMessage(content=f"<REPLAN_ERROR>{error_msg}</REPLAN_ERROR>"))
        state["execution_status"] = "failed"
        state["next_step"] = "response"
        return state

    state["replan_attempts"] = attempts + 1
    if "execution_status" not in state or state["execution_status"] == "failed":
        state["execution_status"] = "in_progress"

    formatted_full_history = "\n"
    if messages:
        recect_messages = messages[-10:] if len(messages) > 10 else messages
        for i, msg in enumerate(recect_messages):
            content_preview = (msg.content[:500]+ "...")if msg.content and len(msg.content) > 500 else (msg.content or "")
            formatted_full_history += f" {i+1}. [{msg.type}]{content_preview}\n "
    else:
        formatted_full_history += " (No previous conversation history)\n"

    replan_prompt = f"""
You are an intelligent replan and analysis agent. Your task is to evaluate why the executor has entered the replanning phase and determine the correct next action.
The executor was tasked with tperforming the following step in the plan: {current_step} 
It has now entered replanning.This can happen for several reasons:
1.The step was **successfully completed** (e.g., a tool was called and its result can be observed in the conversation history).
2.The executor **failed** or encountered an error.
3.The executor **requested specific information** from the user to proceed.
4. The **overall plan is flawed** and needs revision.

**Your Task:**
1.  Carefully review the **Full Conversation History** provided below.
2.  Determine the most likely reason the executor entered replanning by analyzing the recent messages:
    *   **Success Indicators**: Look for a sequence like:
        - An `AIMessage` containing a JSON list of tool calls (this is the executor's action).
        - Followed by one or more `ToolMessage`(s) containing the output/result of the tool execution.
        This sequence strongly indicates successful completion of a tool-based step.
    *   **Failure/Error Indicators**: Look for `<EXECUTION_ERROR>` messages or AI messages indicating problems.
    *   **Request for Input Indicators**: Look for AI messages that seem to be direct requests for user input (e.g., "Please provide...", "Could you specify...").
    *   **Plan Flaw Indicators**: Consider if the step itself or the plan's logic seems fundamentally problematic.
3.  Based on your analysis, decide the most appropriate next step.

**Your Response MUST be ONE of the following JSON formats ONLY:**

**Option 1: Step Completed Successfully**
If the analysis shows a tool was successfully called and its result was observed, indicating the current step is done.
```json
{{
  "action": "step_completed",
  "reasoning": "The executor successfully called the tool(s) for '{current_step}'. This is evidenced by the presence of an AIMessage with tool call details followed by a ToolMessage containing the execution result. The step is complete."
}}
Option 2: Request Information from User
If the executor cannot proceed because it needs specific information that only the user possesses.
{{
  "action": "request_user_input",
  "request_message": "Please provide the path to the .h5ad file you want to analyze.",
  "reasoning": "The executor failed because the 'load_h5ad_data' tool requires a 'file_path' argument. This argument was not found in the conversation history, indicating the user must provide it."
}}
Option 3: Generate a New Plan
If the current plan is fundamentally flawed, unachievable, or based on incorrect premises.
{{
  "action": "regenerate_plan",
  "new_plan": [
    "Clarify the user's objective and required inputs (e.g., dataset file path).",
    "Load the specified dataset using the provided file path.",
    "Perform initial data quality checks.",
    "Conduct the main analysis task as requested."
  ],
  "reasoning": "The original plan started with loading data, but the necessary file path was unknown. A new plan that begins with gathering this essential information is required for success."
}}
Option 4: Indicate Plan Can Continue (Less Common)
If after analysis, you conclude that the executor's issue was a misunderstanding or has been resolved, and the plan can proceed with the current state.
{{
  "action": "continue_plan",
  "reasoning": "The executor's request for input appears to be based on outdated information. The file path 'data/sample.h5ad' was provided earlier and should be usable."
}}
IMPORTANT INSTRUCTIONS:

Your entire response must be a single, valid JSON object matching one of the four formats above.
Do not include any other text, explanations, or markdown formatting outside the JSON.
Base your decision and all reasoning strictly on the provided Full Conversation History.
Be precise and factual in your reasoning.
For request_user_input, ensure the request_message is user-friendly and specific.
For regenerate_plan,
ensure the new_plan is a list of clear, discrete, and actionable steps.
Full Conversation History:
{formatted_full_history}
"""
    try:
        decision_message: AIMessage = llm.invoke(replan_prompt)
        decision_content = decision_message.content.strip()
        print(f"Replanner Decision: {decision_content}")

        decision_data, parse_error = _parse_replanner_json(decision_content)
        if decision_data is None:
            error_msg = (
                f"Failed to parse Replanner's decision after cleanup. Error: {parse_error or 'unknown error'}."
            )
            logger.error("[Intelligent Replanner] %s", error_msg)
            state["messages"].append(AIMessage(content=f"<REPLAN_ERROR>{error_msg}</REPLAN_ERROR>"))
            fallback_request = (
                "I'm unable to determine the next action. Could you clarify the required files or instructions to proceed?"
            )
            state["messages"].append(AIMessage(content=f"<USER_REQUEST>{fallback_request}</USER_REQUEST>"))
            state["execution_status"] = "failed"
            state["next_step"] = "response"
            return state

        action = decision_data.get("action")

        if action == "step_completed":
            reason = decision_data.get("reasoning", "No reasoning provided")
            print(f"[Intelligent Replanner] Step completed successfully: {reason}")

            observation_content = f"<observation>Replanner confirmed step'{current_step}' completion. Reason: {reason}.Advancing to next step.</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            if current_plan:
                updated_plan = current_plan[1:]
                state["plan"] = updated_plan
                print(f"Updated plan: {updated_plan}")
                if not updated_plan:
                    state["execution_status"] = "completed"
                    state["next_step"] = "response"
                else:
                    state["execution_status"] = "in_progress"
                    state["next_step"] = "general_executor"
            else:
                # In the unlikely event there was no active plan entry, finish gracefully.
                state["next_step"] = "response"
                state["execution_status"] = state.get("execution_status", "completed") or "completed"
            state["replan_attempts"] = 0

        elif action == "request_user_input":
            request_message = decision_data.get("request_message", "Could you please provide more information?")
            reason = decision_data.get("reasoning", "No reasoning provided")
            print(f"[Intelligent Replanner] Requesting user input: {request_message} (Reason: {reason})")

            state["messages"].append(AIMessage(content=f"<USER_REQUEST>{request_message}</USER_REQUEST>"))

            user_input = input(f"f{request_message}")
            state["messages"].append(AIMessage(content=f"<USER_INPUT>{user_input}</USER_INPUT>"))
            print(f"User input received: {user_input}")
            state["next_step"] = "general_executor"
            state["execution_status"] = "in_progress"
            state["replan_attempts"] = 0

        elif action == "regenerate_plan":
            new_plan = decision_data.get("new_plan", [])
            reasoning = decision_data.get("reasoning", "No reasoning provided")
            print(f"[Intelligent Replanner] Regenerating plan: {new_plan} (Reason: {reasoning})")

            if not isinstance(new_plan, list) or not new_plan:
                error_msg = "Replanner generated an invalid or empty new plan."
                print(f"[Intelligent Replanner] {error_msg}")
                state["messages"].append(AIMessage(content=f"<PLAN_ERROR>{error_msg}</PLAN_ERROR>"))
                state["next_step"] = "response"
                state["execution_status"] = "failed"
                return state
            if _config_flag(state, "track_metrics", True):
                metrics = state.setdefault("metrics", {})
                metrics["plan_regenerations"] = metrics.get("plan_regenerations", 0) + 1
            plan_summary = "\n".join([f"{i+1}. {step}" for i, step in enumerate(new_plan)])
            observation_content = f"<observation>Replanner generated a new plan. Reasoning: {reasoning}\nNew Plan:\n{plan_summary}</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            state["plan"] = new_plan
            _prune_completed_plan(state)
            pruned_plan = state.get("plan", [])
            if not pruned_plan:
                state["messages"].append(
                    AIMessage(
                        content=(
                            "<observation>检测到重规划后的所有步骤均已在当前数据集上完成，直接进入总结响应。</observation>"
                        )
                    )
                )
                state["execution_status"] = "completed"
                state["next_step"] = "response"
            else:
                state["next_step"] = "general_executor"
                state["execution_status"] = "in_progress"
            state["replan_attempts"] = 0
        elif action == "continue_plan":
            reasoning = decision_data.get("reasoning", "")
            print(f"Replanner decided plan can continue: {reasoning}")

            observation_content = f"<observation>Replanner assessed the situation: {reasoning}. Decided to continue with the current plan.</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            state["next_step"] = "general_executor"
            state["execution_status"] = "in_progress"
            state["replan_attempts"] = 0
        else:
            error_msg = f"Replanner returned an invalid action: {action}"
            print(error_msg)
            state["messages"].append(AIMessage(content=f"<REPLAN_ERROR>{error_msg}</REPLAN_ERROR>"))
            state["next_step"] = "replanner"

    except Exception as e:
        error_msg = f"An unexpected error occurred in replanner: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        state["messages"].append(AIMessage(content=f"<REPLAN_ERROR>{error_msg}</REPLAN_ERROR>"))
        state["next_step"] = "response"
        state["execution_status"] = "failed"
    return state


def route_after_intent(state: AgentState) -> str:
    """"""
    if state["next_step"]=="planner":
        return "planner"  
    else:
        return "response" 


def route_after_replan(state: AgentState) -> str:
    
    if state["execution_status"] == "completed":
        return END
    elif state["execution_status"] == "failed":
        return END
    else:
        return "general_executor"
def build_graph():

    g = StateGraph(AgentState)

    g.add_node("intent_recognition", intent_recognition)
    g.add_node("response", response) # Note: Typo in function name 'response'
    g.add_node("general_planner", general_planner)
    g.add_node("general_executor", general_executor)
    g.add_node("intelligent_replanner", intelligent_replanner)
    
    
    g.add_edge(START, "intent_recognition")
    g.add_conditional_edges(
        "intent_recognition",
        route_after_intent,
        {
            "planner": "general_planner",
            "response": "response",
        }
    )
    g.add_conditional_edges(
        "general_planner",
        lambda state: state["next_step"],
        {
            "general_executor": "general_executor",
            "response": "response",
            
        }
    )
    g.add_conditional_edges(
        "general_executor",
        lambda state: state["next_step"],
        {
            "general_executor": "general_executor", # For retries or continuing plan
            "replanner": "intelligent_replanner",
            "response": "response",
        }
    )

    g.add_conditional_edges(
        "intelligent_replanner",
        lambda state: state["next_step"],
        {
            "general_executor": "general_executor", # If step completed or plan continues
            "response": "response", # If replan fails or decides to end with response
            # Add "replanner" if you want to allow replan -> replan (e.g., on error)
            "replanner": "intelligent_replanner", 
        }
    )
    g.add_edge("response", END)

    return g.compile(checkpointer=memory)


def create_initial_state(
    objective: str,
    input_files: Optional[List[str]],
    thread_id: Optional[str],
    runtime_config: Optional[AgentRuntimeConfig] = None,
) -> Tuple[AgentState, str]:
    config = runtime_config or AgentRuntimeConfig()
    resolved_thread_id = (thread_id or str(uuid4())).strip() or str(uuid4())

    if config.enable_memory:
        memory_context = conversation_memory.load_context(
            thread_id=resolved_thread_id, objective=objective
        )
        memory_messages = conversation_memory.build_context_messages(memory_context)
        project_state = (
            json.loads(json.dumps(memory_context.project_state))
            if memory_context.project_state
            else {}
        )
        project_message = build_project_state_message(project_state)
        if project_message:
            memory_messages.insert(0, SystemMessage(content=project_message))
    else:
        memory_context = MemoryContext()
        memory_messages: List[SystemMessage] = []
        project_state = {}

    state: AgentState = {
        "objective": objective,
        "messages": list(memory_messages),
        "input_files": input_files or [],
        "intents": [],
        "plan": [],
        "next_step": None,
        "memory_summary": memory_context.summary,
        "memory_records": [record.__dict__ for record in getattr(memory_context, "records", [])],
        "thread_id": resolved_thread_id,
        "replan_attempts": 0,
        "max_replan_attempts": max(1, config.max_replan_attempts),
        "execution_status": "in_progress",
        "intent_trace": {},
        "work_dir": None,
        "tool_history": [],
        "analysis_notes": {},
        "recognized_intents": [],
        "project_state": project_state,
        "runtime_config": config.to_dict(),
    }

    if config.track_metrics:
        state["metrics"] = {}

    if project_state:
        datasets = project_state.get("datasets", {})
        active_dataset = project_state.get("active_dataset") or project_state.get("last_dataset")
        if active_dataset and isinstance(datasets.get(active_dataset), dict):
            active_entry = datasets[active_dataset]
            work_dir = active_entry.get("work_dir")
            if work_dir:
                state["work_dir"] = work_dir
            saved_inputs = active_entry.get("input_files")
            if saved_inputs and not state["input_files"]:
                if isinstance(saved_inputs, list):
                    state["input_files"] = list(saved_inputs)

    return state, resolved_thread_id


async def run_objective(
    objective: str,
    input_files: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
    runtime_config: Optional[AgentRuntimeConfig] = None,
    return_diagnostics: bool = False,
):
    """运行通用智能体"""
    config = runtime_config or AgentRuntimeConfig()
    logger.info(f"[Agent] 开始执行任务: {objective}")
    if input_files:
        logger.info(f"[Agent] 输入文件: {input_files}")


    initial_state, resolved_thread_id = create_initial_state(
        objective, input_files, thread_id, config
    )

    graph = build_graph()
    final_output: Optional[Any] = None
    final_state: Optional[AgentState] = None

    async for event in graph.astream(
        initial_state,
        config={"recursion_limit": 50, "configurable": {"thread_id": resolved_thread_id}}
    ):
        should_break = False
        for k, v in event.items():
            if k == "__end__":
                continue

            final_state = v

            if k == "response" and v:
                logger.info(f"[Agent] 任务完成: {v}")
                final_output = v["messages"][-1]
                should_break = True
                break
            if k == "error_info" and v:
                logger.error(f"[Agent] 任务失败: {v}")
                final_output = f"任务执行失败: {v}"
                should_break = True
                break
        if should_break:
            break

    final_text: Optional[str] = None
    if isinstance(final_output, BaseMessage):
        final_text = getattr(final_output, "content", None)
    elif final_output is not None:
        final_text = str(final_output)

    if final_state is not None and config.enable_memory:
        messages = final_state.get("messages", [])
        conversation_memory.store_conversation(
            thread_id=resolved_thread_id,
            objective=objective,
            messages=messages,
            result_text=final_text,
            metadata={
                "input_files": input_files or [],
                "project_state": serialise_project_state(final_state.get("project_state", {})),
            },
        )

    if return_diagnostics:
        return (final_text or "任务执行完成", final_state)

    if final_text is not None:
        return final_text

    return "任务执行完成"

if __name__ == "__main__":
    import asyncio
    
    # 测试用例
    test_objectives = [
        # "对scRNA-seq数据进行预处理",
        "Annotate cell types on scRNA seq data",
        # "推断scRNA-seq数据的调控网络",
        # "对scRNA-seq数据进行通路富集分析",
        # "对scRNA-seq数据进行细胞注释并做通路分析"
    ]
    
    for i, objective in enumerate(test_objectives):
        print(f"\n=== 测试用例 {i+1} ===")
        print(f"输入: {objective}")
        files = [
            "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/test_l3_stratified_5pct.h5ad"
        ]
        result = asyncio.run(run_objective(objective, files))
        print(f"结果: {result}")
        print("=" * 50)