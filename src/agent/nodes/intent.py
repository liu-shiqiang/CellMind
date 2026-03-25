"""
意图识别节点
分析用户输入，识别是否需要执行分析任务
包含完整的LLM意图识别和规则匹配
"""
import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError, ConfigDict
from pydantic import Field as PydanticField

from src.agent.state import AgentState, Intent, IntentResponse
from src.utils.llm_manager import get_llm

logger = logging.getLogger(__name__)

# ============== 意图识别配置 ==============

# 允许的意图类型
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

# 非任务意图（不需要执行工具）
NON_TASK_INTENTS = {"memory_query", "status_check", "direct_response", "clarification", "greeting", "chitchat"}

# 最小任务置信度
MIN_TASK_CONFIDENCE = 0.55

# 意图同义词映射
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

# 默认意图描述
DEFAULT_INTENT_DESCRIPTIONS: Dict[str, str] = {
    "memory_query": "从长期记忆中检索先前的对话上下文或任务。",
    "status_check": "报告当前任务的状态或进度。",
    "direct_response": "直接回答用户，不执行分析工具。",
    "clarification": "提出澄清问题以更好地理解请求。",
    "greeting": "处理对话问候，不进行分析。",
    "chitchat": "进行闲聊，不触发工具。",
}

# 记忆查询模式
MEMORY_QUERY_PATTERNS = [
    r"what\s+did\s+i\s+ask\s+(you\s+)?(before|last\s+time)",
    r"remember\s+(our|the)\s+(last|previous)\s+(conversation|task)",
    r"what\s+was\s+my\s+previous\s+(request|question|mission)",
    r"recall\s+(?:the\s+)?(earlier|previous)\s+(?:instructions|task)",
]

# 状态查询模式
STATUS_QUERY_PATTERNS = [
    r"how\s+is\s+(the\s+)?(analysis|task|job|mission)\s+(going|progressing)",
    r"what\s+is\s+the\s+status\s+of",
    r"give\s+me\s+an\s+update",
    r"have\s+you\s+finished",
]


# ============== 辅助函数 ==============

def _looks_like_memory_query(text: str) -> bool:
    """启发式检测是否为记忆查询"""
    if not text:
        return False

    lowered = text.lower()
    for pattern in MEMORY_QUERY_PATTERNS:
        if re.search(pattern, lowered):
            return True

    keywords = {"remember", "memory", "recall", "previous", "last time"}
    return any(keyword in lowered for keyword in keywords)


def _looks_like_status_query(text: str) -> bool:
    """启发式检测是否为状态查询"""
    if not text:
        return False

    lowered = text.lower()
    for pattern in STATUS_QUERY_PATTERNS:
        if re.search(pattern, lowered):
            return True

    keywords = {"status", "progress", "update", "finished"}
    return any(keyword in lowered for keyword in keywords)


def _normalize_intent_label(raw_label: Optional[str]) -> str:
    """标准化意图标签"""
    if not raw_label:
        return "generic"

    cleaned = raw_label.strip().lower()
    cleaned = INTENT_SYNONYMS.get(cleaned, cleaned)

    if cleaned not in ALLOWED_INTENT_TYPES:
        return "generic"

    return cleaned


def _is_task_intent(intent: Intent) -> bool:
    """判断是否为任务意图"""
    normalized = _normalize_intent_label(intent.intent)
    if normalized in NON_TASK_INTENTS:
        return False
    return intent.confidence >= MIN_TASK_CONFIDENCE


def _sanitize_intent(intent: Intent) -> Intent:
    """清理意图对象"""
    intent_dict = intent.model_dump()
    normalized_label = _normalize_intent_label(intent_dict.get("intent", ""))
    description = intent_dict.get("description") or DEFAULT_INTENT_DESCRIPTIONS.get(normalized_label, "")
    justification = intent_dict.get("justification") or description or ""
    dependencies = intent_dict.get("dependencies") or []

    return Intent(
        intent=normalized_label,
        description=description,
        confidence=intent_dict.get("confidence", 0.0),
        dependencies=list(dependencies),
        justification=justification,
    )


def _postprocess_intent_response(response: IntentResponse) -> IntentResponse:
    """后处理意图识别响应"""
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


def _fallback_direct_response(reason: str) -> IntentResponse:
    """回退到直接响应"""
    logger.info("[Intent Recognition] Falling back to direct response: %s", reason)
    fallback_intent = Intent(
        intent="direct_response",
        description=DEFAULT_INTENT_DESCRIPTIONS["direct_response"],
        confidence=0.0,
        dependencies=[],
        justification=f"Fallback because structured intent parsing failed: {reason}",
    )
    return IntentResponse(intents=[fallback_intent], is_task=False)


def _rule_based_intent(objective: str) -> Optional[IntentResponse]:
    """基于规则的意图识别"""
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


def _append_unique_human_message(state: AgentState, content: str) -> None:
    """添加唯一的用户消息"""
    if not content:
        return

    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage) and last_message.content == content:
            return

    state["messages"].append(HumanMessage(content=content))


def _apply_intent_result(
    state: AgentState,
    result: IntentResponse,
    *,
    source: str,
    rationale: str,
    raw_response: Optional[Any] = None,
) -> None:
    """应用意图识别结果到状态"""
    task_intents = [intent for intent in result.intents if _is_task_intent(intent)]

    if result.is_task and not task_intents:
        result.is_task = False

    state["intents"] = task_intents if result.is_task else []
    state["recognized_intents"] = [intent.model_dump() for intent in result.intents]
    state["next_step"] = "planner" if result.is_task else "response"
    state["intent_trace"] = {
        "objective": state.get("objective"),
        "source": source,
        "rationale": rationale,
        "is_task": result.is_task,
        "task_intents": [intent.model_dump() for intent in task_intents],
        "classified_intents": [intent.model_dump() for intent in result.intents],
        "raw_response": raw_response.model_dump() if hasattr(raw_response, "model_dump") else raw_response,
    }
    logger.info("[Intent Recognition] Final decision: %s", state["intent_trace"])


# ============== 意图识别节点 ==============

async def intent_recognition(state: AgentState) -> AgentState:
    """
    通用意图识别节点
    使用LLM动态识别用户意图，支持多意图分析

    处理流程:
    1. 检查输入文件，添加数据集消息
    2. 尝试规则匹配（记忆查询、状态查询）
    3. 使用LLM进行结构化意图识别
    4. 后处理和验证识别结果
    """
    objective = state["objective"]
    input_files = state.get("input_files", [])

    # 添加数据集可用性消息
    if input_files:
        dataset_message = f"User dataset available at: {input_files[0]}"
        if not state["messages"] or not (
            isinstance(state["messages"][0], HumanMessage)
            and state["messages"][0].content == dataset_message
        ):
            state["messages"].insert(0, HumanMessage(content=dataset_message))

    # 添加唯一的用户消息
    _append_unique_human_message(state, objective)

    # 1. 首先尝试规则匹配
    rule_based = _rule_based_intent(objective)
    if rule_based:
        rationale = rule_based.intents[0].justification if rule_based.intents else "Rule-based classification"
        _apply_intent_result(state, rule_based, source="rule", rationale=rationale, raw_response=rule_based)
        return state

    # 2. 使用LLM进行意图识别
    llm = get_llm()
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
        # 使用 JSON 响应格式（兼容智谱 API）
        json_prompt = intent_prompt + """

IMPORTANT: You must respond ONLY with a valid JSON object. Do not include any explanatory text before or after the JSON.
Example format:
{
  "intents": [{"intent": "clustering_analysis", "description": "...", "confidence": 0.9, "dependencies": [], "justification": "..."}],
  "is_task": true
}
"""
        # GLM API 对 SystemMessage 支持不完善，使用 HumanMessage 代替
        messages = [HumanMessage(content=json_prompt)] + state["messages"]

        # 直接调用 LLM 获取 JSON 响应
        response = llm.invoke(messages)
        response_text = getattr(response, "content", str(response)).strip()

        # 尝试提取 JSON（处理可能的 markdown 代码块）
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1).strip()
        elif response_text.startswith("```"):
            response_text = response_text.strip("`").replace("json", "").strip()

        logger.info("[Intent Recognition] Raw LLM response: %s", response_text[:500])

        # 解析 JSON
        parsed = json.loads(response_text)

        # 构建 IntentResponse 对象
        intents_data = parsed.get("intents", [])
        intents = []
        for intent_data in intents_data:
            intents.append(Intent(**intent_data))

        raw_result = IntentResponse(intents=intents, is_task=parsed.get("is_task", False))
        logger.info("[Intent Recognition] Parsed result: %s", raw_result)

        # 后处理结果
        processed_result = _postprocess_intent_response(raw_result)
        _apply_intent_result(
            state,
            processed_result,
            source="llm",
            rationale="JSON intent classification",
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


# ============== 辅助功能（供其他节点使用）==============

def get_recognized_intent_labels(state: AgentState) -> List[str]:
    """返回当前轮次识别到的意图标签列表"""
    raw_intents = state.get("recognized_intents", []) or []
    labels: List[str] = []

    for item in raw_intents:
        if isinstance(item, Intent):
            labels.append(item.intent)
        elif isinstance(item, dict):
            labels.append(_normalize_intent_label(item.get("intent", "")))
        elif hasattr(item, "intent"):
            labels.append(_normalize_intent_label(getattr(item, "intent", "")))

    return labels
