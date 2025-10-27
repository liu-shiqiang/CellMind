import re
import json
import operator
from typing import Annotated, List, Tuple, Union, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from typing import Literal
import logging
import structlog

from langchain import hub
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.exceptions import OutputParserException
from langgraph.graph import END
from langgraph.graph import StateGraph, START 
from langgraph.checkpoint.memory import MemorySaver


from src.agent.tool_registry import TOOLS
from src.utils.llm_manager import LLMManager
from src.utils.path_manager import path_manager, extract_paths_from_objective, validate_h5ad_file, create_analysis_work_dir
from src.memory.conversation_memory import ConversationMemoryStore


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
    state["next_step"] = "planner" if result.is_task else "response"
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

llm = ChatOllama(model="qwen3:30b", 
                 temperature=0.6,
                 base_url="http://localhost:11434"
                 )

llm_tool = ChatOllama(model="qwen3:30b", 
                 temperature=0.6,
                 base_url="http://localhost:11434"
                 )

llm_with_tools = llm_tool.bind_tools(tools)

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

    response_prompt = """
You are a helpful biomedical assistant who can view the user's Q&A history and give professional answers to the user's questions.

You can observe the environment and identify potential errors that may occur during program execution. When you notice that the most recent message in history is a program error, provide users with possible solutions

Requirement: You must give an answer.

"""
    
    conversation_history = list(state.get("messages", []))
    if not conversation_history or conversation_history[-1].type != "human":
        objective = state.get("objective")
        if objective:
            conversation_history.append(HumanMessage(content=objective))

    message = [SystemMessage(content = response_prompt)] + conversation_history
    response = llm.invoke(message)

    logger.info(f"[Response] Response content: {response.content}")

    state["messages"].append(AIMessage(content=response.content))

    state["next_step"] = "end"

    return state


async def general_planner(state: AgentState) -> AgentState:
    """
    Dynamically generate execution plans based on intent and available tools
    """
    logger.info("[General Planner] Start planning based on user intent and available tools")
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

    if plan_generation_success and plan:
        plan_json_str = plan.model_dump_json(indent = 2)
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

    for tool_call in decision_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"Attempting to call tool: {tool_name} with args: {tool_args}")

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
                tool_to_call: BaseTool = tools_by_name[tool_name]
                
                if hasattr(tool_to_call, 'ainvoke'):
                    tool_result = await tool_to_call.ainvoke(tool_args)
                else:
                    tool_result = tool_to_call.invoke(tool_args)

                result_str = f"<excute>Tool {tool_name} call result: {str(tool_result)}</excute>"
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
    plan = state.get("plan", [])
    if not plan:
        print("[General Executor] No remaining plan steps to execute.")
        state["messages"].append(AIMessage(content="<observation>No remaining plan steps to execute.</observation>"))
        state["next_step"] = "response"
        return state
    current_step = plan[0]
    
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
            
    except Exception as e:
        error_msg = f"An unexpected error occurred in execute_step: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        state["messages"].append(AIMessage(content=f"<EXECUTION_ERROR>{error_msg}</EXECUTION_ERROR>"))
        state["next_step"] = "replanner"
        
    return state



async def intelligent_replanner(state: AgentState) -> AgentState:
    """
    Intelligent replanning: Analyze execution status, dynamically adjust plans or provide responses
    """
    logger.info("[Intelligent Replanner] Start re planning and analyzing")
    
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
                else:
                    state["execution_status"] = "in_progress"
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
            plan_summary = "\n".join([f"{i+1}. {step}" for i, step in enumerate(new_plan)])
            observation_content = f"<observation>Replanner generated a new plan. Reasoning: {reasoning}\nNew Plan:\n{plan_summary}</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            state["plan"] = new_plan
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


async def run_objective(objective: str, input_files: Optional[List[str]] = None):
    """运行通用智能体"""
    logger.info(f"[Agent] 开始执行任务: {objective}")
    if input_files:
        logger.info(f"[Agent] 输入文件: {input_files}")
    

    thread_id = "1"
    memory_context = conversation_memory.load_context(thread_id=thread_id, objective=objective)
    memory_messages = conversation_memory.build_context_messages(memory_context)

    initial_state = AgentState(
        objective=objective,
        messages=list(memory_messages),
        input_files=input_files or [],
        intents=[],
        plan=[],
        next_step=None,
        memory_summary=memory_context.summary,
        memory_records=[record.__dict__ for record in memory_context.records],
        thread_id=thread_id,
        replan_attempts=0,
        max_replan_attempts=4,
        execution_status="in_progress",
        intent_trace={},

    )

    graph = build_graph()
    final_output: Optional[Any] = None
    final_state: Optional[AgentState] = None

    async for event in graph.astream(
        initial_state,
        config={"recursion_limit": 50, "configurable": {"thread_id": thread_id}}
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

    if final_state is not None:
        messages = final_state.get("messages", [])
        result_text: Optional[str]
        if isinstance(final_output, BaseMessage):
            result_text = getattr(final_output, "content", None)
        else:
            result_text = str(final_output) if final_output is not None else None

        conversation_memory.store_conversation(
            thread_id=thread_id,
            objective=objective,
            messages=messages,
            result_text=result_text,
            metadata={"input_files": input_files or []},
        )

    if isinstance(final_output, BaseMessage):
        return final_output
    if final_output is not None:
        return final_output

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