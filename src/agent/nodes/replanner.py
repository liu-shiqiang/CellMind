"""
智能重规划节点
分析执行状态，动态调整计划
包含完整的决策逻辑和JSON解析
"""
import re
import json
import logging
from typing import Optional, Dict, Any

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.utils.llm_manager import get_llm

logger = logging.getLogger(__name__)


# ============== JSON解析辅助函数 ==============

def _clean_replanner_output(content: str) -> str:
    """移除LLM输出中常见的包装格式"""
    if not content:
        return ""

    cleaned = content.strip()
    # 移除markdown代码块
    cleaned = re.sub(r"<resolved>.*?</resolved>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()


def _parse_replanner_json(raw_content: str) -> tuple[dict | None, str | None]:
    """尝试解析重规划器的JSON响应"""
    candidates: list[str] = []

    if raw_content and raw_content.strip():
        candidates.append(raw_content.strip())

    cleaned = _clean_replanner_output(raw_content)
    if cleaned and cleaned not in candidates:
        candidates.append(cleaned)

    # 查找JSON对象
    brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0).strip()
        if candidate not in candidates:
            candidates.append(candidate)

    last_error: str | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as err:
            last_error = str(err)

    return None, last_error


# ============== 智能重规划节点 ==============

async def intelligent_replanner(state: AgentState) -> AgentState:
    """
    智能重规划节点
    分析执行状态，动态调整计划或提供响应

    处理流程:
    1. 检查重试次数限制
    2. 分析对话历史判断执行结果
    3. 根据分析结果决定下一步行动
    4. 处理四种情况: 步骤完成、请求用户输入、重新生成计划、继续执行
    """
    logger.info("[Intelligent Replanner] Starting replanning and analysis")

    messages = state.get("messages", [])
    current_plan = state.get("plan", [])
    current_step = current_plan[0] if current_plan else "Unknown or finished step"

    # 检查重试次数
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

    # 格式化对话历史
    formatted_full_history = "\n"
    if messages:
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        for i, msg in enumerate(recent_messages):
            content_preview = (msg.content[:500] + "...") if msg.content and len(msg.content) > 500 else (msg.content or "")
            formatted_full_history += f" {i+1}. [{msg.type}]{content_preview}\n"
    else:
        formatted_full_history += " (No previous conversation history)\n"

    # 重规划提示词
    replan_prompt = f"""
You are an intelligent replan and analysis agent. Your task is to evaluate why the executor has entered the replanning phase and determine the correct next action.
The executor was tasked with performing the following step in the plan: {current_step}
It has now entered replanning. This can happen for several reasons:
1. The step was **successfully completed** (e.g., a tool was called and its result can be observed in the conversation history).
2. The executor **failed** or encountered an error.
3. The executor **requested specific information** from the user to proceed.
4. The **overall plan is flawed** and needs revision.

**Your Task:**
1. Carefully review the **Full Conversation History** provided below.
2. Determine the most likely reason the executor entered replanning by analyzing the recent messages:
    * **Success Indicators**: Look for a sequence like:
        - An `AIMessage` containing a JSON list of tool calls (this is the executor's action).
        - Followed by one or more `ToolMessage`(s) containing the output/result of the tool execution.
        This sequence strongly indicates successful completion of a tool-based step.
    * **Failure/Error Indicators**: Look for `<EXECUTION_ERROR>` messages or AI messages indicating problems.
    * **Request for Input Indicators**: Look for AI messages that seem to be direct requests for user input (e.g., "Please provide...", "Could you specify...").
    * **Plan Flaw Indicators**: Consider if the step itself or the plan's logic seems fundamentally problematic.
3. Based on your analysis, decide the most appropriate next step.

**Your Response MUST be ONE of the following JSON formats ONLY:**

**Option 1: Step Completed Successfully**
If the analysis shows a tool was successfully called and its result was observed, indicating the current step is done.
```json
{{
  "action": "step_completed",
  "reasoning": "The executor successfully called the tool(s) for '{current_step}'. This is evidenced by the presence of an AIMessage with tool call details followed by a ToolMessage containing the execution result. The step is complete."
}}
```
**Option 2: Request Information from User**
If the executor cannot proceed because it needs specific information that only the user possesses.
```json
{{
  "action": "request_user_input",
  "request_message": "Please provide the path to the .h5ad file you want to analyze.",
  "reasoning": "The executor failed because the 'load_h5ad_data' tool requires a 'file_path' argument. This argument was not found in the conversation history, indicating the user must provide it."
}}
```
**Option 3: Generate a New Plan**
If the current plan is fundamentally flawed, unachievable, or based on incorrect premises.
```json
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
```
**Option 4: Indicate Plan Can Continue (Less Common)**
If after analysis, you conclude that the executor's issue was a misunderstanding or has been resolved, and the plan can proceed with the current state.
```json
{{
  "action": "continue_plan",
  "reasoning": "The executor's request for input appears to be based on outdated information. The file path 'data/sample.h5ad' was provided earlier and should be usable."
}}
```
IMPORTANT INSTRUCTIONS:

Your entire response must be a single, valid JSON object matching one of the four formats above.
Do not include any other text, explanations, or markdown formatting outside the JSON.
Base your decision and all reasoning strictly on the provided Full Conversation History.
Be precise and factual in your reasoning.
For request_user_input, ensure the request_message is user-friendly and specific.
For regenerate_plan, ensure the new_plan is a list of clear, discrete, and actionable steps.

Full Conversation History:
{formatted_full_history}
"""

    try:
        llm = get_llm()
        decision_message: AIMessage = llm.invoke(replan_prompt)
        decision_content = decision_message.content.strip()
        print(f"Replanner Decision: {decision_content}")

        # 解析JSON响应
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
            # 步骤成功完成
            reason = decision_data.get("reasoning", "No reasoning provided")
            print(f"[Intelligent Replanner] Step completed successfully: {reason}")

            observation_content = f"<observation>Replanner confirmed step '{current_step}' completion. Reason: {reason}. Advancing to next step.</observation>"
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
                state["next_step"] = "response"
                state["execution_status"] = state.get("execution_status", "completed") or "completed"
            state["replan_attempts"] = 0

        elif action == "request_user_input":
            # 请求用户输入
            request_message = decision_data.get("request_message", "Could you please provide more information?")
            reason = decision_data.get("reasoning", "No reasoning provided")
            print(f"[Intelligent Replanner] Requesting user input: {request_message} (Reason: {reason})")

            # 在Web环境中，不能使用input()阻塞等待
            state["messages"].append(AIMessage(content=f"<USER_REQUEST>{request_message}</USER_REQUEST>"))
            state["next_step"] = "response"
            state["execution_status"] = "waiting_for_input"
            state["replan_attempts"] = 0

        elif action == "regenerate_plan":
            # 重新生成计划
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

            # 剪除已完成的步骤
            from src.agent.nodes.planner import prune_completed_steps
            prune_completed_steps(state)
            pruned_plan = state.get("plan", [])

            if not pruned_plan:
                state["messages"].append(
                    AIMessage(
                        content="<observation>检测到重规划后的所有步骤均已在当前数据集上完成，直接进入总结响应。</observation>"
                    )
                )
                state["execution_status"] = "completed"
                state["next_step"] = "response"
            else:
                state["next_step"] = "general_executor"
                state["execution_status"] = "in_progress"
            state["replan_attempts"] = 0

        elif action == "continue_plan":
            # 继续执行当前计划
            reasoning = decision_data.get("reasoning", "")
            print(f"Replanner decided plan can continue: {reasoning}")

            observation_content = f"<observation>Replanner assessed the situation: {reasoning}. Decided to continue with the current plan.</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            state["next_step"] = "general_executor"
            state["execution_status"] = "in_progress"
            state["replan_attempts"] = 0

        else:
            # 无效的操作
            error_msg = f"Replanner returned an invalid action: {action}"
            print(error_msg)
            state["messages"].append(AIMessage(content=f"<REPLAN_ERROR>{error_msg}</REPLAN_ERROR>"))
            state["next_step"] = "intelligent_replanner"

    except Exception as e:
        error_msg = f"An unexpected error occurred in replanner: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        state["messages"].append(AIMessage(content=f"<REPLAN_ERROR>{error_msg}</REPLAN_ERROR>"))
        state["next_step"] = "response"
        state["execution_status"] = "failed"

    return state
