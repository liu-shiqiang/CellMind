"""
工具执行节点
执行当前计划步骤，调用相应工具
包含完整的工具调用处理、缓存和错误处理
"""
import json
import logging
import time
from typing import Dict, Any, List
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import BaseTool

from src.agent.state import AgentState
from src.agent.tool_registry import TOOLS
from src.utils.llm_manager import get_llm_with_tools
from src.utils.result_persistence import get_persistence_manager
from src.agent.nodes.planner import identify_tool_from_step, is_tool_completed

logger = logging.getLogger(__name__)

# 全局工具映射
tools_by_name = {tool.name: tool for tool in TOOLS}

FILE_PATH_TOOLS = {
    "load_h5ad_data",
    "calculate_qc_metrics",
    "normalize_and_hvg",
    "pca_reduction",
    "cluster_and_umap",
    "find_marker_genes",
    "annotate_cells",
    "differential_expression",
    "generate_analysis_report",
    "extract_embeddings_with_scgpt",
    "run_cellphonedb_core",
    "run_pseudotime_analysis",
    "run_ora_enrichment",
    "run_ssgsea_enrichment",
}

WORK_DIR_TOOLS = {
    "run_pseudotime_analysis",
    "cluster_and_diff",
    "interpret_cluster_results",
}


# ============== 辅助函数 ==============

def _get_active_dataset_entry(state: AgentState) -> tuple[str | None, dict | None]:
    """获取当前活动的数据集条目"""
    project_state = state.get("project_state") or {}
    datasets = project_state.get("datasets") or {}
    dataset_id = project_state.get("active_dataset") or project_state.get("last_dataset")
    entry: dict | None = None
    if dataset_id and isinstance(datasets.get(dataset_id), dict):
        entry = datasets[dataset_id]
    return dataset_id, entry


def _safe_json_loads(payload: Any) -> dict | None:
    """最佳尝试的JSON解码工具函数"""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _update_project_state_from_tool(
    state: AgentState, tool_name: str, tool_args: Dict[str, Any], tool_result: Any
) -> None:
    """根据工具执行结果更新项目状态"""
    payload = _safe_json_loads(tool_result)
    project_state = state.setdefault("project_state", {})

    work_dir = tool_args.get("work_dir") if isinstance(tool_args, dict) else None
    dataset_hint = None

    if isinstance(tool_args, dict):
        input_file = tool_args.get("file_path") or tool_args.get("input_path") or tool_args.get("h5ad_path")
    else:
        input_file = None

    if payload and isinstance(payload, dict):
        work_dir = payload.get("work_dir") or work_dir or payload.get("work_path") or payload.get("result_path")
        dataset_hint = payload.get("dataset_id")

    # 获取或创建数据集条目
    dataset_id = None
    dataset_entry = None

    if work_dir:
        from pathlib import Path
        dataset_id = Path(work_dir).name
    elif dataset_hint:
        dataset_id = dataset_hint
    else:
        dataset_id, dataset_entry = _get_active_dataset_entry(state)
        if not dataset_id:
            dataset_id = state.get("run_id")

    # 初始化数据集条目
    datasets: dict = project_state.setdefault("datasets", {})
    if not dataset_id:
        if work_dir:
            from pathlib import Path
            dataset_id = Path(work_dir).name
        elif state.get("work_dir"):
            from pathlib import Path
            dataset_id = Path(state["work_dir"]).name
        else:
            dataset_id = project_state.get("active_dataset") or project_state.get("last_dataset")

    if dataset_id:
        dataset_entry = datasets.setdefault(dataset_id, {})
        dataset_entry.setdefault("completed_steps", [])
        dataset_entry.setdefault("input_files", [])
        if input_file and input_file not in dataset_entry["input_files"]:
            dataset_entry["input_files"].append(input_file)

    if dataset_entry is None:
        return

    if work_dir:
        dataset_entry["work_dir"] = work_dir
        state["work_dir"] = work_dir

    # 更新具体工具的结果路径
    if tool_name == "load_h5ad_data" and payload:
        loaded_path = (
            payload.get("result_path")
            or payload.get("file_path")
            or input_file
            or dataset_entry.get("loaded_path")
        )
        dataset_entry["loaded_path"] = loaded_path
        if payload.get("work_dir"):
            dataset_entry["work_dir"] = payload["work_dir"]
            state["work_dir"] = payload["work_dir"]
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "calculate_qc_metrics" and payload:
        dataset_entry["qc_path"] = payload.get("result_path") or dataset_entry.get("qc_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "normalize_and_hvg" and payload:
        dataset_entry["normalized_path"] = payload.get("result_path") or dataset_entry.get("normalized_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "pca_reduction" and payload:
        dataset_entry["pca_path"] = payload.get("result_path") or dataset_entry.get("pca_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "cluster_and_umap" and payload:
        dataset_entry["clustered_path"] = payload.get("result_path") or dataset_entry.get("clustered_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "find_marker_genes" and payload:
        dataset_entry["markers_path"] = payload.get("result_path") or dataset_entry.get("markers_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "annotate_cells" and payload:
        dataset_entry["annotated_path"] = payload.get("result_path") or dataset_entry.get("annotated_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "differential_expression" and payload:
        dataset_entry["de_path"] = payload.get("result_path") or dataset_entry.get("de_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "generate_analysis_report" and payload:
        dataset_entry.setdefault("reports", {})["analysis_report"] = payload.get("result_path")
        _mark_completed(dataset_entry, tool_name)

    # 更新已完成的步骤记录
    completed_notes = state.setdefault("analysis_notes", {}).setdefault("completed_steps", [])
    from src.agent.nodes.planner import get_tool_step_label
    human_label = get_tool_step_label(tool_name)
    if human_label not in completed_notes:
        completed_notes.append(human_label)


def _mark_completed(entry: dict, tool_name: str) -> None:
    """标记工具为已完成"""
    completed = entry.setdefault("completed_steps", [])
    if tool_name not in completed:
        completed.append(tool_name)


def _extract_artifact_paths(payload: Dict[str, Any]) -> List[str]:
    """Extract artifact paths from a tool result payload."""
    paths: List[str] = []
    for key, value in payload.items():
        if not value:
            continue
        if isinstance(value, str) and key.endswith("path"):
            paths.append(value)
        elif key.endswith("paths"):
            if isinstance(value, list):
                paths.extend([item for item in value if isinstance(item, str)])
            elif isinstance(value, dict):
                paths.extend([item for item in value.values() if isinstance(item, str)])
        elif isinstance(value, dict) and key in {"result_paths", "artifacts"}:
            paths.extend([item for item in value.values() if isinstance(item, str)])
    return paths


# ============== 工具调用处理 ==============

async def _handle_tool_calls(
    state: AgentState,
    decision_message: AIMessage,
    current_step: str
) -> AgentState:
    """处理一个或多个工具调用"""
    tool_outputs = []
    tool_call_state = False
    tool_failed = False
    failure_payload: Dict[str, Any] | None = None

    # 获取持久化管理器
    persistence_manager = get_persistence_manager()
    run_id = state.get("run_id", str(uuid4()))

    for tool_call in decision_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {}) or {}
        tool_call_id = tool_call.get("id")

        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name in FILE_PATH_TOOLS:
            has_path = any(
                key in tool_args and tool_args.get(key)
                for key in ("file_path", "input_path", "h5ad_path")
            )
            if not has_path and state.get("input_files"):
                tool_args = dict(tool_args)
                tool_args["file_path"] = state["input_files"][0]

        if tool_name in WORK_DIR_TOOLS and not tool_args.get("work_dir"):
            work_dir = state.get("work_dir")
            if work_dir:
                tool_args = dict(tool_args)
                tool_args["work_dir"] = work_dir

        print(f"Attempting to call tool: {tool_name} with args: {tool_args}")

        if tool_name not in tools_by_name:
            error_msg = f"Tool '{tool_name}' not found in available tools."
            print(error_msg)

            # 保存错误记录
            persistence_manager.save_tool_result(
                run_id=run_id,
                tool_name=tool_name,
                params=tool_args,
                result={"status": "error", "message": error_msg}
            )

            tool_outputs.append(
                ToolMessage(
                    content=f"Error: {error_msg}",
                    name=tool_name,
                    tool_call_id=tool_call_id
                )
            )
        else:
            try:
                start_time = time.time()
                tool_to_call: BaseTool = tools_by_name[tool_name]

                # 检查是否有缓存
                cached_result = persistence_manager.get_cached_result(tool_name, tool_args)
                if cached_result:
                    print(f"Using cached result for {tool_name}")
                    tool_result = cached_result.get("result")
                    execution_time = cached_result.get("execution_time", 0)
                else:
                    if hasattr(tool_to_call, 'ainvoke'):
                        tool_result = await tool_to_call.ainvoke(tool_args)
                    else:
                        tool_result = tool_to_call.invoke(tool_args)
                    execution_time = time.time() - start_time

                    # 保存到缓存
                    if isinstance(tool_result, dict):
                        persistence_manager.save_cache(tool_name, tool_args, tool_result)

                # 记录工具调用历史
                state.setdefault("tool_history", []).append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                        "execution_time": execution_time,
                    }
                )

                # 处理特定工具的结果
                if tool_name == "dataset_bio_qa":
                    dataset_payload = _safe_json_loads(tool_result)
                    if dataset_payload:
                        state.setdefault("analysis_notes", {})["dataset_bio_qa"] = dataset_payload
                        if "work_dir" not in dataset_payload and state.get("work_dir"):
                            dataset_payload["work_dir"] = state.get("work_dir")

                # 更新项目状态
                _update_project_state_from_tool(state, tool_name, tool_args, tool_result)

                # 保存执行记录
                payload = _safe_json_loads(tool_result)
                if isinstance(tool_result, dict):
                    result_dict = tool_result
                elif payload:
                    result_dict = payload
                else:
                    result_dict = {"result": str(tool_result), "status": "success"}

                persistence_manager.save_tool_result(
                    run_id=run_id,
                    tool_name=tool_name,
                    params=tool_args,
                    result=result_dict,
                    execution_time=execution_time
                )

                state.setdefault("analysis_notes", {}).setdefault("tool_runs", []).append(
                    {
                        "tool": tool_name,
                        "status": result_dict.get("status", "success"),
                        "message": result_dict.get("message"),
                        "artifacts": _extract_artifact_paths(result_dict),
                    }
                )

                # 格式化结果消息
                if isinstance(tool_result, (dict, list)):
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                else:
                    tool_result_str = str(tool_result)

                result_str = f"<execute>Tool {tool_name} result: {tool_result_str}</execute>"
                print(f"Tool '{tool_name}' executed successfully in {execution_time:.2f}s.")

                if result_dict.get("status") == "error":
                    error_msg = result_dict.get("message") or "Tool returned error status."
                    failure_payload = {
                        "tool": tool_name,
                        "message": error_msg,
                        "step": current_step,
                    }
                    tool_outputs.append(
                        ToolMessage(
                            content=f"Error: {error_msg}",
                            name=tool_name,
                            tool_call_id=tool_call_id,
                        )
                    )
                    tool_failed = True
                else:
                    # 移动到下一步
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

                # 保存错误记录
                persistence_manager.save_tool_result(
                    run_id=run_id,
                    tool_name=tool_name,
                    params=tool_args,
                    result={"status": "error", "message": error_msg}
                )

                tool_outputs.append(
                    ToolMessage(
                        content=f"Error: {error_msg}",
                        name=tool_name,
                        tool_call_id=tool_call_id
                    )
                )
                state.setdefault("analysis_notes", {}).setdefault("tool_runs", []).append(
                    {
                        "tool": tool_name,
                        "status": "error",
                        "message": error_msg,
                        "artifacts": [],
                    }
                )
                failure_payload = {
                    "tool": tool_name,
                    "message": error_msg,
                    "step": current_step,
                }
                tool_failed = True

        if tool_failed:
            break

    # 添加工具输出到消息历史
    state["messages"].extend(tool_outputs)

    # 确定下一步
    if tool_failed:
        detail = failure_payload.get("message") if failure_payload else "Tool execution failed."
        tool_name = failure_payload.get("tool") if failure_payload else "unknown_tool"
        state["execution_status"] = "failed"
        state["error_info"] = {
            "detail": detail,
            "tool": tool_name,
            "step": failure_payload.get("step") if failure_payload else current_step,
        }
        state["messages"].append(
            AIMessage(
                content=(
                    f"<EXECUTION_ERROR>{tool_name} failed: {detail}</EXECUTION_ERROR>"
                    "<USER_REQUEST>质量控制失败。需要我尝试自动修复数据格式问题，还是停止分析？</USER_REQUEST>"
                )
            )
        )
        state["next_step"] = "response"
    elif tool_call_state:
        remaining_plan = state.get("plan", [])
        if remaining_plan:
            state["next_step"] = "general_executor"
        else:
            state["next_step"] = "response"
    else:
        state["next_step"] = "response"

    return state


# ============== 通用执行器节点 ==============

async def general_executor(state: AgentState) -> AgentState:
    """
    通用执行器节点
    执行当前计划步骤，支持工具调用和错误处理

    处理流程:
    1. 检查是否有计划需要执行
    2. 检查步骤是否已完成（可跳过）
    3. 使用LLM决定如何执行当前步骤
    4. 处理工具调用
    5. 更新状态和计划
    """
    plan = state.get("plan", [])
    if not plan:
        print("[General Executor] No remaining plan steps to execute.")
        state["messages"].append(AIMessage(content="<observation>No remaining plan steps to execute.</observation>"))
        state["next_step"] = "response"
        return state

    current_step = plan[0]

    # 检查步骤是否已完成
    dataset_id, dataset_entry = _get_active_dataset_entry(state)
    tool_name = identify_tool_from_step(str(current_step))
    if tool_name and dataset_entry and is_tool_completed(dataset_entry, tool_name):
        logger.info(
            "[General Executor] Skipping completed step '%s' for dataset '%s'",
            tool_name,
            dataset_id
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

    # 格式化对话历史
    messages = state.get("messages", [])
    formatted_history = "\n"

    if messages:
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        for i, msg in enumerate(recent_messages):
            content_preview = (msg.content[:500] + "...") if msg.content and len(msg.content) > 500 else (msg.content or "")
            formatted_history += f" {i+1}. [{msg.type}]{content_preview}\n"
    else:
        formatted_history += " (No previous conversation history)\n"

    # 构建执行提示
    task_formatted = f"""
You are a task execution agent responsible for completing a specific step in a larger plan.
You can observe the contextual environment of this step, and you need to extract necessary parameters from the previous conversation history to complete the current task.
You cannot fabricate parameters for tool calls. Parameters must be included in the conversation history, otherwise you will be punished.
If there are no tools in the conversation history to call the required parameters, make a request to the user.

You need to complete this task in the current step: {current_step}

Here is the recent conversation history: This provides a complete background of the interactions to date. Read carefully to understand:
*What has already been done.
*What data or variables may already exist in the execution environment (from the previous'<execute>'block).
*Any specific instructions or feedback given by the user or observed from previous executions.

{formatted_history}
"""

    try:
        llm_with_tools = get_llm_with_tools(TOOLS)
        decision_message = llm_with_tools.invoke(task_formatted)
        print(f"LLM Decision Message: {decision_message}")

        if decision_message.tool_calls:
            # 添加工具调用消息
            concise_tool_call_message = AIMessage(
                content=json.dumps(decision_message.tool_calls, indent=2),
                tool_calls=decision_message.tool_calls
            )
            state["messages"].append(concise_tool_call_message)

            # 处理工具调用
            state = await _handle_tool_calls(state, decision_message, current_step)
        else:
            # LLM没有调用工具
            response_content = decision_message.content.strip()

            if not response_content:
                error_msg = "LLM did not call any tool and provided no response content. You need to reanalyze the conversation history to complete the task"
                print(error_msg)
                state["messages"].append(AIMessage(content=f"<EXECUTION_ERROR>{error_msg}</EXECUTION_ERROR>"))
                state["next_step"] = "intelligent_replanner"

            print(f"LLM provided a direct response or request for input: {response_content}")

            observation_content = f"<observation>LLM response to task '{current_step}':{response_content}</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            state["next_step"] = "intelligent_replanner"

    except Exception as e:
        error_msg = f"An unexpected error occurred in execute_step: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        state["messages"].append(AIMessage(content=f"<EXECUTION_ERROR>{error_msg}</EXECUTION_ERROR>"))
        state["next_step"] = "intelligent_replanner"

    return state
