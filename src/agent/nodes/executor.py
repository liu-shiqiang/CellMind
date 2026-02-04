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
    "annotate_cells",  # 智能注释工具
    "annotate_with_simple_markers",
    "annotate_with_cima_markers",
    "annotate_with_blood_markers",
    "annotate_with_llm",  # LLM注释工具
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

    # 辅助函数：从 payload 提取 result_path
    # 修复：result_path 位于 payload["artifacts"]["result_path"]，而非 payload["result_path"]
    def get_result_path(payload: dict | None) -> str | None:
        if not payload:
            return None
        return payload.get("artifacts", {}).get("result_path")

    if payload and isinstance(payload, dict):
        work_dir = payload.get("work_dir") or work_dir or payload.get("work_path")
        result_path = get_result_path(payload)
        dataset_hint = payload.get("dataset_id")
    else:
        result_path = None

    # 获取或创建数据集条目
    dataset_id = None
    dataset_entry = None

    if work_dir:
        from pathlib import Path
        dataset_id = Path(work_dir).name
    elif result_path:
        # 从结果路径中提取 run_id
        from pathlib import Path
        result_path_obj = Path(result_path)
        # 查找 runs 目录下的 run_id
        if "runs" in result_path_obj.parts:
            runs_idx = result_path_obj.parts.index("runs")
            if runs_idx + 1 < len(result_path_obj.parts):
                dataset_id = result_path_obj.parts[runs_idx + 1]
        if not dataset_id:
            # 如果没有找到，使用现有 active_dataset
            dataset_id = project_state.get("active_dataset") or project_state.get("last_dataset")
    elif dataset_hint:
        dataset_id = dataset_hint
    else:
        dataset_id, dataset_entry = _get_active_dataset_entry(state)
        if not dataset_id:
            dataset_id = state.get("run_id")

    # 初始化数据集条目 - 确保始终能找到或创建 dataset_id
    datasets: dict = project_state.setdefault("datasets", {})
    if not dataset_id:
        if work_dir:
            from pathlib import Path
            dataset_id = Path(work_dir).name
        elif state.get("work_dir"):
            from pathlib import Path
            dataset_id = Path(state["work_dir"]).name
        elif result_path:
            # 从文件路径提取 run_id 作为 dataset_id
            from pathlib import Path
            result_path_obj = Path(result_path)
            if "runs" in result_path_obj.parts:
                runs_idx = result_path_obj.parts.index("runs")
                if runs_idx + 1 < len(result_path_obj.parts):
                    dataset_id = result_path_obj.parts[runs_idx + 1]
        if not dataset_id:
            dataset_id = project_state.get("active_dataset") or project_state.get("last_dataset") or "default"

    # 确保 dataset_entry 被创建
    if dataset_id:
        dataset_entry = datasets.setdefault(dataset_id, {})
        dataset_entry.setdefault("completed_steps", [])
        dataset_entry.setdefault("input_files", [])
        if input_file and input_file not in dataset_entry.get("input_files", []):
            dataset_entry["input_files"].append(input_file)
    else:
        # 极端情况：创建默认数据集条目
        logger.warning(f"[{tool_name}] 无法确定 dataset_id，创建默认条目")
        dataset_id = "default"
        dataset_entry = datasets.setdefault(dataset_id, {})
        dataset_entry.setdefault("completed_steps", [])
        dataset_entry.setdefault("input_files", [])

    if dataset_entry is None:
        return

    # 设置当前活动的数据集
    if dataset_id:
        project_state["active_dataset"] = dataset_id
        project_state["last_dataset"] = dataset_id

    if work_dir:
        dataset_entry["work_dir"] = work_dir
        state["work_dir"] = work_dir

    # 更新具体工具的结果路径
    # 修复：使用 get_result_path() 辅助函数从 payload["artifacts"]["result_path"] 提取路径
    if tool_name == "load_h5ad_data" and payload:
        loaded_path = (
            get_result_path(payload)
            or payload.get("data", {}).get("file_path")
            or input_file
            or dataset_entry.get("loaded_path")
        )
        dataset_entry["loaded_path"] = loaded_path
        if payload.get("data", {}).get("work_dir"):
            dataset_entry["work_dir"] = payload["data"]["work_dir"]
            state["work_dir"] = payload["data"]["work_dir"]

        # 保存 n_cells 和 n_genes，用于后续报告生成
        if payload.get("data", {}).get("n_cells"):
            dataset_entry["n_cells"] = payload["data"]["n_cells"]
        if payload.get("data", {}).get("n_genes"):
            dataset_entry["n_genes"] = payload["data"]["n_genes"]
        if payload.get("n_cells"):
            dataset_entry["n_cells"] = payload["n_cells"]
        if payload.get("n_genes"):
            dataset_entry["n_genes"] = payload["n_genes"]

        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "calculate_qc_metrics" and payload:
        dataset_entry["qc_path"] = get_result_path(payload) or dataset_entry.get("qc_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "normalize_and_hvg" and payload:
        dataset_entry["normalized_path"] = get_result_path(payload) or dataset_entry.get("normalized_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "pca_reduction" and payload:
        dataset_entry["pca_path"] = get_result_path(payload) or dataset_entry.get("pca_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "cluster_and_umap" and payload:
        dataset_entry["clustered_path"] = get_result_path(payload) or dataset_entry.get("clustered_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "find_marker_genes" and payload:
        dataset_entry["markers_path"] = get_result_path(payload) or dataset_entry.get("markers_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "annotate_cells" and payload:
        dataset_entry["annotated_path"] = get_result_path(payload) or dataset_entry.get("annotated_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name in ("annotate_with_simple_markers", "annotate_with_cima_markers", "annotate_with_blood_markers", "annotate_with_llm", "annotate_cells") and payload:
        dataset_entry["annotated_path"] = get_result_path(payload) or dataset_entry.get("annotated_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "differential_expression" and payload:
        dataset_entry["de_path"] = get_result_path(payload) or dataset_entry.get("de_path")
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "generate_analysis_report" and payload:
        dataset_entry.setdefault("reports", {})["analysis_report"] = get_result_path(payload)
        _mark_completed(dataset_entry, tool_name)

    elif tool_name == "generate_comprehensive_report" and payload:
        # 保存报告路径
        if payload.get("status") == "success":
            report_paths = payload.get("report_paths", [])
            if report_paths:
                dataset_entry.setdefault("reports", {})["comprehensive_report"] = report_paths
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
    3. 自动识别并执行当前步骤对应的工具
    4. 更新状态和计划
    5. 继续执行下一个步骤（循环）
    """
    plan = state.get("plan", [])

    # 循环执行所有计划步骤，直到完成或出错
    while plan:
        current_step = plan[0]

        # 检查步骤是否已完成
        dataset_id, dataset_entry = _get_active_dataset_entry(state)
        tool_name = identify_tool_from_step(str(current_step))

        # 如果步骤已完成，跳过
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
            plan = state["plan"]
            continue

        # 如果无法识别工具，跳过此步骤
        if not tool_name:
            logger.warning("[General Executor] Could not identify tool for step: %s", current_step)
            observation = f"<observation>无法识别步骤对应的工具: {current_step}，跳过此步骤。</observation>"
            state["messages"].append(AIMessage(content=observation))
            state["plan"] = plan[1:]
            plan = state["plan"]
            continue

        # 检查工具是否可用
        if tool_name not in tools_by_name:
            error_msg = f"Tool '{tool_name}' not found in available tools."
            print(error_msg)
            state["messages"].append(AIMessage(content=f"<EXECUTION_ERROR>{error_msg}</EXECUTION_ERROR>"))
            state["execution_status"] = "failed"
            state["error_info"] = {"detail": error_msg, "tool": tool_name, "step": str(current_step)}
            state["next_step"] = "response"
            return state

        # 构建工具调用参数
        tool_args = _build_tool_args(state, tool_name)

        # 创建模拟的 AIMessage 包含工具调用
        from uuid import uuid4
        tool_call_id = str(uuid4())
        mock_ai_message = AIMessage(
            content=f"Executing tool: {tool_name}",
            tool_calls=[{
                "name": tool_name,
                "args": tool_args,
                "id": tool_call_id
            }]
        )

        # 添加工具调用消息到历史
        state["messages"].append(mock_ai_message)

        # 处理工具调用
        state = await _handle_tool_calls(state, mock_ai_message, str(current_step))

        # 检查执行是否失败
        if state.get("execution_status") == "failed":
            return state

        # 更新 plan 引用
        plan = state.get("plan", [])

    # 所有步骤已完成
    print("[General Executor] All plan steps completed.")
    state["execution_status"] = "completed"
    state["next_step"] = "response"
    return state


def _build_tool_args(state: AgentState, tool_name: str) -> Dict[str, Any]:
    """根据工具名称构建工具调用参数

    按优先级获取文件路径：
    1. 使用上一步骤保存的最新结果路径
    2. 使用原始上传的文件路径
    3. 从 work_dir/artifacts/data 中查找最新的 h5ad 文件
    4. 从 work_dir 中查找最新的 h5ad 文件
    """
    tool_args = {}

    # 获取最新的文件路径（从project_state中获取上一步的结果）
    latest_file_path = _get_latest_result_path(state, tool_name)

    # 从 state 获取文件路径
    if tool_name in FILE_PATH_TOOLS:
        if latest_file_path:
            tool_args["file_path"] = latest_file_path
        else:
            # 回退策略1: 从 work_dir 的 artifacts/data 查找最新 h5ad
            if state.get("work_dir"):
                from pathlib import Path
                work_dir = Path(state["work_dir"])
                artifacts_dir = work_dir / "artifacts" / "data"
                if artifacts_dir.exists():
                    h5ad_files = sorted(
                        artifacts_dir.glob("*.h5ad"),
                        key=lambda x: x.stat().st_mtime,
                        reverse=True
                    )
                    if h5ad_files:
                        tool_args["file_path"] = str(h5ad_files[0])
                        logger.info(f"[{tool_name}] 使用 artifacts 中最新文件: {h5ad_files[0]}")

            # 回退策略2: 使用 input_files
            if "file_path" not in tool_args:
                input_files = state.get("input_files", [])
                if input_files:
                    tool_args["file_path"] = input_files[0]

            # 回退策略3: 从 work_dir 根目录查找最新的 h5ad 文件
            if "file_path" not in tool_args and state.get("work_dir"):
                from pathlib import Path
                work_dir = Path(state["work_dir"])
                h5ad_files = sorted(work_dir.glob("*.h5ad"), key=lambda x: x.stat().st_mtime, reverse=True)
                if h5ad_files:
                    tool_args["file_path"] = str(h5ad_files[0])
                    logger.info(f"[{tool_name}] 使用 work_dir 中最新文件: {h5ad_files[0]}")

    # 从 state 获取 work_dir
    if tool_name in WORK_DIR_TOOLS or tool_name in {
        "calculate_qc_metrics", "normalize_and_hvg", "pca_reduction",
        "cluster_and_umap", "find_marker_genes", "annotate_cells",
        "differential_expression", "generate_analysis_report",
        "generate_comprehensive_report"
    }:
        work_dir = state.get("work_dir")
        if work_dir:
            tool_args["work_dir"] = work_dir

    # 特殊工具的额外参数
    if tool_name == "annotate_cells":
        # 检查是否有组织类型信息
        messages = state.get("messages", [])
        for msg in reversed(messages[-20:]):
            if msg.content and "tissue" in msg.content.lower():
                # 简单提取组织类型（可根据需要增强）
                tool_args["tissue_type"] = "human"  # 默认值

    elif tool_name == "generate_comprehensive_report":
        # 获取 run_id
        run_id = state.get("run_id", "")

        # 获取输入文件
        input_files = state.get("input_files", [])
        data_file = input_files[0] if input_files else ""

        # 从 dataset_entry 获取 n_cells 和 n_genes
        _, dataset_entry = _get_active_dataset_entry(state)
        n_cells = dataset_entry.get("n_cells", 0) if dataset_entry else 0
        n_genes = dataset_entry.get("n_genes", 0) if dataset_entry else 0

        # 如果 dataset_entry 中没有 n_cells/n_genes，尝试从原始加载信息中获取
        if n_cells == 0 or n_genes == 0:
            analysis_notes = state.get("analysis_notes", {})
            tool_runs = analysis_notes.get("tool_runs", [])
            for run in tool_runs:
                if run.get("tool") == "load_h5ad_data":
                    # 尝试从工具运行结果中获取
                    if "n_cells" in run:
                        n_cells = run["n_cells"]
                    if "n_genes" in run:
                        n_genes = run["n_genes"]
                    break

        # 收集所有工具执行结果作为 analysis_results
        # build_report_from_results 期望的格式: results["tools"] 是一个 dict
        # 其中 key 是 tool_name, value 是单个 tool_result dict (不是 list)
        tool_runs = state.get("analysis_notes", {}).get("tool_runs", [])

        # 构建 tools 字典 - 每个工具只保留最后一次执行结果
        tools_dict = {}
        for run in tool_runs:
            tool_n = run.get("tool")
            if tool_n:
                # 保留最后一次的结果
                tools_dict[tool_n] = run

        # 构建完整的 results 字典
        results_dict = {
            "tools": tools_dict,
            "tool_runs": tool_runs,
        }

        tool_args = {
            "run_id": run_id or "",
            "data_file": data_file,
            "n_cells": n_cells if n_cells else 0,
            "n_genes": n_genes if n_genes else 0,
            "analysis_results": json.dumps(results_dict, ensure_ascii=False),
            "output_format": "both",
        }

    return tool_args


def _get_latest_result_path(state: AgentState, tool_name: str) -> str | None:
    """获取工具应该使用的最新结果路径

    每个工具应该使用上一步骤保存的结果文件，这样可以确保：
    1. 依次传递处理后的数据
    2. 每个步骤都基于最新的结果继续处理
    """
    # 工具到其依赖的前置工具的映射
    tool_dependency_map = {
        "calculate_qc_metrics": ["load_h5ad_data"],
        "normalize_and_hvg": ["calculate_qc_metrics"],
        "pca_reduction": ["normalize_and_hvg"],
        "cluster_and_umap": ["pca_reduction"],
        "find_marker_genes": ["cluster_and_umap"],
        "annotate_cells": ["find_marker_genes"],  # 智能注释工具
        "annotate_with_simple_markers": ["find_marker_genes"],
        "annotate_with_cima_markers": ["find_marker_genes"],
        "annotate_with_blood_markers": ["find_marker_genes"],
        "annotate_with_llm": ["find_marker_genes"],  # LLM注释工具
        "differential_expression": ["cluster_and_umap"],
        "generate_analysis_report": ["annotate_cells", "annotate_with_simple_markers", "annotate_with_cima_markers", "annotate_with_blood_markers", "annotate_with_llm"],
    }

    # 工具结果路径在 dataset_entry 中的字段名
    path_field_map = {
        "load_h5ad_data": "loaded_path",
        "calculate_qc_metrics": "qc_path",
        "normalize_and_hvg": "normalized_path",
        "pca_reduction": "pca_path",
        "cluster_and_umap": "clustered_path",
        "find_marker_genes": "markers_path",
        "annotate_cells": "annotated_path",
        "annotate_with_simple_markers": "annotated_path",
        "annotate_with_cima_markers": "annotated_path",
        "annotate_with_blood_markers": "annotated_path",
        "annotate_with_llm": "annotated_path",
        "differential_expression": "de_path",
    }

    # 获取当前数据集条目
    _, dataset_entry = _get_active_dataset_entry(state)

    # 查找依赖工具的结果路径
    if dataset_entry:
        dependency_tools = tool_dependency_map.get(tool_name, [])
        for dep_tool in reversed(dependency_tools):  # 反向查找，取最接近的
            field_name = path_field_map.get(dep_tool)
            if field_name and field_name in dataset_entry:
                result_path = dataset_entry[field_name]
                if result_path:
                    logger.info(f"[{tool_name}] Using result from {dep_tool}: {result_path}")
                    return result_path

    # 回退策略: 基于文件名模式查找
    if state.get("work_dir"):
        from pathlib import Path
        work_dir = Path(state["work_dir"])
        artifacts_dir = work_dir / "artifacts" / "data"

        # 根据工具依赖关系查找对应的文件模式
        file_patterns = {
            "find_marker_genes": "*cluster*",
            "annotate_cells": "*marker*",
            "annotate_with_simple_markers": "*marker*",
            "annotate_with_cima_markers": "*marker*",
            "annotate_with_blood_markers": "*marker*",
            "annotate_with_llm": "*marker*",
            "differential_expression": "*cluster*",
            "generate_analysis_report": "*annotate*",
        }

        pattern = file_patterns.get(tool_name)
        if pattern and artifacts_dir.exists():
            matching_files = list(artifacts_dir.glob(f"{pattern}.h5ad"))
            if matching_files:
                latest = max(matching_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"[{tool_name}] 通过模式匹配找到文件: {latest}")
                return str(latest)

        # 如果没有找到特定模式的文件，使用最新的任意 h5ad 文件
        if artifacts_dir.exists():
            h5ad_files = sorted(artifacts_dir.glob("*.h5ad"), key=lambda x: x.stat().st_mtime, reverse=True)
            if h5ad_files:
                logger.info(f"[{tool_name}] 使用 artifacts 中最新文件: {h5ad_files[0]}")
                return str(h5ad_files[0])

    return None
