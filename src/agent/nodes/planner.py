"""
计划生成节点
根据用户意图和可用工具生成执行计划
包含完整的重试机制和错误处理
"""
import json
import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.agent.state import AgentState, Plan
from src.agent.tool_registry import TOOLS, get_tool_registry
from src.utils.llm_manager import get_llm
from src.utils.path_manager import create_analysis_work_dir

logger = logging.getLogger(__name__)

# ============== 工具步骤标签 ==============

TOOL_STEP_LABELS: dict = {
    # 单细胞核心工具
    "load_h5ad_data": "加载与预处理数据",
    "calculate_qc_metrics": "质量控制分析",
    "normalize_and_hvg": "标准化与高变基因识别",
    "pca_reduction": "主成分分析降维",
    "cluster_and_umap": "聚类与UMAP可视化",
    "find_marker_genes": "标记基因识别",
    "annotate_cells": "智能细胞类型注释",
    "annotate_with_simple_markers": "简单标记基因注释",
    "annotate_with_cima_markers": "CIMA标记基因注释",
    "annotate_with_blood_markers": "血液细胞标记注释",
    "annotate_with_llm": "LLM智能细胞注释",
    "differential_expression": "差异表达分析",
    "generate_analysis_report": "生成分析报告",
    # 高级工具
    "extract_embeddings_with_scgpt": "提取 scGPT 嵌入",
    "run_cellphonedb_core": "CellPhoneDB细胞通讯分析",
    "run_pseudotime_analysis": "伪时间轨迹分析",
    "run_ora_enrichment": "ORA富集分析",
    "generate_comprehensive_report": "生成综合报告",
    # 兼容旧工具
    "cluster_and_diff": "聚类与差异分析",
    "annotate_with_markers": "细胞类型注释",
    "interpret_cluster_results": "聚类功能解读",
    "interpret_celltype_results": "细胞类型叙述生成",
    "run_ssgsea_enrichment": "ssGSEA 富集分析",
    "dataset_bio_qa": "知识库问答总结",
}


# ============== 辅助函数 ==============

def _identify_tool_from_plan_step(step: str) -> str | None:
    """从计划步骤中识别工具名称

    支持通过以下方式识别:
    1. 英文工具名称直接匹配
    2. 中文标签匹配 (TOOL_STEP_LABELS)
    """
    if not isinstance(step, str):
        return None

    lowered = step.lower()

    # 首先尝试直接匹配英文工具名称
    for tool_name in (
        "load_h5ad_data",
        "calculate_qc_metrics",
        "normalize_and_hvg",
        "pca_reduction",
        "cluster_and_umap",
        "find_marker_genes",
        "annotate_cells",
        "annotate_with_simple_markers",
        "annotate_with_cima_markers",
        "annotate_with_blood_markers",
        "annotate_with_llm",
        "differential_expression",
        "generate_analysis_report",
        "generate_comprehensive_report",
        "extract_embeddings_with_scgpt",
        "run_cellphonedb_core",
        "run_pseudotime_analysis",
        "run_ora_enrichment",
        # 兼容旧工具
        "cluster_and_diff",
        "annotate_with_markers",
        "interpret_cluster_results",
        "interpret_celltype_results",
        "run_ssgsea_enrichment",
    ):
        if tool_name in lowered:
            return tool_name

    # 尝试通过中文标签匹配
    for tool_name, label in TOOL_STEP_LABELS.items():
        if label in step:
            return tool_name

    # 关键词匹配
    if '标记基因' in step or 'marker' in lowered:
        return 'find_marker_genes'
    if '聚类' in step and 'umap' in lowered:
        return 'cluster_and_umap'
    if 'pca' in lowered or '主成分' in step:
        return 'pca_reduction'
    if '注释' in step and 'cima' in lowered:
        return 'annotate_with_cima_markers'
    if '注释' in step and '血液' in step:
        return 'annotate_with_blood_markers'
    if '注释' in step and '简单' in step:
        return 'annotate_with_simple_markers'
    if '注释' in step or 'annotate' in lowered:
        return 'annotate_cells'  # 智能注释工具
    if '质控' in step or 'qc' in lowered:
        return 'calculate_qc_metrics'
    if '标准化' in step or 'hvg' in lowered:
        return 'normalize_and_hvg'
    if '报告' in step or 'report' in lowered:
        return 'generate_analysis_report'
    if '差异' in step or 'de' in lowered:
        return 'differential_expression'

    return None


def _is_tool_already_completed(dataset_entry: dict, tool_name: str) -> bool:
    """检查工具是否已在当前数据集上完成"""
    if not dataset_entry:
        return False

    # 新的单细胞核心工具
    if tool_name == "load_h5ad_data":
        return bool(dataset_entry.get("loaded_path") or dataset_entry.get("preprocessed_path"))
    if tool_name == "calculate_qc_metrics":
        return bool(dataset_entry.get("qc_path"))
    if tool_name == "normalize_and_hvg":
        return bool(dataset_entry.get("normalized_path"))
    if tool_name == "pca_reduction":
        return bool(dataset_entry.get("pca_path"))
    if tool_name == "cluster_and_umap":
        return bool(dataset_entry.get("clustered_path"))
    if tool_name == "find_marker_genes":
        return bool(dataset_entry.get("markers_path"))
    if tool_name == "annotate_cells":
        return bool(dataset_entry.get("annotated_path"))
    if tool_name in ("annotate_with_simple_markers", "annotate_with_cima_markers", "annotate_with_blood_markers", "annotate_with_llm"):
        return bool(dataset_entry.get("annotated_path"))
    if tool_name == "differential_expression":
        return bool(dataset_entry.get("de_path"))
    if tool_name == "generate_analysis_report":
        reports = dataset_entry.get("reports", {})
        return bool(reports.get("analysis_report"))

    # 高级工具
    if tool_name == "extract_embeddings_with_scgpt":
        return bool(dataset_entry.get("embeddings_path"))

    # 兼容旧工具
    if tool_name == "cluster_and_diff":
        return bool(dataset_entry.get("clustered_path") and dataset_entry.get("diff_gene_path"))
    if tool_name == "annotate_with_markers":
        return bool(
            dataset_entry.get("annotated_path") or dataset_entry.get("annotation", {}).get("result_path")
        )
    if tool_name == "interpret_cluster_results":
        interpretation = dataset_entry.get("interpretation", {})
        return bool(interpretation.get("dataset_report") or interpretation.get("clusters"))
    if tool_name == "interpret_celltype_results":
        report = dataset_entry.get("celltype_report", {})
        return bool(report.get("report_path") or report.get("celltype_context_path"))
    if tool_name == "run_ssgsea_enrichment":
        enrichment = dataset_entry.get("enrichment", {}).get("ssgsea", {})
        return bool(enrichment.get("result_paths") or enrichment.get("status"))
    return False


def _get_active_dataset_entry(state: AgentState) -> tuple[str | None, dict | None]:
    """获取当前活动的数据集条目"""
    project_state = state.get("project_state") or {}
    datasets = project_state.get("datasets") or {}
    dataset_id = project_state.get("active_dataset") or project_state.get("last_dataset")
    entry: dict | None = None
    if dataset_id and isinstance(datasets.get(dataset_id), dict):
        entry = datasets[dataset_id]
    return dataset_id, entry


def _prune_completed_plan(state: AgentState) -> None:
    """从计划中移除已完成的步骤"""
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


def _inject_missing_dependencies(state: AgentState) -> None:
    """自动注入缺失的依赖步骤到计划中

    确保计划中的每个工具的依赖都已满足，如果依赖步骤不在计划中且未完成，
    则自动将其插入到正确的位置。
    """
    plan = state.get("plan") or []
    if not plan:
        return

    _, dataset_entry = _get_active_dataset_entry(state)
    tool_registry = get_tool_registry()

    # 获取依赖关系映射
    deps_map = tool_registry.TOOL_DEPENDENCIES

    # 构建已完成工具集合
    completed = set()
    if dataset_entry:
        completed_steps = dataset_entry.get("completed_steps") or []
        completed = set(completed_steps)

    # 构建计划中的工具集合（保持顺序）
    plan_tools = []
    for step in plan:
        tool_name = _identify_tool_from_plan_step(step)
        if tool_name:
            plan_tools.append(tool_name)

    # 找出需要注入的依赖工具
    injected_tools = []
    final_plan_tools = []

    for tool in plan_tools:
        # 获取工具的依赖
        deps = deps_map.get(tool, [])

        # 检查每个依赖是否满足
        for dep in deps:
            # 如果依赖不在已完成集合中，也不在计划中，需要注入
            if dep not in completed and dep not in final_plan_tools and dep not in plan_tools:
                # 递归检查依赖的依赖
                dep_deps = deps_map.get(dep, [])
                for dep_dep in reversed(dep_deps):
                    if dep_dep not in completed and dep_dep not in final_plan_tools and dep_dep not in plan_tools:
                        final_plan_tools.append(dep_dep)
                final_plan_tools.append(dep)
                injected_tools.append(dep)

        final_plan_tools.append(tool)

    if injected_tools:
        logger.info(f"[Planner] 自动注入缺失的依赖步骤: {injected_tools}")

        # 重建计划步骤
        new_plan = []
        tool_to_step = {}
        for step in plan:
            tool_name = _identify_tool_from_plan_step(step)
            if tool_name:
                tool_to_step[tool_name] = step

        for tool in final_plan_tools:
            if tool in tool_to_step:
                new_plan.append(tool_to_step[tool])
            else:
                # 为注入的工具生成步骤描述
                step_label = TOOL_STEP_LABELS.get(tool, tool)
                new_plan.append(f"{step_label}")

        state["plan"] = new_plan
        logger.info(f"[Planner] 依赖注入后的计划: {new_plan}")


# ============== 计划生成节点 ==============

async def general_planner(state: AgentState) -> AgentState:
    """
    通用规划节点
    根据意图和可用工具动态生成执行计划
    支持重试机制和错误恢复
    """
    logger.info("[General Planner] Starting planning based on user intent")
    input_files = state.get("input_files", [])

    # 创建工作目录
    if input_files and "input_file_info" in state:
        first_file_info = state["input_file_info"][0]
        base_name = first_file_info["file_name"].replace('.h5ad', '')
        work_dir = create_analysis_work_dir(base_name)
        state["work_dir"] = str(work_dir)
        logger.info(f"[General Planner] Created working directory: {work_dir}")

    # 获取工具注册表和工具描述
    tool_registry = get_tool_registry()
    tools_info = tool_registry.get_tool_description_for_llm()

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
            llm = get_llm()

            plan_prompt = plan_prompt_template.format(tools_info=tools_info)

            logger.info(f"[General Planner] Attempting to generate plan (Attempt {attempt + 1}/{max_attempts})")

            if attempt > 0 and previous_errors_feedback:
                feedback_prompt = f"""
IMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT (Attempt {attempt + 1}):
The plan generated in the previous attempt had the following issues:
{previous_errors_feedback}

Please carefully review the feedback above and regenerate the plan, ensuring it adheres strictly to the required JSON format and content guidelines.
"""
                plan_prompt = plan_prompt + "\n\n" + feedback_prompt

            # 添加 JSON 格式要求（兼容智谱 API）
            plan_prompt += """

IMPORTANT: You must respond ONLY with a valid JSON object. Do not include any explanatory text before or after the JSON.
Example format:
{
  "steps": ["步骤1", "步骤2", "步骤3"]
}
"""

            state["messages"].append(HumanMessage(content=plan_prompt))

            plan_messages = state["messages"]

            # 使用 JSON 响应格式
            response = llm.invoke(plan_messages)
            response_text = getattr(response, "content", str(response)).strip()

            # 尝试提取 JSON（处理可能的 markdown 代码块）
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()
            elif response_text.startswith("```"):
                response_text = response_text.strip("`").replace("json", "").strip()

            logger.info(f"[General Planner] Raw LLM response: {response_text[:500]}")

            # 解析 JSON
            parsed = json.loads(response_text)
            raw_steps = parsed.get("steps", [])

            # 处理步骤格式：兼容字符串和字典两种格式
            processed_steps = []
            for step in raw_steps:
                if isinstance(step, dict):
                    # 字典格式：转换为字符串描述
                    tool_name = step.get("tool", "")
                    params = step.get("params", {})
                    # 移除 None 值
                    params = {k: v for k, v in params.items() if v is not None}
                    if params:
                        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                        processed_steps.append(f"{tool_name}({param_str})")
                    else:
                        processed_steps.append(tool_name)
                elif isinstance(step, str):
                    processed_steps.append(step)

            plan = Plan(steps=processed_steps)
            logger.info(f"[General Planner] Plan generation result: {plan}")

            if plan and isinstance(plan.steps, list) and len(plan.steps) > 0:
                # 检查步骤是否非空
                non_empty_steps = [step for step in plan.steps if step.strip()]
                if non_empty_steps:
                    tool_names = []
                    for step in non_empty_steps:
                        tool_name = _identify_tool_from_plan_step(step)
                        if tool_name:
                            tool_names.append(tool_name)
                    validation = tool_registry.validate_plan(tool_names)
                    logger.info("[General Planner] Planned tools: %s", tool_names)
                    if not validation["valid"]:
                        previous_errors_feedback = "Plan uses unavailable tools: " + ", ".join(validation["errors"])
                        logger.warning("[General Planner] %s", previous_errors_feedback)
                        continue
                    plan_generation_success = True
                    state["plan"] = non_empty_steps
                    # 自动注入缺失的依赖步骤
                    _inject_missing_dependencies(state)
                    _prune_completed_plan(state)
                    current_plan = state.get("plan", [])
                    if not current_plan:
                        logger.info(
                            "[General Planner] All proposed steps already completed; skipping execution."
                        )
                        state["messages"].append(
                            AIMessage(
                                content="<PLAN_GENERATED>\n{\n  \"steps\": []\n}\n</PLAN_GENERATED>"
                            )
                        )
                        state["next_step"] = "response"
                        return state
                    logger.info("[General Planner] Plan generated successfully.")
                    previous_errors_feedback = ""
                    break
                else:
                    error_msg = "Plan generation produced an object but steps list is invalid or empty."
                    logger.warning(f"[General Planner] All steps are empty (Attempt {attempt + 1}).")
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

    # 根据生成结果决定下一步
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


# ============== 导出函数供其他模块使用 ==============

def get_tool_step_label(tool_name: str) -> str:
    """获取工具的中文标签"""
    return TOOL_STEP_LABELS.get(tool_name, tool_name)


def identify_tool_from_step(step: str) -> str | None:
    """从步骤描述中识别工具名称"""
    return _identify_tool_from_plan_step(step)


def is_tool_completed(dataset_entry: dict, tool_name: str) -> bool:
    """检查工具是否已完成"""
    return _is_tool_already_completed(dataset_entry, tool_name)


def prune_completed_steps(state: AgentState) -> None:
    """从计划中移除已完成的步骤"""
    _prune_completed_plan(state)
