"""
Agent状态图构建
使用LangGraph构建Agent执行流程
"""
import json
from uuid import uuid4
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.nodes import (
    intent_recognition,
    general_planner,
    general_executor,
    intelligent_replanner,
    response_node,
)
from src.agent.nodes.planner import TOOL_STEP_LABELS
from src.memory.conversation_memory import ConversationMemoryStore


# 全局内存存储实例
conversation_memory = ConversationMemoryStore()


def route_after_intent(state: AgentState) -> str:
    """意图识别后的路由"""
    if state.get("next_step") == "planner":
        return "planner"
    return "response"


def route_after_planner(state: AgentState) -> str:
    """规划后的路由"""
    next_step = state.get("next_step", "response")
    if next_step == "general_executor":
        return "executor"
    return "response"


def route_after_executor(state: AgentState) -> str:
    """执行后的路由"""
    next_step = state.get("next_step", "response")
    if next_step == "general_executor":
        return "executor"
    elif next_step == "replanner":
        return "replanner"
    return "response"


def route_after_replan(state: AgentState) -> str:
    """重规划后的路由"""
    if state.get("execution_status") in ["completed", "failed"]:
        return END
    return "executor"


def build_graph():
    """构建Agent状态图"""
    # 创建状态图
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("intent_recognition", intent_recognition)
    graph.add_node("planner", general_planner)
    graph.add_node("executor", general_executor)
    graph.add_node("replanner", intelligent_replanner)
    graph.add_node("response", response_node)

    # 添加边
    graph.add_edge(START, "intent_recognition")

    # 意图识别后的条件边
    graph.add_conditional_edges(
        "intent_recognition",
        route_after_intent,
        {
            "planner": "planner",
            "response": "response"
        }
    )

    # 规划后的条件边
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "executor": "executor",
            "response": "response"
        }
    )

    # 执行后的条件边
    graph.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "executor": "executor",
            "replanner": "replanner",
            "response": "response"
        }
    )

    # 重规划后的条件边
    graph.add_conditional_edges(
        "replanner",
        route_after_replan,
        {
            "executor": "executor",
            "response": "response",
            "__end__": END
        }
    )

    graph.add_edge("response", END)

    # 编译（带检查点）
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# 全局图实例
_agent_graph = None


def get_agent_graph():
    """获取Agent图单例"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_graph()
    return _agent_graph


def build_project_state_message(project_state: dict) -> str | None:
    """构建项目状态消息"""
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

        # 新的单细胞核心工具结果
        if entry.get("loaded_path"):
            lines.append(f"  • 数据已加载：`{entry['loaded_path']}`")
        if entry.get("qc_path"):
            lines.append(f"  • 质量控制已完成：`{entry['qc_path']}`")
        if entry.get("normalized_path"):
            lines.append(f"  • 标准化已完成：`{entry['normalized_path']}`")
        if entry.get("pca_path"):
            lines.append(f"  • PCA降维已完成：`{entry['pca_path']}`")
        if entry.get("clustered_path"):
            lines.append(f"  • 聚类结果：`{entry['clustered_path']}`")
        if entry.get("markers_path"):
            lines.append(f"  • 标记基因：`{entry['markers_path']}`")
        if entry.get("annotated_path"):
            lines.append(f"  • 细胞注释：`{entry['annotated_path']}`")
        if entry.get("de_path"):
            lines.append(f"  • 差异表达分析：`{entry['de_path']}`")

        # 高级工具
        if entry.get("embeddings_path"):
            lines.append(f"  • 嵌入文件：`{entry['embeddings_path']}`")

        # 兼容旧工具
        enrichment = entry.get("enrichment", {}).get("ssgsea", {})
        if enrichment.get("result_paths"):
            lines.append("  • 已计算 ssGSEA 富集分析结果，可直接复用。")

        reports = entry.get("reports", {})
        if reports.get("analysis_report"):
            lines.append(f"  • 分析报告：`{reports['analysis_report']}`")

    return "\n".join(lines)


def serialise_project_state(project_state: dict) -> dict:
    """序列化项目状态"""
    def _convert(value):
        if isinstance(value, dict):
            return {key: _convert(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, set):
            return sorted(str(item) for item in value)
        return value

    return _convert(project_state or {})


def create_initial_state(
    objective: str,
    input_files: list = None,
    thread_id: str = None,
    session_id: str = None,
    run_id: str = None
) -> AgentState:
    """
    创建初始Agent状态
    包含内存上下文加载和项目状态恢复
    """
    resolved_thread_id = (thread_id or str(uuid4())).strip() or str(uuid4())
    resolved_run_id = run_id or str(uuid4())

    # 加载内存上下文
    memory_context = conversation_memory.load_context(
        thread_id=resolved_thread_id, objective=objective
    )
    memory_messages = conversation_memory.build_context_messages(memory_context)
    project_state = (
        json.loads(json.dumps(memory_context.project_state))
        if memory_context.project_state
        else {}
    )

    # 如果有新的输入文件，重置project_state以避免基于旧状态跳过步骤
    if input_files and len(input_files) > 0:
        project_state = {}
        memory_context.project_state = {}

    # 构建项目状态消息
    project_message = build_project_state_message(project_state)
    if project_message:
        memory_messages.insert(0, SystemMessage(content=project_message))

    # 创建初始状态
    state: AgentState = {
        "objective": objective,
        "messages": list(memory_messages),
        "input_files": input_files or [],
        "intents": [],
        "plan": [],
        "next_step": None,
        "execution_status": "in_progress",
        "replan_attempts": 0,
        "max_replan_attempts": 4,
        "work_dir": None,
        "tool_history": [],
        "analysis_notes": {},
        "thread_id": resolved_thread_id,
        "session_id": session_id or resolved_thread_id,
        "run_id": resolved_run_id,
        "memory_summary": memory_context.summary,
        "memory_records": [record.__dict__ for record in memory_context.records],
        "project_state": project_state,
    }

    # 恢复项目状态中的工作目录和输入文件（仅当没有新输入文件时）
    if project_state and not input_files:
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

    return state
