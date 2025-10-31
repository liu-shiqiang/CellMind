import pytest

pytest.importorskip("langchain_core", reason="LangChain core library not available")

from langchain_core.messages import HumanMessage

from src.agent.agent_new import (
    Intent,
    build_project_state_message,
    response,
    _prune_completed_plan,
)


@pytest.mark.asyncio
async def test_memory_query_short_circuits_to_memory_response():
    state = {
        "messages": [HumanMessage(content="请帮我回顾一下之前的任务")],
        "objective": "What result files have been generated through previous analysis of this data?",
        "input_files": [],
        "intents": [],
        "plan": [],
        "next_step": None,
        "memory_summary": "我们已经完成数据预处理并生成了聚类结果。",
        "memory_records": [
            {
                "objective": "Run clustering",
                "created_at": "2024-05-12T10:30:00",
                "summary": "完成 PCA 与 UMAP 并标记了 8 个簇。",
                "highlights": "输出文件：analysis/sample_emb.h5ad",
            }
        ],
        "thread_id": "thread-1",
        "replan_attempts": 0,
        "max_replan_attempts": 3,
        "execution_status": "idle",
        "intent_trace": {},
        "work_dir": None,
        "tool_history": [],
        "analysis_notes": {},
        "recognized_intents": [
            Intent(
                intent="memory_query",
                description="Retrieve prior conversation context or missions from long-term memory.",
                confidence=1.0,
                dependencies=[],
                justification="Detected memory retrieval phrasing in the request.",
            ).model_dump()
        ],
        "project_state": {},
    }

    updated_state = await response(state)  # type: ignore[arg-type]
    final_message = updated_state["messages"][-1].content

    assert "长期记忆检索结果" in final_message
    assert "会话摘要" in final_message
    assert "Run clustering" in final_message
    assert "analysis/sample_emb.h5ad" in final_message
    assert "last_rag" not in updated_state["analysis_notes"]
    assert updated_state["next_step"] == "end"


@pytest.mark.asyncio
async def test_status_check_reports_progress_details():
    state = {
        "messages": [HumanMessage(content="当前任务进度如何？")],
        "objective": "Provide project status",
        "input_files": [],
        "intents": [],
        "plan": [
            "Load dataset",
            "Run clustering",
            "Summarise results",
        ],
        "next_step": None,
        "memory_summary": "",
        "memory_records": [],
        "thread_id": "thread-2",
        "replan_attempts": 1,
        "max_replan_attempts": 4,
        "execution_status": "in_progress",
        "intent_trace": {},
        "work_dir": None,
        "tool_history": [
            {
                "tool": "load_dataset",
                "status": "completed",
            },
            {
                "tool": "run_clustering",
                "status": "failed",
                "error": "File missing",
            },
        ],
        "analysis_notes": {
            "completed_steps": ["Load dataset"],
            "last_note": "正在重新准备聚类输入。",
        },
        "recognized_intents": [
            Intent(
                intent="status_check",
                description="Report the current status or progress of the active task.",
                confidence=0.9,
                dependencies=[],
                justification="Detected progress inquiry phrasing in the request.",
            ).model_dump()
        ],
        "project_state": {},
    }

    updated_state = await response(state)  # type: ignore[arg-type]
    final_message = updated_state["messages"][-1].content

    assert "项目状态更新" in final_message
    assert "当前执行状态" in final_message
    assert "Load dataset" in final_message
    assert "run_clustering" in final_message
    assert "正在重新准备聚类输入" in final_message
    assert "last_rag" not in updated_state["analysis_notes"]
    assert updated_state["next_step"] == "end"


def test_prune_completed_plan_removes_finished_steps():
    state = {
        "plan": [
            "Use the 'load_h5ad_data' tool to preprocess the dataset",
            "Run the 'run_ssgsea_enrichment' tool to compute functional enrichment",
        ],
        "project_state": {
            "active_dataset": "sample",
            "datasets": {
                "sample": {
                    "work_dir": "/tmp/sample",
                    "completed_steps": ["load_h5ad_data"],
                    "preprocessed_path": "/tmp/sample/sample_preprocessed.h5ad",
                }
            },
        },
    }

    _prune_completed_plan(state)  # type: ignore[arg-type]

    assert len(state["plan"]) == 1
    assert "run_ssgsea_enrichment" in state["plan"][0]


def test_build_project_state_message_includes_completed_steps():
    project_state = {
        "active_dataset": "sample",
        "datasets": {
            "sample": {
                "work_dir": "/tmp/sample",
                "completed_steps": ["load_h5ad_data", "cluster_and_diff"],
                "embeddings_path": "/tmp/sample/sample_emb.h5ad",
            }
        },
    }

    message = build_project_state_message(project_state)
    assert message is not None
    assert "加载与预处理数据" in message
    assert "/tmp/sample" in message
    assert "sample_emb.h5ad" in message
