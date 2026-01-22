"""
响应生成节点
生成最终回复，支持记忆查询、状态检查、数据集跟踪和RAG增强
"""
import logging
from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agent.state import AgentState
from src.utils.llm_manager import get_llm

logger = logging.getLogger(__name__)


# ============== 辅助函数 ==============

def _extract_latest_user_question(messages: list, fallback: str = "") -> str:
    """提取最新的用户问题"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = (message.content or "").strip()
            if content and not content.startswith("user dataset available at:"):
                return content
    return fallback


def _format_memory_response(state: AgentState) -> str:
    """构建记忆查询回复"""
    summary = (state.get("memory_summary") or "").strip()
    records = state.get("memory_records") or []

    if not summary and not records:
        return (
            "当前会话没有可供检索的长期记忆记录。如需我在未来记住关键信息，"
            "请明确告知我哪些结论或文件需要保存。"
        )

    lines = ["### 长期记忆检索结果"]

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


def _format_status_response(state: AgentState) -> str:
    """构建状态查询回复"""
    status_text = (state.get("execution_status") or "未开始").strip() or "未开始"
    lines = ["### 项目状态更新", f"- 当前执行状态：{status_text}"]

    remaining_plan = [step for step in state.get("plan", []) if isinstance(step, str) and step.strip()]
    if remaining_plan:
        lines.append("- 待执行步骤：")
        for idx, step in enumerate(remaining_plan, 1):
            lines.append(f"  {idx}. {step.strip()}")

    tool_history = state.get("tool_history") or []
    if tool_history:
        lines.append("- 最近的工具调用：")
        for entry in tool_history[-3:]:
            description_parts = []
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


def _build_dataset_followup_message(dataset_result: Dict[str, Any], work_dir: Optional[str]) -> str:
    """构建数据集分析完成后的跟进消息"""
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
        local_refs_val = local_refs if isinstance(local_refs, int) else (len(local_refs or []) if isinstance(local_refs, list) else 0)
        pubmed_refs_val = pubmed_refs if isinstance(pubmed_refs, int) else (len(pubmed_refs or []) if isinstance(pubmed_refs, list) else 0)
        summary_lines.append(f"- 本地知识库引用：{local_refs_val} 条")
        summary_lines.append(f"- PubMed 摘要引用：{pubmed_refs_val} 条")

    summary_lines.append("")
    summary_lines.append("如果你还有其他生物信息学问题，请继续提问，我会结合知识库和对话历史继续解答。")
    return "\n".join(summary_lines)


def _generate_bio_rag_answer(question: str, state: AgentState) -> tuple[str, Dict[str, Any]]:
    """生成RAG增强的答案"""
    try:
        try:
            from config.setting import settings
            top_k_local = int(getattr(settings, "RETRIVE_TOP_K", 3))
        except (TypeError, ValueError, AttributeError):
            top_k_local = 3
        top_k_local = max(1, top_k_local)

        work_dir = state.get("work_dir")

        try:
            from src.tools.dataset_qa import retrieve_bio_context
            references = retrieve_bio_context(
                question,
                work_dir=work_dir,
                top_k_local=top_k_local,
                top_k_pubmed=3,
            )
        except Exception as exc:
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
            dataset_summary = []
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

        llm = get_llm()
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

    except Exception as exc:
        logger.exception("[Response] RAG answer generation failed: %s", exc)
        error_msg = f"抱歉，生成答案时出现问题：{exc}"
        return error_msg, {"error": str(exc)}


def _format_execution_summary(state: AgentState) -> Optional[str]:
    """Build a user-facing summary based on executed tool results."""
    analysis_notes = state.get("analysis_notes") or {}
    tool_runs = analysis_notes.get("tool_runs") or []
    if not tool_runs:
        return None

    execution_status = state.get("execution_status")
    failed = next((entry for entry in tool_runs if entry.get("status") == "error"), None)
    completed_tools = [entry.get("tool") for entry in tool_runs if entry.get("status") != "error"]
    artifacts: list[str] = []
    for entry in tool_runs:
        artifacts.extend(entry.get("artifacts") or [])
    artifacts = [path for path in artifacts if isinstance(path, str)]

    lines: list[str] = []
    if failed or execution_status == "failed":
        reason = failed.get("message") if failed else (state.get("error_info") or {}).get("detail")
        reason = reason or "执行过程中发生错误。"
        lines.append("本次分析未完成。")
        lines.append(f"- 失败步骤：{failed.get('tool') if failed else '未知'}")
        lines.append(f"- 原因：{reason}")
        if completed_tools:
            lines.append(f"- 已完成步骤：{', '.join(completed_tools)}")
        if artifacts:
            lines.append("- 已产出文件：")
            for path in artifacts:
                lines.append(f"  • {path}")
        lines.append("请确认是否需要我尝试自动修复数据格式问题（例如基因名冲突），或直接停止本次分析。")
        return "\n".join(lines)

    lines.append("分析完成。")
    if completed_tools:
        lines.append(f"- 完成步骤：{', '.join(completed_tools)}")
    if artifacts:
        lines.append("- 产出文件：")
        for path in artifacts:
            lines.append(f"  • {path}")
    return "\n".join(lines)


# ============== 响应生成节点 ==============

async def response_node(state: AgentState) -> AgentState:
    """
    响应生成节点
    生成最终回复，支持多种回复模式：
    - 记忆查询：返回长期记忆信息
    - 状态查询：返回项目状态
    - 数据集跟进：返回分析完成信息
    - RAG增强：使用知识库增强答案
    """
    analysis_notes = state.setdefault("analysis_notes", {})
    final_message: Optional[str] = None

    try:
        # 获取识别的意图标签
        recognized_labels = _get_recognized_intent_labels(state)
        dataset_result = analysis_notes.get("dataset_bio_qa")

        # 根据意图类型生成不同的回复
        if "memory_query" in recognized_labels:
            final_message = _format_memory_response(state)
        elif "status_check" in recognized_labels:
            final_message = _format_status_response(state)
        elif dataset_result:
            work_dir = dataset_result.get("work_dir") or state.get("work_dir")
            final_message = _build_dataset_followup_message(dataset_result, work_dir)
        else:
            execution_summary = _format_execution_summary(state)
            if execution_summary:
                final_message = execution_summary
            else:
                # 默认：使用RAG增强回答
                question = _extract_latest_user_question(
                    state.get("messages", []),
                    state.get("objective", ""),
                )
                if question:
                    answer_text, rag_metadata = _generate_bio_rag_answer(question, state)
                    analysis_notes["last_rag"] = {
                        **rag_metadata,
                        "answer": answer_text,
                    }
                    final_message = answer_text.strip()
                else:
                    logger.warning("[Response] No user question detected; falling back to default prompt.")

    except Exception as exc:
        logger.exception("[Response] Structured answer generation failed: %s", exc)
        final_message = None

    # 如果没有生成消息，使用默认prompt
    if not final_message:
        response_prompt = (
            "You are a helpful biomedical assistant who can view the Q&A history and give professional answers to the "
            "user's questions. If the latest messages contain execution errors, propose concrete next steps."
        )
        conversation_history = list(state.get("messages", []))
        if not conversation_history or conversation_history[-1].type != "human":
            objective = state.get("objective")
            if objective:
                conversation_history.append(HumanMessage(content=objective))

        llm = get_llm()
        fallback_messages = [SystemMessage(content=response_prompt)] + conversation_history
        fallback_response = llm.invoke(fallback_messages)
        final_message = getattr(fallback_response, "content", str(fallback_response))
        if isinstance(final_message, list):
            final_message = "\n".join(str(part) for part in final_message)

    # 添加结束语
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

    logger.info(f"[Response] Response content: {final_message[:100]}...")
    return state


# ============== 导出函数供其他模块使用 ==============

def _get_recognized_intent_labels(state: AgentState) -> list[str]:
    """返回当前轮次识别到的意图标签"""
    from src.agent.nodes.intent import _normalize_intent_label

    raw_intents = state.get("recognized_intents", []) or []
    labels: list[str] = []

    for item in raw_intents:
        if isinstance(item, str):
            labels.append(item)
        elif isinstance(item, dict):
            labels.append(_normalize_intent_label(item.get("intent", "")))
        elif hasattr(item, "intent"):
            labels.append(_normalize_intent_label(getattr(item, "intent", "")))

    return labels
