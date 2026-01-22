"""
Agent服务
Agent模式的分析服务
"""
import asyncio
from datetime import datetime, timezone
from typing import Callable, Dict, Any, List, Optional
from uuid import uuid4

from src.agent.graph import get_agent_graph, create_initial_state
from src.web.schemas import AgentRunRequest


class AgentService:
    """Agent执行服务"""

    async def run_agent(
        self,
        objective: str,
        input_files: List[str],
        session_id: str,
        event_handler: Callable
    ) -> Dict[str, Any]:
        """
        运行Agent

        Args:
            objective: 用户目标
            input_files: 输入文件列表
            session_id: 会话ID
            event_handler: 事件处理回调

        Returns:
            最终结果字典
        """
        run_id = str(uuid4())
        thread_id = run_id

        # 发送启动事件
        await event_handler({
            "type": "start",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"objective": objective, "sessionId": session_id}
        })

        # 创建初始状态
        initial_state = create_initial_state(
            objective=objective,
            input_files=input_files,
            thread_id=thread_id,
            session_id=session_id,
            run_id=run_id
        )

        # 获取图并执行
        graph = get_agent_graph()

        try:
            final_state = None
            async for event in graph.astream(
                initial_state,
                config={
                    "recursion_limit": 50,
                    "configurable": {"thread_id": thread_id}
                }
            ):
                # 处理事件并发送给前端
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        continue

                    final_state = node_output

                    # 转换并发送事件
                    sse_event = self._create_node_event(
                        run_id, session_id, node_name, node_output
                    )
                    if sse_event:
                        await event_handler(sse_event)

            # 返回最终结果
            return {
                "run_id": run_id,
                "final_message": final_state.get("messages", [])[-1] if final_state else None,
                "status": "completed"
            }

        except Exception as e:
            # 发送错误事件
            await event_handler({
                "type": "error",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"error": str(e)}
            })
            return {"run_id": run_id, "status": "failed", "error": str(e)}

    def _create_node_event(
        self,
        run_id: str,
        session_id: str,
        node_name: str,
        output: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """从节点输出创建SSE事件"""
        if node_name == "intent_recognition":
            return {
                "type": "intent_recognized",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "intents": output.get("intents", []),
                    "sessionId": session_id
                }
            }

        elif node_name == "planner":
            return {
                "type": "plan_generated",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {"plan": output.get("plan", [])}
            }

        elif node_name == "executor":
            # 检查是否有工具调用结果
            messages = output.get("messages", [])
            if messages and messages[-1]:
                return {
                    "type": "step_completed",
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {"output": str(messages[-1].content)[:200]}
                }

        return None
