from langchain_experimental.plan_and_execute import (
    load_agent_executor, load_chat_planner, PlanAndExecute
)
from langchain_ollama import ChatOllama
from agent.tool_registry import TOOLS
from pathlib import Path

LLM = ChatOllama(model="deepseek-r1:14b", base_url="http://localhost:11434")

planner_prompt = Path("prompts/plan_prompt.txt").read_text()
planner = load_chat_planner(LLM, planner_prompt)
executor = load_agent_executor(LLM, TOOLS, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
