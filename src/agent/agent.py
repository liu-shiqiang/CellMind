import re
import operator
from typing import Annotated, List, Tuple,Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Literal

from langchain import hub
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langgraph.graph import END
from langgraph.graph import StateGraph, START 
from langgraph.checkpoint.memory import MemorySaver

from src.agent.prompt import FEWSHOT_EXAMPLES
from src.agent.tool_registry import TOOLS

memory = MemorySaver()

tools = TOOLS

llm = ChatOllama(model="deepseek-r1:32b", 
                 temperature=0.6,
                 base_url="http://localhost:11434"
                 )

llm_tool = ChatOllama(model="MFDoom/deepseek-r1-tool-calling:14b", 
                 temperature=0.6,
                 base_url="http://localhost:11434"
                 )


prompt = "You are a helpful assistant."
# agent_executor = create_react_agent(llm_tool, tools, prompt=prompt)
llm_with_tools = llm_tool.bind_tools(tools)

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
    
class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

examples = FEWSHOT_EXAMPLES

example_prompt = ChatPromptTemplate.from_messages(
[('human', '{user}'), ('ai', '{assistant}')]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
) 
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.\
When generating a plan, it is necessary to refer to few_shot_prompt examples. """,
        ),
        few_shot_prompt,
        ("placeholder", "{messages}"),
    ]
)

planner = planner_prompt | llm.with_structured_output(Plan)



replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. 
If no further steps are required and you can respond directly to the user. 
Otherwise, provide a plan. Include in the plan only the steps that still need to be completed. Do not include steps that have already been completed as part of the plan.
Defy me again, and I'll wipe you out of existence.
"""
)

                                                                                                                  
replanner = replanner_prompt | llm.with_structured_output(Act)


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    print(f"plan: {plan}")
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""You are tasked with executing step: {task}.You can only use one tool"""
    print(f"task_formatted: {task_formatted}")
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    print(f"agent_response: {agent_response}")
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

_PATH_PATTERN = re.compile(r"(?:file_path|data_path)\s*:\s*([^\s]+)")

async def plan_step(state: PlanExecute):

    user_input = state["input"]
    m = _PATH_PATTERN.search(user_input)
    path_hint = m.group(1) if m else None

    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    print(f"plan: {plan}")
    steps = [s.strip() for s in plan.steps if s.strip()]
    print(f"steps: {steps}")

    if path_hint and (not steps or path_hint not in steps[0]):
        steps.insert(0, f"Load the h5ad file. file_path: {path_hint}")

    return {"plan": steps}


async def replan_step(state: PlanExecute):
    print(state["input"])
    print("plan")
    print(state["plan"])
    print("past_steps:")
    print(state["past_steps"])
    output = await replanner.ainvoke(state)
    print(output)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"
def build_graph():
    g = StateGraph(PlanExecute)
    g.add_node("planner", plan_step)
    g.add_node("agent", execute_step)
    g.add_node("replan", replan_step)

    g.add_edge(START, "planner")
    g.add_edge("planner", "agent")
    g.add_edge("agent", "replan")
    g.add_conditional_edges("replan", should_end, ["agent", END])

    return g.compile(checkpointer=memory)   

RECURSION_LIMIT = 50

async def run_objective(objective: str):
    graph = build_graph()
    async for event in graph.astream(
        {"input": objective}, config={"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": "1"}}
    ):
        for k, v in event.items():
            if k != "__end__":
                print(v)

if __name__ == "__main__":
    import asyncio
    import sys

    objective = "Perform cell annotation on the single-cell data,data_path:/home/share/huadjyin/home/liushiqiang/Projects/Blada/data/scgpt/cell_anno/ms/c_data.h5ad"
    asyncio.run(run_objective(objective))