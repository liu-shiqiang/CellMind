import re

from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate

from langchain_experimental.plan_and_execute.planners.base import LLMPlanner
from langchain_experimental.plan_and_execute.schema import (
    Plan,
    PlanOutputParser,
    Step,
)

from src.agent.tool_registry import TOOLS
from src.agent.prompt import FEWSHOT_EXAMPLES

SYSTEM_PROMPT = (
    "Let's first understand the problem and devise a plan to solve the problem."
    " Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
    "Please make the plan the minimum number of steps required "
    "For each step, specify the TOOL NAME and its ARGUMENTS in parentheses,"
    "to accurately complete the task. If the task is a question, "
    "the final step should almost always be 'Given the above steps taken, "
    "please respond to the users original question'. "
    "At the end of your plan, say '<END_OF_PLAN>'"
)

tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in TOOLS)

class PlanningOutputParser(PlanOutputParser):
    """Planning output parser."""

    def parse(self, text: str) -> Plan:
        steps = [Step(value=v) for v in re.split("\n\s*\d+\. ", text)[1:]]
        return Plan(steps=steps)


def load_chat_planner(
    llm: BaseLanguageModel, system_prompt: str = SYSTEM_PROMPT
) -> LLMPlanner:
    """
    Load a chat planner.

    Args:
        llm: Language model.
        system_prompt: System prompt.

    Returns:
        LLMPlanner
    """
    examples = FEWSHOT_EXAMPLES

    example_prompt = ChatPromptTemplate.from_messages(
    [('human', '{user}'), ('ai', '{assistant}')]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    ) 
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt+"\n\nAvailable tools:\n" + tool_descriptions),
            few_shot_prompt,
            ('human', '{input}'),
        ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return LLMPlanner(
        llm_chain=llm_chain,
        output_parser=PlanningOutputParser(),
        stop=["<END_OF_PLAN>"],
    )
