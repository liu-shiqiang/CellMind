from langchain_experimental.plan_and_execute import (
    load_agent_executor, load_chat_planner, PlanAndExecute
)
from langchain.memory import ConversationBufferMemory
from src.agent.tool_registry import TOOLS
from src.scripts.llm_loader import ModelLoader

class Agent():
    def __init__(self, model_name, openai, chroma_path, gui_mode, cpu, rag) -> None:
        self.model_name = model_name
        self.openai = openai
        self.chroma_path = chroma_path
        self.gui_mode = gui_mode
        self.cpu = cpu
        self.rag = rag
        self.llm = ModelLoader(model_name).load_model()
    def invoke(self, user_task):
        planner = load_chat_planner(self.llm)
        executor = load_agent_executor(self.llm, TOOLS, verbose=True)
        memory = ConversationBufferMemory()
        agent = PlanAndExecute(planner=planner, executor=executor,memory=memory, verbose=True)
        result = agent.invoke(user_task)
        return result
