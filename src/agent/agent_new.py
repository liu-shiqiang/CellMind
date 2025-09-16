import re
import json
import operator
from typing import Annotated, List, Tuple, Union, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal
import logging
import structlog

from langchain import hub
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.exceptions import OutputParserException
from langgraph.graph import END
from langgraph.graph import StateGraph, START 
from langgraph.checkpoint.memory import MemorySaver


from src.agent.tool_registry import TOOLS 
from src.utils.llm_manager import LLMManager
from src.utils.path_manager import path_manager, extract_paths_from_objective, validate_h5ad_file, create_analysis_work_dir


structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

memory = MemorySaver()

tools = TOOLS

tools_by_name = {tool.name: tool for tool in tools}

llm = ChatOllama(model="qwen3:30b", 
                 temperature=0.6,
                 base_url="http://localhost:11434"
                 )

llm_tool = ChatOllama(model="qwen3:8b", 
                 temperature=0.6,
                 base_url="http://localhost:11434"
                 )

llm_with_tools = llm_tool.bind_tools(tools)
print("LLM绑定的工具名称：", [tool["function"]["name"] for tool in llm_with_tools.kwargs["tools"]])
class Intent(BaseModel):
    """Intent recognition schema"""
    description: str = Field(description="Detailed task description")
    confidence: float = Field(default=0.8, ge=0, le=1, description="Recognition confidence level (0-1)")
    dependencies: List[str] = Field(
        default_factory=list, 
        description="Prerequisite intention of dependency"
    )

    @field_validator('confidence')
    def round_confidence(cls, v):
        """Confidence level to two decimal places"""
        return round(v, 2)

class IntentResponse(BaseModel):
    """Intent recognition response"""
    intents: List[Intent] = Field(description="List of Identified Intentions")
    is_task: bool = Field(default=True, description="Does the user intend to generate a plan and call tools to complete it")

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class AgentState(TypedDict):
    """Intelligent agent state, used to support complex tasks and dynamic programming"""
    messages: list[BaseMessage]
    objective: str                    
    input_files: List[str]           
    intents: List[Intent]
    plan: List[str]
    next_step: Optional[str]

# User intent recognition
async def intent_recognition(state: AgentState) -> AgentState:
    """
    Universal intent recognition: using LLM to dynamically recognize user intent, capable of analyzing multiple user intentions and potential user intentions
    """
    objective = state["objective"]
    state["messages"].insert(0, HumanMessage(content=f"User dataset available at: {state['input_files'][0]}"))

    intent_prompt = """
You are a helpful biological data analysis assistant responsible for analyzing user intent.
Your task is to analyze user intent using a step-by-step approach.

Firstly, analyze whether the user request requires the execution of bioinformatics analysis tasks. 
If bioinformatics analysis tasks are not required, the user's intention is determined as a 'direct response'.  
If bioinformatics analysis tasks are required, analyze the bioinformatics research tasks included in the user request. 

You need to analyze multiple intents that may be included in user requests and label their execution order dependencies.

"""
    
    try:
        intent_llm = llm.with_structured_output(IntentResponse)
        state["messages"].append(HumanMessage(content=objective)) 
        messages = [SystemMessage(content = intent_prompt)] + state["messages"]
        intent_result = intent_llm.invoke(messages)
        
        logging.info(f"[Intent Recognition] Intent recognition result: {intent_result}")

        state["intents"] = intent_result.intents
        state["messages"].append(AIMessage(content=json.dumps(intent_result.model_dump())))
        
        if intent_result.is_task:
            state["next_step"] = "planner"
        else:
            state["next_step"] = "response"

        logger.info(f"[Intent Recognition] Correctly identify intent: {intent_result.intents}")
        
    except Exception as e:
        logger.error(f"[Intent Recognition] Intention recognition failed: {e}")
        error_msg = f"Intention recognition failed: {str(e)}"
        state["messages"].append(AIMessage(content=error_msg))
        state["next_step"] = "response"
    
    return state

async def response(state: AgentState):

    response_prompt = """
You are a helpful biomedical assistant who can view the user's Q&A history and give professional answers to the user's questions.

You can observe the environment and identify potential errors that may occur during program execution. When you notice that the most recent message in history is a program error, provide users with possible solutions

Requirement: You must give an answer.

"""
    message = [SystemMessage(content = response_prompt)] + state["messages"]
    response = llm.invoke(message)

    logger.info(f"[Response] Response content: {response.content}")

    state["messages"].append(AIMessage(content=response.content))

    state["next_step"] = "end"

    return state


async def general_planner(state: AgentState) -> AgentState:
    """
    Dynamically generate execution plans based on intent and available tools
    """
    logger.info("[General Planner] Start planning based on user intent and available tools")
    input_files = state.get("input_files", [])
    
    # Create a working directory
    if input_files and "input_file_info" in state:
        print(state["input_file_info"])
        first_file_info = state["input_file_info"][0]
        base_name = first_file_info["file_name"].replace('.h5ad', '')
        work_dir = create_analysis_work_dir(base_name)
        state["work_dir"] = str(work_dir)
        logger.info(f"[General Planner] Create a working directory {work_dir}")
    def tools_to_string(tools):
        result = ["Available tools:\n"]
        for i, tool in enumerate(tools, 1):
            
            name = tool.name
            description = getattr(tool, 'description', 'No description')
            
            lines = [
                f"{i}. Name: {name}",
                f"   Description: {description}"
            ]
            result.append("\n".join(lines))
        return "\n\n" + "\n".join(result) + "\n"
    
    tools_info = tools_to_string(tools)

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
            plan_llm = llm.with_structured_output(Plan)

            plan_prompt =plan_prompt_template.format(
                tools_info=tools_info
            )

            print(f"plan_prompt{plan_prompt} type:{type(plan_prompt)}")

            logger.info(f"[General Planner] Attempting to generate plan (Attempt {attempt + 1}/{max_attempts})")
            if attempt > 0 and previous_errors_feedback:
                feedback_prompt = f"""
IMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT (Attempt {attempt}):
The plan generated in the previous attempt had the following issues:
{previous_errors_feedback}

Please carefully review the feedback above and regenerate the plan, ensuring it adheres strictly to the required JSON format and content guidelines.
"""
                plan_prompt = plan_prompt + "\n\n" + feedback_prompt 
            
            state["messages"].append(HumanMessage(content=plan_prompt))

            plan_messages = state["messages"]
            plan = plan_llm.invoke(plan_messages)
            logger.info(f"[General Planner] Plan generation result: {plan}")

            if plan and isinstance(plan.steps, list) and len(plan.steps) > 0:
                # Check if steps are non-empty
                non_empty_steps = [step for step in plan.steps if step.strip()]
                if non_empty_steps:
                    plan_generation_success = True
                    state["plan"] = non_empty_steps
                    logger.info("[General Planner] Plan generated successfully.")
                    previous_errors_feedback = ""
                    break
                else:
                    error_msg = "Plan generation produced an object but steps list is invalid or empty."
                    logger.warning(f"[General Planner] Plan generated but all steps are empty (Attempt {attempt + 1}).")
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

    if plan_generation_success and plan:
        plan_json_str = plan.model_dump_json(indent = 2)
        plan_message_content = f"<PLAN_GENERATED>\n{plan_json_str}\n</PLAN_GENERATED>"
        state["messages"].append(AIMessage(content=plan_message_content))
        state["next_step"] = "general_executor"

    else:
        error_msg = f"Failed to generate a valid plan after {max_attempts} attempts. Final error feedback: {previous_errors_feedback}"
        logger.error(f"[General Planner] {error_msg}")
        state["messages"].append(AIMessage(content=f"<PLAN_ERROR>{error_msg}</PLAN_ERROR>"))
        state["next_step"] = "response" 
    
    return state

async def _handle_tool_calls(state: AgentState, decision_message: AIMessage, current_step: str) -> AgentState:
    """Handles the logic for executing one or more tool calls. Modifies state in place."""

    tool_outputs = []
    tool_call_state = False

    for tool_call in decision_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"Attempting to call tool: {tool_name} with args: {tool_args}")

        if tool_name not in tools_by_name:
            error_msg = f"Tool '{tool_name}' not found in available tools."
            print(error_msg)
            
            tool_outputs.append(
                ToolMessage(
                    content=f"Error: {error_msg}",
                    name = tool_name,
                    tool_call_id=tool_call_id
                )
            )
        else: 
            try:
                tool_to_call: BaseTool = tools_by_name[tool_name]
                
                if hasattr(tool_to_call, 'ainvoke'):
                    tool_result = await tool_to_call.ainvoke(tool_args)
                else:
                    tool_result = tool_to_call.invoke(tool_args)

                result_str = f"<excute>Tool {tool_name} call result: {str(tool_result)}</excute>"
                print(f"Tool '{tool_name}' executed successfully.")

                state["plan"] = state["plan"][1:]

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
                tool_outputs.append(
                    ToolMessage(
                        content=f"Error: {error_msg}",
                        name=tool_name,
                        tool_call_id=tool_call_id
                    )
                )

    state["messages"].extend(tool_outputs)

    if tool_call_state :
        state["next_step"] = "general_executor"        
    else:
        state["next_step"] = "response"
        
    return state

            
async def general_executor(state: AgentState) -> AgentState:
    """
    Universal executor: executes the current planned step, supports tool invocation and error handling
    """
    plan = state["plan"]
    current_step = plan[0]
    
    messages = state.get("messages", [])

    formatted_history = "\n"

    if messages:
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        for i, msg in enumerate(recent_messages):
            content_preview = (msg.content[:500]+ "...")if msg.content and len(msg.content) > 500 else (msg.content or "")
            formatted_history += f" {i+1}. [{msg.type}]{content_preview}\n "
    else:
        formatted_history += " (No previous conversation history)\n"

    task_formatted = f"""
You are a task execution agent responsible for completing a specific step in a larger plan.
You can observe the contextual environment of this step, and you need to extract necessary parameters from the previous conversation history to complete the current task.
You cannot fabricate parameters for tool calls. Parameters must be included in the conversation history, otherwise you will be punished.
If there are no tools in the conversation history to call the required parameters, make a request to the user.

You need to complete this task in the current step : {current_step}

Here is the recent conversation history: This provides a complete background of the interactions to date.Read carefully to understand:
*What has already been done.
*What data or variables may already exist in the execution environment (from the previous'<execute>'block).
*Any specific instructions or feedback given by the user or observed from previous executions.

{formatted_history}

"""
    try:
        decision_message = llm_with_tools.invoke(task_formatted)
        print(f"LLM Decision Message: {decision_message}")

        if decision_message.tool_calls:
            concise_tool_call_message = AIMessage(
                content = json.dumps(decision_message.tool_calls, indent = 2),
                tool_calls = decision_message.tool_calls
            )

            state["messages"].append(concise_tool_call_message)
            state = await _handle_tool_calls(state, decision_message,current_step)

        else:
            response_content = decision_message.content.strip()

            if not response_content:
                error_msg = "LLM did not call any tool and provided no reponse content.You need to reanalyze the conversation history to complete the task"
                print(error_msg)
                state["messages"].append(AIMessage(content=f"<EXECUTION_ERROR>{error_msg}</EXECUTION_ERROR>"))
                state["next_step"] = "general_executor"
            
            print(f"LLM provided a direct response or requesr for input: {response_content}")
            
            observation_content = f"<observation>LLM response to task '{current_step}':{response_content}</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            state["next_step"] = "replanner"
            
    except Exception as e:
        error_msg = f"An unexpected error occurred in execute_step: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        state["messages"].append(AIMessage(content=f"<EXECUTION_ERROR>{error_msg}</EXECUTION_ERROR>"))
        state["next_step"] = "replanner"
        
    return state



async def intelligent_replanner(state: AgentState) -> AgentState:
    """
    Intelligent replanning: Analyze execution status, dynamically adjust plans or provide responses
    """
    logger.info("[Intelligent Replanner] Start re planning and analyzing")
    
    messages: List[BaseMessage] = state.get("messages", [])
    current_plan: List[str] = state.get("plan", [])
    current_step = current_plan[0] if current_plan else "Unknown or finished step"

    formatted_full_history = "\n"
    if messages:
        recect_messages = messages[-10:] if len(messages) > 10 else messages
        for i, msg in enumerate(recect_messages):
            content_preview = (msg.content[:500]+ "...")if msg.content and len(msg.content) > 500 else (msg.content or "")
            formatted_full_history += f" {i+1}. [{msg.type}]{content_preview}\n "
    else:
        formatted_full_history += " (No previous conversation history)\n"

    replan_prompt = f"""
You are an intelligent replan and analysis agent. Your task is to evaluate why the executor has entered the replanning phase and determine the correct next action.
The executor was tasked with tperforming the following step in the plan: {current_step} 
It has now entered replanning.This can happen for several reasons:
1.The step was **successfully completed** (e.g., a tool was called and its result can be observed in the conversation history).
2.The executor **failed** or encountered an error.
3.The executor **requested specific information** from the user to proceed.
4. The **overall plan is flawed** and needs revision.

**Your Task:**
1.  Carefully review the **Full Conversation History** provided below.
2.  Determine the most likely reason the executor entered replanning by analyzing the recent messages:
    *   **Success Indicators**: Look for a sequence like:
        - An `AIMessage` containing a JSON list of tool calls (this is the executor's action).
        - Followed by one or more `ToolMessage`(s) containing the output/result of the tool execution.
        This sequence strongly indicates successful completion of a tool-based step.
    *   **Failure/Error Indicators**: Look for `<EXECUTION_ERROR>` messages or AI messages indicating problems.
    *   **Request for Input Indicators**: Look for AI messages that seem to be direct requests for user input (e.g., "Please provide...", "Could you specify...").
    *   **Plan Flaw Indicators**: Consider if the step itself or the plan's logic seems fundamentally problematic.
3.  Based on your analysis, decide the most appropriate next step.

**Your Response MUST be ONE of the following JSON formats ONLY:**

**Option 1: Step Completed Successfully**
If the analysis shows a tool was successfully called and its result was observed, indicating the current step is done.
```json
{{
  "action": "step_completed",
  "reasoning": "The executor successfully called the tool(s) for '{current_step}'. This is evidenced by the presence of an AIMessage with tool call details followed by a ToolMessage containing the execution result. The step is complete."
}}
Option 2: Request Information from User
If the executor cannot proceed because it needs specific information that only the user possesses.
{{
  "action": "request_user_input",
  "request_message": "Please provide the path to the .h5ad file you want to analyze.",
  "reasoning": "The executor failed because the 'load_h5ad_data' tool requires a 'file_path' argument. This argument was not found in the conversation history, indicating the user must provide it."
}}
Option 3: Generate a New Plan
If the current plan is fundamentally flawed, unachievable, or based on incorrect premises.
{{
  "action": "regenerate_plan",
  "new_plan": [
    "Clarify the user's objective and required inputs (e.g., dataset file path).",
    "Load the specified dataset using the provided file path.",
    "Perform initial data quality checks.",
    "Conduct the main analysis task as requested."
  ],
  "reasoning": "The original plan started with loading data, but the necessary file path was unknown. A new plan that begins with gathering this essential information is required for success."
}}
Option 4: Indicate Plan Can Continue (Less Common)
If after analysis, you conclude that the executor's issue was a misunderstanding or has been resolved, and the plan can proceed with the current state.
{{
  "action": "continue_plan",
  "reasoning": "The executor's request for input appears to be based on outdated information. The file path 'data/sample.h5ad' was provided earlier and should be usable."
}}
IMPORTANT INSTRUCTIONS:

Your entire response must be a single, valid JSON object matching one of the four formats above.
Do not include any other text, explanations, or markdown formatting outside the JSON.
Base your decision and all reasoning strictly on the provided Full Conversation History.
Be precise and factual in your reasoning.
For request_user_input, ensure the request_message is user-friendly and specific.
For regenerate_plan,
ensure the new_plan is a list of clear, discrete, and actionable steps.
Full Conversation History:
{formatted_full_history}
"""
    try:
        decision_message: AIMessage = llm.invoke(replan_prompt)
        decision_content = decision_message.content.strip()
        print(f"Replanner Decision: {decision_content}")

        decision_data = json.loads(decision_content)
        action = decision_data.get("action")

        if action == "step_completed":
            reason = decision_data.get("reasoning", "No reasoning provided")
            print(f"[Intelligent Replanner] Step completed successfully: {reason}")

            observation_content = f"<observation>Replanner confirmed step'{current_step}' completion. Reason: {reason}.Advancing to next step.</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            if current_plan:
                updated_plan = current_plan[1:]
                state["plan"] = updated_plan
                print(f"Updated plan: {updated_plan}")

        elif action == "request_user_input":
            request_message = decision_data.get("request_message", "Could you please provide more information?")
            reason = decision_data.get("reasoning", "No reasoning provided")
            print(f"[Intelligent Replanner] Requesting user input: {request_message} (Reason: {reason})")

            state["messages"].append(AIMessage(content=f"<USER_REQUEST>{request_message}</USER_REQUEST>"))
            
            user_input = input(f"f{request_message}")
            state["messages"].append(AIMessage(content=f"<USER_INPUT>{user_input}</USER_INPUT>"))
            print(f"User input received: {user_input}")
            state["next_step"] = "general_executor"

        elif action == "regenerate_plan":
            new_plan = decision_data.get("new_plan", [])
            reasoning = decision_data.get("reasoning", "No reasoning provided")
            print(f"[Intelligent Replanner] Regenerating plan: {new_plan} (Reason: {reason})")

            if not isinstance(new_plan, list) or not new_plan:
                error_msg = "Replanner generated an invalid or empty new plan."
                print(f"[Intelligent Replanner] {error_msg}")
                state["messages"].append(AIMessage(content=f"<PLAN_ERROR>{error_msg}</PLAN_ERROR>"))
                state["next_step"] = "response"
            plan_summary = "\n".join([f"{i+1}. {step}" for i, step in enumerate(new_plan)])
            observation_content = f"<observation>Replanner generated a new plan. Reasoning: {reasoning}\nNew Plan:\n{plan_summary}</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            state["plan"] = new_plan
            state["next_step"] = "general_executor"
        elif action == "continue_plan":
            reasoning = decision_data.get("reasoning", "")
            print(f"Replanner decided plan can continue: {reasoning}")

            observation_content = f"<observation>Replanner assessed the situation: {reasoning}. Decided to continue with the current plan.</observation>"
            state["messages"].append(AIMessage(content=observation_content))
            state["next_step"] = "general_executor"
        else:
            error_msg = f"Replanner returned an invalid action: {action}"
            print(error_msg)
            state["messages"].append(AIMessage(content=f"<REPLAN_ERROR>{error_msg}</REPLAN_ERROR>"))
            state["next_step"] = "replanner"
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse Replanner's JSON decision: {e}. Content: {decision_content[:200]}..."
        print(error_msg)
        state["messages"].append(AIMessage(content=f"<REPLAN_ERROR>{error_msg}</REPLAN_ERROR>"))
        state["next_step"] = "replanner"
    except Exception as e:
        error_msg = f"An unexpected error occurred in replanner: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        state["messages"].append(AIMessage(content=f"<REPLAN_ERROR>{error_msg}</REPLAN_ERROR>"))
        state["next_step"] = "response"
    return state


def route_after_intent(state: AgentState) -> str:
    """"""
    if state["next_step"]=="planner":
        return "planner"  
    else:
        return "response" 


def route_after_replan(state: AgentState) -> str:
    
    if state["execution_status"] == "completed":
        return END
    elif state["execution_status"] == "failed":
        return END
    else:
        return "general_executor"
def build_graph():

    g = StateGraph(AgentState)
    
    g.add_node("intent_recognition", intent_recognition)
    g.add_node("response", response) # Note: Typo in function name 'response'
    g.add_node("general_planner", general_planner)
    g.add_node("general_executor", general_executor)
    g.add_node("intelligent_replanner", intelligent_replanner)
    
    
    g.add_edge(START, "intent_recognition")
    g.add_conditional_edges(
        "intent_recognition",
        route_after_intent,
        {
            "planner": "general_planner",
            "response": "response",
        }
    )
    g.add_conditional_edges(
        "general_planner",
        lambda state: state["next_step"],
        {
            "general_executor": "general_executor",
            "response": "response",
            
        }
    )
    g.add_conditional_edges(
        "general_executor",
        lambda state: state["next_step"],
        {
            "general_executor": "general_executor", # For retries or continuing plan
            "replanner": "intelligent_replanner",
        }
    )

    g.add_conditional_edges(
        "intelligent_replanner",
        lambda state: state["next_step"],
        {
            "general_executor": "general_executor", # If step completed or plan continues
            "response": "response", # If replan fails or decides to end with response
            # Add "replanner" if you want to allow replan -> replan (e.g., on error)
            "replanner": "intelligent_replanner", 
        }
    )
    g.add_edge("response", END)
    
    return g.compile(checkpointer=memory)


async def run_objective(objective: str, input_files: Optional[List[str]] = None):
    """运行通用智能体"""
    logger.info(f"[Agent] 开始执行任务: {objective}")
    if input_files:
        logger.info(f"[Agent] 输入文件: {input_files}")
    

    initial_state = AgentState(
        objective=objective,
        messages=[],
        input_files=input_files or [],
        intent=[],
        plan=[],
        next_step=None,
        
    )

    graph = build_graph()
    async for event in graph.astream(
        initial_state, 
        config={"recursion_limit": 50, "configurable": {"thread_id": "1"}}
    ):
        for k, v in event.items():
            if k != "__end__":
                if k == "final_response" and v:
                    logger.info(f"[Agent] 任务完成: {v}")
                    return v
                elif k == "error_info" and v:
                    logger.error(f"[Agent] 任务失败: {v}")
                    return f"任务执行失败: {v}"
    
    return "任务执行完成"

if __name__ == "__main__":
    import asyncio
    
# 修改测试用例：明确任务目标
    test_objectives = [
        "从输入的scRNA-seq数据中获取CD8-positive, alpha-beta cytotoxic T cell的Top50 Marker基因，并对这些基因进行通路富集分析"
    ]
    
    for i, objective in enumerate(test_objectives):
        print(f"\n=== 测试用例 {i+1} ===")
        print(f"输入任务: {objective}")
        # 输入文件：替换为你的h5ad文件路径
        files = ["/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_l3_stratified_5pct/test_l3_stratified_5pct_annotated.h5ad"]
        result = asyncio.run(run_objective(objective, files))
        print(f"结果: {result}")
        print("=" * 50)



        '''
   # 测试用例
    test_objectives = [
        # "对scRNA-seq数据进行预处理",
        "Annotate cell types on scRNA seq data",
        # "推断scRNA-seq数据的调控网络",
        # "对scRNA-seq数据进行通路富集分析",
        # "对scRNA-seq数据进行细胞注释并做通路分析"
    ]
    
    for i, objective in enumerate(test_objectives):
        print(f"\n=== 测试用例 {i+1} ===")
        print(f"输入: {objective}")
        files = [
            "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/scgpt/cell_anno/ms/c_data.h5ad"
        ]
        result = asyncio.run(run_objective(objective, files))
        print(f"结果: {result}")
        print("=" * 50)
        '''