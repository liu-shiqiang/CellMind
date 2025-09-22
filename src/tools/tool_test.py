from src.scripts.llm_loader import llm
from langchain_core.tools import BaseTool
from src.tools.load_h5ad import load_h5ad_data

# 导入你要测试的工具


TOOLS = [
    # 加入你要测试的工具
    load_h5ad_data,
]

# 一个最简单的agent 调用工具， 用来测试工具可用性, 代码实现参考langgraph 文档可以，是工具调用的原理

def test_tool(prompt: str):
    tools = TOOLS
    tools_by_name = {tool.name: tool for tool in tools}
   
    agent = llm.bind_tools(tools)
    decision_message = agent.invoke(prompt)
    for tool_call in decision_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"Attempting to call tool: {tool_name} with args: {tool_args}")

        if tool_name not in tools_by_name:
            error_msg = f"Tool '{tool_name}' not found in available tools."
            print(error_msg)
        
        else: 
            try:
                tool_to_call: BaseTool = tools_by_name[tool_name]
                
                if hasattr(tool_to_call, 'ainvoke'):
                    tool_result = tool_to_call.invoke(tool_args)
                else:
                    tool_result = tool_to_call.invoke(tool_args)

                result_str = f"<excute>Tool {tool_name} call result: {str(tool_result)}</excute>"
                print(f"Tool '{tool_name}' executed successfully.")

            except Exception as e:
                error_msg = f"Error calling tool '{tool_name}': {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                
    return result_str


if __name__ == "__main__":
    
    #设计你的prompt，prompt中应该包含工具调用需要的参数
    prompt = "导入该数据并进行预处理:/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/test_l3_stratified_5pct.h5ad"
    result = test_tool(prompt)
    print("Final Result:", result)






