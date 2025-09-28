import requests
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
 
from langchain_core.messages import HumanMessage,SystemMessage
# Import relevant functionality
# from langchain.chat_models import init_chat_model
# from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.prebuilt import create_react_agent
from tools import multiply, add, exponentiate
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from KEY import Key


memory = MemorySaver() #记忆模块
llm = ChatOpenAI(
    openai_api_base="https://api.siliconflow.cn/v1/",
    openai_api_key=Key,    # app_key
    model_name="Qwen/Qwen2.5-7B-Instruct",   # 模型名称
)
 
# messages = [
#     SystemMessage(content="把这段话从中文翻译成意大利语"),
#     HumanMessage(content="你好")
# ]


tools = [multiply, add, exponentiate]

# agent_executor = create_react_agent(llm, tools, checkpointer=memory)

prompt = hub.pull("hwchase17/openai-tools-agent")
print("Agent Prompt:", prompt)

# 构建工具调用智能体
agent = create_openai_tools_agent(llm, tools, prompt)

# 创建智能体执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用智能体执行复杂计算
output = agent_executor.invoke({"input": "调用开药工具，病患信息，年龄：43岁，现病史：间有失眠，舌苔脉象：舌齿印 苔薄白 脉弦偏弱"})

print("Agent Output:", output)