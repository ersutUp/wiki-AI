"""
状态创建
"""

from langchain.agents import create_agent, AgentState
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from my_llm import glm_llm

# 自定义状态 注意参数是 AgentState 类型
class UserState(AgentState):
    userName: str
    sex: str
    age: int


agent =create_agent(
    model=glm_llm,
    system_prompt="你是聊天助手",
    # 必须开启记忆， 才能使用状态
    checkpointer=InMemorySaver(),
    # 定义状态
    state_schema=UserState,
)
#保持记忆的关键固定格式，  configurable.thread_id
config = {"configurable": {"thread_id": "chat1"}}

# 流式输出
resp = agent.invoke({
    "messages": [HumanMessage("你好，我是王总")],
    "sex":"男",
    "age":18,
},stream_mode="checkpoints",config=config)
print(resp)
print(resp[-1]["values"]["messages"][-1].content)

print("-"*50)

resp = agent.invoke({"messages": [HumanMessage("我是谁, 我多大了")]},stream_mode="checkpoints",config=config)
print(resp)
print(resp[-1]["values"]["messages"][-1].content)

