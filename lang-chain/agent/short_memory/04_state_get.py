"""
状态获取
"""

from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

from my_llm import glm_llm

@tool
def get_age(runtime: ToolRuntime) -> int:
    """获取用户的年龄。"""
    print(f"tool runtime {runtime}")

    age = runtime.state["age"]

    print(f" tool get age = {age}")
    # 从状态中获取年龄
    return runtime.state["age"]

# 模型调用前后执行
@before_model
def my_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"before_model state {state}")
    print(f"before_model get age = {state['age']}")
    return None

# 模型调用后执行
@after_model
def my_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"after_model state {state}")
    print(f"after_model get age = {state['age']}")
    return None


# 自定义状态 注意参数是 AgentState 类型
class UserState(AgentState):
    userName: str
    sex: str
    age: int


agent =create_agent(
    model=glm_llm,
    system_prompt="你是聊天助手",
    tools=[get_age],
    middleware=[my_before_model,my_after_model],
    # 开启记忆 保存在内存中
    checkpointer=InMemorySaver(),
    state_schema=UserState,
)
#保持记忆的关键固定格式，  configurable.thread_id
config = {"configurable": {"thread_id": "chat1"}}

# 流式输出
respChunks = agent.stream({
    "messages": [HumanMessage("你好，我是王总")],
    "sex":"男",
    "age":18,
},config=config)
for chunk in respChunks:
    print(f"\nrespChunks1 {chunk}\n")

print("-"*50)

respChunks = agent.stream({"messages": [HumanMessage("我是谁, 我多大了")]},config=config)
for chunk in respChunks:
    print(f"\nrespChunks2 {chunk}\n")

