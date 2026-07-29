"""
状态获取
"""

from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command

from my_llm import glm_llm


# 自定义状态 注意参数是 AgentState 类型
class UserState(AgentState):
    userName: str
    sex: str
    age: int
    count_model: int
    count_call_age_tool: int

@tool
def get_age(runtime: ToolRuntime) -> Command:
    """获取用户的年龄。"""
    print(f" tool runtime {runtime}")
    age = runtime.state["age"]
    print(f" tool get age = {age}")

    count_call_age_tool = runtime.state.get("count_call_age_tool",0)
    count_call_age_tool += 1

    # 从状态中获取年龄 并 修改状态
    return Command(update={
        "count_call_age_tool": count_call_age_tool,
        "messages": [ToolMessage(
            content=f"用户年龄是 {age}",
            tool_call_id=runtime.tool_call_id,
        )],
    })

# 模型调用前后执行
@before_model
def my_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"before_model state {state}")
    # print(f"before_model get age = {state['age']}")

    # 计算模型调用次数
    count_model = state.get("count_model", 0)
    count_model += 1

    # 更新状态
    return {"count_model": count_model}

# 模型调用后执行
@after_model
def my_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"after_model state {state}")
    # print(f"after_model get age = {state['age']}")
    #这里也可以更新状态
    return None


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

print("-"*500)
# 获取状态
print(agent.get_state(config=config))
print("-"*500)

respChunks = agent.stream({"messages": [HumanMessage("我是谁, 我多大了")]},config=config)
for chunk in respChunks:
    print(f"\nrespChunks2 {chunk}\n")

print("-"*500)
# 获取状态
print(agent.get_state(config=config))
print("-"*500)