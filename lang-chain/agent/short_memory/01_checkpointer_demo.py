
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from my_llm import funcloude_claude_llm

agent =create_agent(
    model=funcloude_claude_llm,
    system_prompt="你是聊天助手",
    # 开启记忆 保存在内存中
    checkpointer=InMemorySaver(),
)
#保持记忆的关键固定格式，  configurable.thread_id
config = {"configurable": {"thread_id": "chat1"}}

# 流式输出
resp = agent.invoke({"messages": [HumanMessage("你好，我是王总")]},stream_mode="checkpoints",config=config)
print(resp)
print(resp[-1]["values"]["messages"][-1].content)

print("-"*50)

resp = agent.invoke({"messages": [HumanMessage("我是谁")]},stream_mode="checkpoints",config=config)
print(resp)
print(resp[-1]["values"]["messages"][-1].content)

