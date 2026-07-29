from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from my_llm import chat_llm


@tool
def getweather(location: str) -> str:
    """Get the current weather in a given location."""
    print(f"调用工具：{location}")
    return f"今天天气 {location} 是晴天."

agent = create_agent(
    model=chat_llm,
    system_prompt="你是天气查询助手",
    tools=[getweather],
)

human_message = HumanMessage(content="你好，我想知道北京的天气")
msgList = [human_message]

# 内部会自动调用工具
res1 = agent.invoke({"messages": msgList})
print(res1)
# res1 是一个 dict，其中 "messages" 是整个对话历史的列表（含 Human/AI/Tool 等所有消息）。
# 想只拿到 AI 最后的回复，取 messages 列表的最后一个元素即可。
last_message = res1["messages"][-1]
print(last_message.content)
