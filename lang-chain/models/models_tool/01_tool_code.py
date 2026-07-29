from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from my_llm import chat_llm


@tool
def getweather(location: str) -> str:
    """Get the current weather in a given location."""
    print(f"调用工具：{location}")
    return f"今天天气 {location} 是晴天."

# 注意：BaseChatModel 没有 add_tools 方法，绑定工具用 bind_tools。
# bind_tools 会从 @tool 装饰的函数上读取签名和 docstring，自动生成工具 schema 传给模型。
tools_chat_llm = chat_llm.bind_tools([getweather])

human_message = HumanMessage(content="你好，我想知道北京的天气")
mesList = [human_message]

res1 = tools_chat_llm.invoke(mesList)
print(res1)
mesList.append(res1)

if res1.tool_calls:
    for tool_call in res1.tool_calls:
        if tool_call["name"] == "getweather":
            res2 = getweather.invoke(tool_call)
            print(res2)
            mesList.append(res2)

res3 = tools_chat_llm.invoke(mesList)
print(res3)
