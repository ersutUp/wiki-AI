from langchain.agents import create_agent
from langchain.agents.middleware import ModelResponse, ModelRequest, wrap_model_call
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from my_llm import glm47Llm, glm_llm


@tool
def get_current_location() -> str:
    """获取当前位置。"""
    return "当前位置为北京市。"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}的天气为晴朗，25°C。"

#中间件
@wrap_model_call
def dynamic_model(request: ModelRequest , handler) -> ModelResponse:
    print(request)

    msgs = request.state["messages"]
    #模型切换
    if len(msgs) > 3:
        model = glm_llm
    else:
        model = glm47Llm

    return handler(request.override(model=model))


agent = create_agent(
    model=glm_llm,
    tools=[get_current_location, get_weather],
    middleware=[dynamic_model],
)

m = HumanMessage("今天天气怎么样")
msg = [m]

res = agent.invoke({"messages": msg})
print(res)
print(res["messages"][-1].content)
