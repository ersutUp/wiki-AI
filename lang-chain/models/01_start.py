from langchain.agents import create_agent

from my_llm import glm_llm

agent = create_agent(
    model=glm_llm,
    system_prompt="你是天气查询助手",
    tools=[],
)

res = agent.invoke({"messages": [{"role": "user", "content": "北京天气"}]})
print(type(res))
print(res)
print(res["messages"][-1].pretty_print())
