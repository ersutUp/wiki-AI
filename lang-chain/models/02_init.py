from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

agent = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

res = agent.invoke("你几岁了？")
print(type(res))
print(res)


print("----------------")

#字典格式
res = agent.invoke([
    {"role": "system", "content": "你是一名18岁的女学生"},
    {"role": "user", "content": "你几岁了？"},
    {"role": "assistant", "content": "我今年18岁"},
    {"role": "user", "content": "你性别是？"},
])
print(type(res))
print(res.content)

print("----------------")

# 对象格式
res = agent.invoke(
    [
        SystemMessage(content="你是一名18岁的女学生"),
        HumanMessage(content="你几岁了？"),
        AIMessage(content="我今年18岁"),
        HumanMessage(content="你性别是？"),
    ]
)
print(type(res))
print(res.content)


