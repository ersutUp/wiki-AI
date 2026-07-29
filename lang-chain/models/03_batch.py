from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

agent = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# 顺序执行
res = agent.batch([
    "你几岁了？",
    "你性别是？",
    "你来自哪里？",
])

for item in res:
    print(item)

print("----------------")

# 并发执行
res = agent.batch_as_completed([
    "你几岁了？",
    "你性别是？",
    "你来自哪里？",
])

for item in res:
    print(item)
