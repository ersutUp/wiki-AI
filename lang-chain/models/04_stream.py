from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

agent = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

res = agent.stream("你几岁了？")

for chunk in res:
    # print(type(chunk))
    print(chunk.content, end="", flush=True)


