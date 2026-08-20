"""

"""
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL, FUNCLOUD_API_KEY, FUNCLOUD_BASE_URL

glm_llm = init_chat_model(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model="glm-5.1",
    model_provider="openai",
)



chat_llm = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

glm47Llm = init_chat_model(
    model="glm-4.7",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)



funcloude_claude_llm = init_chat_model(
    # model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    # model="global.anthropic.claude-opus-4-5-20251101-v1:0",
    # model="global.anthropic.claude-opus-4-6-v1",
    # model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    # model="global.anthropic.claude-sonnet-4-6",
    model="global.anthropic.claude-sonnet-5",
    model_provider="anthropic",
    api_key=FUNCLOUD_API_KEY,
    base_url=FUNCLOUD_BASE_URL,
)

funcloude_deepseek_llm = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="openai",
    api_key=FUNCLOUD_API_KEY,
    base_url=f"{FUNCLOUD_BASE_URL}/v1",
)

# funcloude_deepseek_llm = ChatOpenAI(
#     api_key=FUNCLOUD_API_KEY,
#     base_url="https://api.funcloud.ai/v1/official/chat/completions",
#     model="deepseek-v4-pro",
# )