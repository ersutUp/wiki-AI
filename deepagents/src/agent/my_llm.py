"""

"""
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from agent.env_utils import OPENAI_API_KEY, OPENAI_BASE_URL, FUNCLOUD_BASE_URL, FUNCLOUD_API_KEY
from agent.multi_agent.yaml_loader import register_model

glm_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model="glm-5.1",
)

# glm_llm = init_chat_model(
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
#     model="glm-5.1",
#     model_provider="openai",
# )
#
#
#
# chat_llm = init_chat_model(
#     model="glm-5.1",
#     model_provider="openai",
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
# )
#
# glm47Llm = init_chat_model(
#     model="glm-4.7",
#     model_provider="openai",
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
# )
#
#
#
claude_sonnet5_llm = init_chat_model(
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



deepseek_flash4_llm = init_chat_model(
    model="deepseek-v4-flash",
    api_key=f"{FUNCLOUD_API_KEY}",
    base_url=f"{FUNCLOUD_BASE_URL}/v1",
)




gpt55_llm = init_chat_model(
    api_key=FUNCLOUD_API_KEY,
    base_url=f"{FUNCLOUD_BASE_URL}/v1",
    model="gpt-5.5",
)

# ---------------------------------------------------------------------------
# 注册别名：让 YAML 可以用简短的名字引用
# ---------------------------------------------------------------------------
# 注册之后，YAML 里写 `model: glm` / `tools: [websearch]` 就能被解析到这些对象。
register_model("glm51", glm_llm)
register_model("claude_snet5", claude_sonnet5_llm)
register_model("deepseek_flash4", deepseek_flash4_llm)
register_model("gpt55", gpt55_llm)
