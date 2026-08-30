"""

"""
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from agent.env_utils import GLM_API_KEY, GLM_BASE_URL, FUNCLOUD_BASE_URL, FUNCLOUD_API_KEY

glm_llm = ChatOpenAI(
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL,
    model="glm-5.1",
)

claude_sonnet5_llm = ChatAnthropic(
    # model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    # model="global.anthropic.claude-opus-4-5-20251101-v1:0",
    # model="global.anthropic.claude-opus-4-6-v1",
    # model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    # model="global.anthropic.claude-sonnet-4-6",
    model_name="global.anthropic.claude-sonnet-5",
    api_key=FUNCLOUD_API_KEY,
    base_url=FUNCLOUD_BASE_URL,
)



deepseek_flash4_llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=f"{FUNCLOUD_API_KEY}",
    base_url=f"{FUNCLOUD_BASE_URL}/v1",
)




gpt55_llm = ChatOpenAI(
    api_key=FUNCLOUD_API_KEY,
    base_url=f"{FUNCLOUD_BASE_URL}/v1",
    model="gpt-5.5",
)
