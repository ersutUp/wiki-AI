from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from utils.env_utils import LLM_API_KEY, LLM_BASE_URL

# qwen3-vl 是多模态聊天模型，必须走 chat/completions 接口
qwen3_vl_llm = ChatOpenAI(
    model="qwen3-vl-plus",
    api_key=LLM_API_KEY,
    base_url=f"{LLM_BASE_URL}/v1",
)

deepseek_v4_pro = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=LLM_API_KEY,
    base_url=f"{LLM_BASE_URL}/v1",
)

claude_sonnet_4_6 = ChatAnthropic(
    model_name="global.anthropic.claude-sonnet-4-6",
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)
