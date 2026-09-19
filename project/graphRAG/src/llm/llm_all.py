from langchain_openai import ChatOpenAI

from utils.env_utils import LLM_API_KEY, LLM_BASE_URL

# qwen3-vl 是多模态聊天模型，必须走 chat/completions 接口
qwen3_vl_llm = ChatOpenAI(
    model="qwen3-vl-plus",
    api_key=f"{LLM_API_KEY}",
    base_url=f"{LLM_BASE_URL}",
)
