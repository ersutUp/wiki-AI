"""

"""
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from llm.env_utils import FUNCLOUD_API_KEY, FUNCLOUD_BASE_URL

qwen3_vl_llm = init_chat_model(
    model="qwen3-vl-plus",
    model_provider="openai",
    api_key=FUNCLOUD_API_KEY,
    base_url=f"{FUNCLOUD_BASE_URL}/v1",
)