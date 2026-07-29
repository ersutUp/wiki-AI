from typing import Any
from uuid import UUID

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from env_utils import OPENAI_BASE_URL, OPENAI_API_KEY

class MyCallback(BaseCallbackHandler):
    # 注意：ChatModel（对话模型）触发的是 on_chat_model_start，而不是 on_llm_start。
    # on_llm_start 只有普通补全模型（LLM）才会触发。
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        print("call")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        print("end")


llm = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

res = llm.invoke("你好，你是谁？", config= {
    "callbacks": [MyCallback()],
})

print(res)


