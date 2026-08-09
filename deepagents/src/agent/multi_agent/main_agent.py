"""从 YAML 动态加载子智能体，再与主智能体组装的示例入口。

本文件示范「主智能体在代码里写死，子智能体从 YAML 动态来」的组合方式：
    - 主智能体：glm_llm + websearch + LocalShellBackend（代码里固定）
    - 子智能体：从 config/subagents.yml 动态解析
"""

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

from agent.my_llm import glm_llm
from agent.tools.web import websearch
from agent.multi_agent.yaml_loader import (
    build_subagents_from_yaml,
    register_model,
    register_tool,
)


# 如果项目里还有其它模型，可一并注册：
# from agent.my_llm import claude_llm
# register_model("claude", claude_llm)

register_tool("websearch", websearch)


# ---------------------------------------------------------------------------
# 主智能体的后端：决定 ls/read_file/write_file/execute 等工具的行为范围
# ---------------------------------------------------------------------------
backend = LocalShellBackend(
    root_dir="../data",
    # 启用安全边界，禁止使用 ../ ~/ 等特殊路径
    virtual_mode=True,
    # 命令最大输出字节数
    max_output_bytes=3,
    # 命令超时时间（秒）
    timeout=120,
    # 环境变量
    env={},
)


# ---------------------------------------------------------------------------
# 组装：主智能体（代码固定）+ 子智能体（YAML 动态）
# ---------------------------------------------------------------------------
# 从 YAML 读出子智能体规格列表
subagents = build_subagents_from_yaml("config/subagents.yml")

# 主智能体的 model / tools / backend 在代码里决定，子智能体作为参数传入
my_agent = create_deep_agent(
    model=glm_llm,
    backend=backend,
    subagents=subagents,
    system_prompt=(
        "你是一个主控智能体，负责理解用户需求并把任务委派给合适的子智能体。"
        "遇到需要联网检索、资料调研的任务，委派给 researcher；"
        "遇到需要撰写/总结的任务，委派给 writer。"
    ),
)


if __name__ == "__main__":
    # 本地快速自测：发起一次调用，确认 agent 能跑通
    question = "帮我检索一下「LangGraph 多智能体」的最新进展，并写成一段中文摘要。"
    print(f"用户提问: {question}\n")
    result = my_agent.invoke({"messages": [{"role": "user", "content": question}]})
    # 打印最后一条 AI 消息
    last_msg = result["messages"][-1]
    print("\n=== 最终回复 ===")
    print(getattr(last_msg, "content", last_msg))
