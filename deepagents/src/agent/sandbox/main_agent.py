"""从 YAML 动态加载子智能体，再与主智能体组装的示例入口。

本文件示范「主智能体在代码里写死，子智能体从 YAML 动态来」的组合方式：
    - 主智能体：glm_llm + websearch + LocalShellBackend（代码里固定）
    - 子智能体：从 config/subagents.yml 动态解析
"""
from os.path import dirname

from deepagents import create_deep_agent
from opensandbox import SandboxSync
from opensandbox.config import ConnectionConfig, ConnectionConfigSync

import os

from agent.env_utils import WXPUBLIC_SECURE_KEY, WXPUBLIC_APP_ID
from agent.my_llm import glm_llm
from agent.sandbox.open_sendbox import OpenSandboxBackend
from agent.multi_agent.yaml_loader import (
    build_subagents_from_yaml, register_tool,
)
from agent.tools.web import websearch

register_tool("websearch", websearch)


#获取当前文件目录
root_dir = dirname(dirname(dirname(dirname(os.path.abspath(__file__)))))


# 沙盒
open_sandbox = SandboxSync.create(
    "opensandbox/code-interpreter:v1.1.0",
    entrypoint=["/opt/code-interpreter/code-interpreter.sh"],
    connection_config=ConnectionConfigSync(
        # 未指定 domain 时，使用 OPEN_SANDBOX_DOMAIN 环境变量配置的 domain
        domain="10.52.25.34:48999",
        # 这里通过 OPEN_SANDBOX_API_KEY 环境变量配置的 API Key
        # api_key="123123",
    ),
    env={
        "WXPUBLIC_APP_ID": WXPUBLIC_APP_ID,
        "WXPUBLIC_SECURE_KEY": WXPUBLIC_SECURE_KEY,
    }
)

with open_sandbox:
    print(open_sandbox.get_info())


    backend = OpenSandboxBackend(
        open_sandbox
    )

    # ---------------------------------------------------------------------------
    # 把本地 skills 目录同步上传到沙盒（deepagents 的 skills 是在 backend，
    # 即沙盒侧加载的，宿主机本地路径沙盒里不存在，所以要先传进去）
    # ---------------------------------------------------------------------------
    sandbox_skills_dir = "/skills"  # 沙盒内任意可写路径
    local_skills_dir = os.path.join(root_dir, "skills")
    files_to_upload = []
    for root, _dirs, files in os.walk(local_skills_dir):
        for f in files:
            local_path = os.path.join(root, f)
            rel = os.path.relpath(local_path, local_skills_dir)
            remote_path = os.path.join(sandbox_skills_dir, rel)
            with open(local_path, "rb") as fp:
                files_to_upload.append((remote_path, fp.read()))
    if files_to_upload:
        upload_results = backend.upload_files(files_to_upload)
        failed = [r.path for r in upload_results if r.error]
        if failed:
            print(f"[警告] {len(failed)} 个 skill 文件上传失败: {failed}")
        else:
            print(f"[OK] 已上传 {len(files_to_upload)} 个 skill 文件到沙盒 {sandbox_skills_dir}")

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
            "微信公众号抓取内容使用 wxpublic-fetch skill"
        ),
        skills=[
            sandbox_skills_dir
        ]
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
