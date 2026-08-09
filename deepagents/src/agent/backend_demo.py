from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend

from agent.my_llm import glm_llm
from agent.tools.web import websearch

# my_agent = create_deep_agent(
#     model=glm_llm,
#     tools=[websearch],
# )

# 本地文件系统
# my_agent = create_deep_agent(
#     model=glm_llm,
#     tools=[websearch],
#     backend=FilesystemBackend(
#         root_dir="../data",
#         # 启用安全边界，禁止使用 ../ ~/ 等特殊路径
#         virtual_mode=True,
#     )
# )

my_agent=create_deep_agent(
    model=glm_llm,
    tools=[websearch],
    backend= LocalShellBackend(
        root_dir="../data",
        # 启用安全边界，禁止使用 ../ ~/ 等特殊路径
        virtual_mode=True,
        # 命令最大输出字节数
        max_output_bytes=3,
        # 命令超时时间
        timeout=120,
        # 环境变量
        env={}
    ),
)