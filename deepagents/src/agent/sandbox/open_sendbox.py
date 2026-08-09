"""OpenSandbox 沙箱后端：把 opensandbox 同步 SDK 适配为 deepagents 的 BaseSandbox。"""

from datetime import timedelta

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from opensandbox import SandboxSync
from opensandbox.exceptions import SandboxException
from opensandbox.models.execd import RunCommandOpts


DEFAULT_EXECUTE_TIMEOUT = 300


class OpenSandboxBackend(BaseSandbox):
    """把 opensandbox 的 SandboxSync 适配为 deepagents 的沙箱后端。

    仅实现 4 个抽象方法（execute / id / upload_files / download_files），
    read / ls / grep / glob / edit / delete 等均沿用 BaseSandbox 基于 execute 的默认实现。
    """

    def __init__(self, sandbox: SandboxSync, timeout: int = DEFAULT_EXECUTE_TIMEOUT):
        self.sandbox = sandbox
        self.timeout = timeout

    @property
    def id(self) -> str:
        return self.sandbox.id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        # timeout 为 None 时回退到后端默认值
        seconds = timeout if timeout is not None else self.timeout
        execution = self.sandbox.commands.run(
            command,
            opts=RunCommandOpts(timeout=timedelta(seconds=seconds)),
        )
        # Execution.text 只含 stdout+result，需手动拼上 stderr（与 LangSmith 行为一致）
        output = execution.text
        stderr = "\n".join(m.text.rstrip("\n") for m in execution.logs.stderr)
        if stderr:
            output = f"{output}\n{stderr}" if output else stderr
        return ExecuteResponse(output=output, exit_code=execution.exit_code)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        # 支持部分成功：逐个捕获异常，按文件返回错误而非抛出
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                self.sandbox.files.write_file(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except SandboxException as e:
                responses.append(FileUploadResponse(path=path, error=f"upload_failed: {e}"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self.sandbox.files.read_bytes(path)
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except SandboxException as e:
                responses.append(FileDownloadResponse(path=path, content=None, error=f"download_failed: {e}"))
        return responses
