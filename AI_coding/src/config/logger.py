"""
日志拦截器 —— 将应用日志与 Agent 日志分流到不同目录，按天滚动，支持单文件大小上限与保留时长。

══════════════════════════════════════════════════════════════════════════════════
使用方式
══════════════════════════════════════════════════════════════════════════════════

# 1. 在应用入口（如 main.py）中一次性初始化
from src.config.logger import configure_logging
configure_logging()

# 2. 在业务代码中获取 logger
from src.config.logger import get_app_logger, get_agent_logger

app_logger = get_app_logger("api")           # → logger 名: app.api
agent_logger = get_agent_logger("deepagents")  # → logger 名: agent.deepagents

app_logger.info("请求处理完成")
agent_logger.info("Agent 推理结束")

══════════════════════════════════════════════════════════════════════════════════
环境变量配置
══════════════════════════════════════════════════════════════════════════════════

LOG_DIR            日志根目录，默认 logs
APP_LOG_DIR        应用日志子目录，默认 logs/app
AGENT_LOG_DIR      Agent 日志子目录，默认 logs/agent
LOG_LEVEL          日志级别，默认 INFO
LOG_MAX_BYTES      单文件最大字节数，默认 10485760 (10MB)
LOG_RETENTION_DAYS 日志保留天数，默认 30
LOG_ENCODING       文件编码，默认 utf-8
LOG_FORMAT         日志格式，默认 "%(asctime)s | %(levelname)s | %(name)s | %(process)d | %(message)s"
LOG_DATE_FORMAT    日期格式，默认 "%Y-%m-%d %H:%M:%S"

══════════════════════════════════════════════════════════════════════════════════
滚动策略
══════════════════════════════════════════════════════════════════════════════════

- 日期变化时自动切换到新文件（如 app-2026-08-12.log）
- 同一天内超过 LOG_MAX_BYTES 时生成分片（如 app-2026-08-12.1.log）
- 超出 LOG_RETENTION_DAYS 的日志文件在每次写入后自动清理
- uvicorn.error / uvicorn.access 自动归入应用日志目录
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class LoggerSettings:
    """日志配置，支持环境变量覆盖。使用 frozen=True 防止运行时意外修改。"""

    log_dir: Path = Path("logs")
    app_log_dir: Path | None = None
    agent_log_dir: Path | None = None
    level: str = "INFO"
    max_bytes: int = 10 * 1024 * 1024
    retention_days: int = 30
    encoding: str = "utf-8"
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(process)d | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    def __post_init__(self) -> None:
        # 校验日志级别是否合法
        level = self.level.upper()
        if not isinstance(logging._nameToLevel.get(level), int):
            raise ValueError(f"无效的日志级别: {self.level}")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes 必须大于 0")
        if self.retention_days <= 0:
            raise ValueError("retention_days 必须大于 0")
        # frozen dataclass 需要通过 object.__setattr__ 绕过冻结限制
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "log_dir", Path(self.log_dir))
        # 未指定子目录时使用 log_dir 下的默认子目录
        object.__setattr__(self, "app_log_dir", Path(self.app_log_dir or self.log_dir / "app"))
        object.__setattr__(self, "agent_log_dir", Path(self.agent_log_dir or self.log_dir / "agent"))

    @classmethod
    def from_env(cls) -> "LoggerSettings":
        """从环境变量读取配置，未设置时使用默认值。"""
        app_log_dir = os.getenv("APP_LOG_DIR")
        agent_log_dir = os.getenv("AGENT_LOG_DIR")
        return cls(
            log_dir=Path(os.getenv("LOG_DIR", "logs")),
            app_log_dir=Path(app_log_dir) if app_log_dir else None,
            agent_log_dir=Path(agent_log_dir) if agent_log_dir else None,
            level=os.getenv("LOG_LEVEL", "INFO"),
            max_bytes=int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024))),
            retention_days=int(os.getenv("LOG_RETENTION_DAYS", "30")),
            encoding=os.getenv("LOG_ENCODING", "utf-8"),
            log_format=os.getenv(
                "LOG_FORMAT",
                "%(asctime)s | %(levelname)s | %(name)s | %(process)d | %(message)s",
            ),
            date_format=os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"),
        )


class DailySizeRotatingHandler(logging.Handler):
    """
    组合滚动文件 Handler：同时检查日期边界和文件大小。

    滚动规则：
    - 日期变化 → 切换到新文件，序号归零
    - 同一天超过 max_bytes → 序号递增，生成分片（如 .1.log, .2.log）
    - 每次写入后清理超过 retention_days 的旧文件
    """

    # 匹配文件名格式：prefix-YYYY-MM-DD.log 或 prefix-YYYY-MM-DD.N.log
    _file_pattern = re.compile(r"^(?P<prefix>.+)-(?P<day>\d{4}-\d{2}-\d{2})(?:\.(?P<index>\d+))?\.log$")

    def __init__(self, directory: Path, prefix: str, settings: LoggerSettings) -> None:
        super().__init__()
        self.directory = directory
        self.prefix = prefix
        self.max_bytes = settings.max_bytes
        self.retention_days = settings.retention_days
        self.encoding = settings.encoding
        self._stream: TextIO | None = None
        self._day: date | None = None
        self._index = 0
        # 确保日志目录存在
        self.directory.mkdir(parents=True, exist_ok=True)
        self._open_for(datetime.now().date())

    def _path(self, day: date, index: int) -> Path:
        """构建日志文件路径，index=0 时不带序号后缀。"""
        suffix = f".{index}" if index else ""
        return self.directory / f"{self.prefix}-{day.isoformat()}{suffix}.log"

    def _open_for(self, day: date, index: int = 0) -> None:
        """关闭当前文件流，打开指定日期/序号的新文件。"""
        if self._stream:
            self._stream.close()
        self._day = day
        self._index = index
        self._stream = self._path(day, index).open("a", encoding=self.encoding)

    def _needs_rollover(self, message: str, day: date) -> bool:
        """判断是否需要滚动：日期变化或当前文件写入后会超出大小上限。"""
        if day != self._day:
            return True
        return self._stream is not None and self._stream.tell() + len(message.encode(self.encoding)) > self.max_bytes

    def emit(self, record: logging.LogRecord) -> None:
        """核心写入逻辑：格式化 → 判断滚动 → 写入 → 刷新 → 清理过期文件。"""
        try:
            message = self.format(record) + "\n"
            today = datetime.now().date()
            if self._needs_rollover(message, today):
                # 日期变了 → 序号归零；同一天超大小 → 序号递增
                self._open_for(today, 0 if today != self._day else self._index + 1)
            assert self._stream is not None
            self._stream.write(message)
            self._stream.flush()
            self._cleanup(today)
        except Exception:
            self.handleError(record)

    def _cleanup(self, today: date) -> None:
        """删除早于 retention_days 的日志文件。"""
        cutoff = today - timedelta(days=self.retention_days - 1)
        for path in self.directory.glob(f"{self.prefix}-*.log"):
            match = self._file_pattern.match(path.name)
            if not match:
                continue
            try:
                file_day = date.fromisoformat(match.group("day"))
            except ValueError:
                continue
            if file_day < cutoff:
                path.unlink(missing_ok=True)

    def close(self) -> None:
        """关闭文件流并释放资源。"""
        if self._stream:
            self._stream.close()
            self._stream = None
        super().close()


class ChannelFilter(logging.Filter):
    """
    日志分流过滤器：根据 logger 名称前缀决定是否写入对应 channel 的文件。

    app channel 额外放行 uvicorn 系列 logger，确保 uvicorn 日志归入应用目录。
    """

    def __init__(self, channel: str) -> None:
        super().__init__()
        self.channel = channel

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn.error / uvicorn.access 归入应用日志
        if self.channel == "app" and record.name.startswith("uvicorn"):
            return True
        # 匹配 channel 本身或以 channel. 开头的 logger
        return record.name == self.channel or record.name.startswith(f"{self.channel}.")


def _build_handler(directory: Path, prefix: str, settings: LoggerSettings, channel: str) -> logging.Handler:
    """构建一个带格式、级别和分流过滤器的文件 Handler。"""
    handler = DailySizeRotatingHandler(directory, prefix, settings)
    handler.setLevel(settings.level)
    handler.setFormatter(logging.Formatter(settings.log_format, settings.date_format))
    handler.addFilter(ChannelFilter(channel))
    return handler


def configure_logging(settings: LoggerSettings | None = None) -> LoggerSettings:
    """
    幂等初始化日志系统：创建控制台 Handler + 应用文件 Handler + Agent 文件 Handler。

    通过 _wiki_ai_handler 标记识别本模块创建的 Handler，重复调用时先移除旧的再重建，
    兼容 uvicorn reload 等开发场景。
    """
    settings = settings or LoggerSettings.from_env()
    root = logging.getLogger()
    root.setLevel(settings.level)

    # 移除本模块之前创建的 handler，避免 reload 时重复
    for handler in list(root.handlers):
        if getattr(handler, "_wiki_ai_handler", False):
            root.removeHandler(handler)
            handler.close()

    # 控制台 handler —— 所有日志统一输出到终端
    formatter = logging.Formatter(settings.log_format, settings.date_format)
    console = logging.StreamHandler()
    console.setLevel(settings.level)
    console.setFormatter(formatter)
    console._wiki_ai_handler = True
    root.addHandler(console)

    # 应用日志文件 handler —— 只写 app.* 和 uvicorn 日志
    app_handler = _build_handler(settings.app_log_dir, "app", settings, "app")
    agent_handler = _build_handler(settings.agent_log_dir, "agent", settings, "agent")
    app_handler._wiki_ai_handler = True
    agent_handler._wiki_ai_handler = True
    root.addHandler(app_handler)
    root.addHandler(agent_handler)

    # 确保 uvicorn 系列 logger 存在且向上传播到 root handler
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.setLevel(settings.level)
        logger.propagate = True

    return settings


def get_app_logger(name: str = "app") -> logging.Logger:
    """获取应用 logger，自动补全 app. 命名空间前缀。"""
    return logging.getLogger(name if name == "app" or name.startswith("app.") else f"app.{name}")


def get_agent_logger(name: str = "agent") -> logging.Logger:
    """获取 Agent logger，自动补全 agent. 命名空间前缀。"""
    return logging.getLogger(name if name == "agent" or name.startswith("agent.") else f"agent.{name}")
    log_dir: Path = Path("logs")
    app_log_dir: Path | None = None
    agent_log_dir: Path | None = None
    level: str = "INFO"
    max_bytes: int = 10 * 1024 * 1024
    retention_days: int = 30
    encoding: str = "utf-8"
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(process)d | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    def __post_init__(self) -> None:
        level = self.level.upper()
        if not isinstance(logging._nameToLevel.get(level), int):
            raise ValueError(f"无效的日志级别: {self.level}")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes 必须大于 0")
        if self.retention_days <= 0:
            raise ValueError("retention_days 必须大于 0")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "log_dir", Path(self.log_dir))
        object.__setattr__(self, "app_log_dir", Path(self.app_log_dir or self.log_dir / "app"))
        object.__setattr__(self, "agent_log_dir", Path(self.agent_log_dir or self.log_dir / "agent"))

    @classmethod
    def from_env(cls) -> "LoggerSettings":
        app_log_dir = os.getenv("APP_LOG_DIR")
        agent_log_dir = os.getenv("AGENT_LOG_DIR")
        return cls(
            log_dir=Path(os.getenv("LOG_DIR", "logs")),
            app_log_dir=Path(app_log_dir) if app_log_dir else None,
            agent_log_dir=Path(agent_log_dir) if agent_log_dir else None,
            level=os.getenv("LOG_LEVEL", "INFO"),
            max_bytes=int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024))),
            retention_days=int(os.getenv("LOG_RETENTION_DAYS", "30")),
            encoding=os.getenv("LOG_ENCODING", "utf-8"),
            log_format=os.getenv(
                "LOG_FORMAT",
                "%(asctime)s | %(levelname)s | %(name)s | %(process)d | %(message)s",
            ),
            date_format=os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"),
        )


class DailySizeRotatingHandler(logging.Handler):
    _file_pattern = re.compile(r"^(?P<prefix>.+)-(?P<day>\d{4}-\d{2}-\d{2})(?:\.(?P<index>\d+))?\.log$")

    def __init__(self, directory: Path, prefix: str, settings: LoggerSettings) -> None:
        super().__init__()
        self.directory = directory
        self.prefix = prefix
        self.max_bytes = settings.max_bytes
        self.retention_days = settings.retention_days
        self.encoding = settings.encoding
        self._stream: TextIO | None = None
        self._day: date | None = None
        self._index = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        self._open_for(datetime.now().date())

    def _path(self, day: date, index: int) -> Path:
        suffix = f".{index}" if index else ""
        return self.directory / f"{self.prefix}-{day.isoformat()}{suffix}.log"

    def _open_for(self, day: date, index: int = 0) -> None:
        if self._stream:
            self._stream.close()
        self._day = day
        self._index = index
        self._stream = self._path(day, index).open("a", encoding=self.encoding)

    def _needs_rollover(self, message: str, day: date) -> bool:
        if day != self._day:
            return True
        return self._stream is not None and self._stream.tell() + len(message.encode(self.encoding)) > self.max_bytes

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record) + "\n"
            today = datetime.now().date()
            if self._needs_rollover(message, today):
                self._open_for(today, 0 if today != self._day else self._index + 1)
            assert self._stream is not None
            self._stream.write(message)
            self._stream.flush()
            self._cleanup(today)
        except Exception:
            self.handleError(record)

    def _cleanup(self, today: date) -> None:
        cutoff = today - timedelta(days=self.retention_days - 1)
        for path in self.directory.glob(f"{self.prefix}-*.log"):
            match = self._file_pattern.match(path.name)
            if not match:
                continue
            try:
                file_day = date.fromisoformat(match.group("day"))
            except ValueError:
                continue
            if file_day < cutoff:
                path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
        super().close()


class ChannelFilter(logging.Filter):
    def __init__(self, channel: str) -> None:
        super().__init__()
        self.channel = channel

    def filter(self, record: logging.LogRecord) -> bool:
        if self.channel == "app" and record.name.startswith("uvicorn"):
            return True
        return record.name == self.channel or record.name.startswith(f"{self.channel}.")


def _build_handler(directory: Path, prefix: str, settings: LoggerSettings, channel: str) -> logging.Handler:
    handler = DailySizeRotatingHandler(directory, prefix, settings)
    handler.setLevel(settings.level)
    handler.setFormatter(logging.Formatter(settings.log_format, settings.date_format))
    handler.addFilter(ChannelFilter(channel))
    return handler


def configure_logging(settings: LoggerSettings | None = None) -> LoggerSettings:
    settings = settings or LoggerSettings.from_env()
    root = logging.getLogger()
    root.setLevel(settings.level)

    for handler in list(root.handlers):
        if getattr(handler, "_wiki_ai_handler", False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(settings.log_format, settings.date_format)
    console = logging.StreamHandler()
    console.setLevel(settings.level)
    console.setFormatter(formatter)
    console._wiki_ai_handler = True
    root.addHandler(console)

    app_handler = _build_handler(settings.app_log_dir, "app", settings, "app")
    agent_handler = _build_handler(settings.agent_log_dir, "agent", settings, "agent")
    app_handler._wiki_ai_handler = True
    agent_handler._wiki_ai_handler = True
    root.addHandler(app_handler)
    root.addHandler(agent_handler)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.setLevel(settings.level)
        logger.propagate = True

    return settings


def get_app_logger(name: str = "app") -> logging.Logger:
    return logging.getLogger(name if name == "app" or name.startswith("app.") else f"app.{name}")


def get_agent_logger(name: str = "agent") -> logging.Logger:
    return logging.getLogger(name if name == "agent" or name.startswith("agent.") else f"agent.{name}")
