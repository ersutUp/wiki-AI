"""从 YAML 文件动态配置 DeepAgents 子智能体。

本模块只做一件事：把一份声明式的 YAML 解析成 deepagents 能识别的
**子智能体（subagents）规格列表**。主智能体的 ``model`` / ``tools`` /
``backend`` 等仍由调用方在代码里决定，本模块不插手。

子智能体 YAML 中「工具 / 模型」引用支持三种写法：
    - 注册表别名：先调用 ``register_tool`` / ``register_model`` 注册，
      YAML 里直接写别名（如 ``glm``、``websearch``）。
    - 动态导入：写成 ``"模块:属性"`` 形式（如
      ``agent.my_llm:glm_llm``、``agent.tools.web:websearch``），
      无需手动注册即可加载。
    - 模型还支持 ``provider:model`` 字符串（如 ``openai:gpt-4o``），
      原样交给 deepagents 内部解析。

典型用法::

    from deepagents import create_deep_agent
    from agent.multi_agent.yaml_loader import (
        build_subagents_from_yaml, register_model, register_tool,
    )

    register_model("glm", glm_llm)
    register_tool("websearch", websearch)

    subagents = build_subagents_from_yaml("config/subagents.yml")
    agent = create_deep_agent(model=glm_llm, tools=[websearch], subagents=subagents)
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

# ---------------------------------------------------------------------------
# 注册表：别名 -> 对象
# ---------------------------------------------------------------------------
# 全局注册表，供 YAML 按简短别名引用工具与模型。
_TOOL_REGISTRY: dict[str, BaseTool | Callable[..., Any]] = {}
_MODEL_REGISTRY: dict[str, str | BaseChatModel] = {}


def register_tool(name: str, tool: BaseTool | Callable[..., Any]) -> BaseTool | Callable[..., Any]:
    """注册一个工具到全局注册表，使其可在 YAML 中用 ``name`` 引用。

    Args:
        name: YAML 中使用的别名。
        tool: 工具对象（``@tool`` 装饰的函数或 ``BaseTool`` 实例）。

    Returns:
        传入的 tool（方便链式/装饰器式使用）。
    """
    _TOOL_REGISTRY[name] = tool
    return tool


def register_model(name: str, model: str | BaseChatModel) -> None:
    """注册一个模型（或 ``provider:model`` 字符串）到全局注册表。

    Args:
        name: YAML 中使用的别名。
        model: 已实例化的 ``BaseChatModel``，或 ``"provider:model"`` 字符串。
    """
    _MODEL_REGISTRY[name] = model


# ---------------------------------------------------------------------------
# 引用解析
# ---------------------------------------------------------------------------
def _try_import(ref: str) -> Any | None:
    """尝试按 ``"模块:属性"`` 形式动态导入；失败或格式不符则返回 ``None``。

    Args:
        ref: 形如 ``agent.tools.web:websearch`` 的引用字符串。

    Returns:
        导入到的对象，或 ``None``。
    """
    if ":" not in ref:
        return None
    module_path, _, attr = ref.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    return getattr(module, attr, None)


def resolve_tool(ref: str) -> BaseTool | Callable[..., Any]:
    """把 YAML 中的工具引用解析成真实工具对象。

    解析顺序：注册表别名 -> ``模块:属性`` 动态导入。

    Args:
        ref: 工具引用（别名或导入路径）。

    Returns:
        工具对象。

    Raises:
        ValueError: 引用既不在注册表中，也无法动态导入。
    """
    if ref in _TOOL_REGISTRY:
        return _TOOL_REGISTRY[ref]
    obj = _try_import(ref)
    if obj is not None:
        return obj
    msg = (
        f"无法解析工具引用 {ref!r}：既未在注册表中注册，"
        f"也无法按 '模块:属性' 导入。"
    )
    raise ValueError(msg)


def _looks_like_import(ref: str) -> bool:
    """判断一个含冒号的引用是否像 ``模块:属性`` 的动态导入路径。

    启发式：冒号前的部分若包含点号，视为 Python 包路径
    （如 ``agent.my_llm:glm_llm``）；否则视为 ``provider:model`` 字符串
    （如 ``openai:gpt-4o``，provider 名中不含点）。

    Args:
        ref: 待判断的引用字符串。

    Returns:
        ``True`` 表示应当走动态导入。
    """
    return "." in ref.split(":", 1)[0]


def resolve_model(ref: str | None) -> str | BaseChatModel | None:
    """把 YAML 中的模型引用解析成模型对象或 ``provider:model`` 字符串。

    解析顺序：注册表别名 -> ``模块:属性`` 动态导入 -> 原样返回(视作 provider:model)。

    Args:
        ref: 模型引用（别名 / 导入路径 / ``provider:model`` 字符串），可为 ``None``。

    Returns:
        解析后的模型，或 ``None``（表示「未指定，交给上层兜底」）。
    """
    if ref is None:
        return None
    if ref in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[ref]
    # 仅当看起来像 "模块:属性" 时才尝试动态导入，避免把 "provider:model" 误判
    if _looks_like_import(ref):
        obj = _try_import(ref)
        if obj is not None:
            return obj
    # 当作 "provider:model" 字符串，交给 deepagents 内部 resolve_model 解析
    return ref


# ---------------------------------------------------------------------------
# YAML 解析与组装
# ---------------------------------------------------------------------------
def load_yaml_config(path: str | Path) -> list[dict[str, Any]]:
    """读取并解析 YAML 配置文件，返回子智能体条目列表。

    YAML 顶层支持两种写法：
        1. 直接是一个列表（每个元素是一个子智能体）。
        2. 一个字典，取其中的 ``subagents`` 键（列表）。

    Args:
        path: YAML 文件路径。

    Returns:
        子智能体条目列表（每个是 dict）。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 顶层结构不是列表/字典，或为空。
    """
    p = Path(path)
    if not p.exists():
        msg = f"配置文件不存在: {p}"
        raise FileNotFoundError(msg)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("subagents") or []
    else:
        msg = f"配置文件顶层必须是列表或字典，得到 {type(data).__name__}"
        raise ValueError(msg)

    if not items:
        msg = f"配置文件 {p} 未定义任何子智能体（需要顶层 subagents 列表）"
        raise ValueError(msg)
    return items


def _build_subagent_spec(item: dict[str, Any]) -> dict[str, Any]:
    """把单个子智能体 YAML 条目转成 deepagents 的 ``SubAgent`` dict。

    Args:
        item: YAML 中 ``subagents`` 列表里的一个条目。

    Returns:
        可直接传给 ``create_deep_agent(subagents=[...])`` 的 dict。

    Raises:
        ValueError: 缺少 ``name`` / ``description`` / ``system_prompt`` 必填字段。
    """
    for key in ("name", "description", "system_prompt"):
        if key not in item:
            msg = f"子智能体配置缺少必填字段 {key!r}: {item}"
            raise ValueError(msg)

    spec: dict[str, Any] = {
        "name": item["name"],
        "description": item["description"],
        "system_prompt": item["system_prompt"],
    }
    # model/tools 为可选；不填则由 deepagents 继承主智能体
    if item.get("model") is not None:
        spec["model"] = resolve_model(item["model"])
    if item.get("tools") is not None:
        spec["tools"] = [resolve_tool(t) for t in item["tools"]]
    # 其余 deepagents 支持的可选字段透传
    for opt in ("interrupt_on", "skills", "permissions"):
        if item.get(opt) is not None:
            spec[opt] = item[opt]
    return spec


def build_subagents_from_yaml(path: str | Path) -> list[dict[str, Any]]:
    """从 YAML 文件解析出子智能体规格列表。

    只负责子智能体：读取 YAML、解析其中工具/模型引用，返回可直接传给
    ``create_deep_agent(subagents=...)`` 的列表。主智能体的 model / tools /
    backend 不在本模块职责内。

    Args:
        path: YAML 配置文件路径。

    Returns:
        子智能体规格 dict 列表。
    """
    items = load_yaml_config(path)
    return [_build_subagent_spec(item) for item in items]


if __name__ == "__main__":
    # 仅供演示：先注册别名，再纯解析（不构建 agent、不依赖 deepagents），打印解析结果便于自查。
    import json

    from agent.my_llm import glm_llm
    from agent.tools.web import websearch

    register_model("glm", glm_llm)
    register_tool("websearch", websearch)

    _demo_path = Path(__file__).resolve().parents[3] / "config" / "subagents.yml"
    print(f"加载配置: {_demo_path}")
    subagents = build_subagents_from_yaml(_demo_path)
    print(json.dumps(subagents, ensure_ascii=False, indent=2, default=str))
