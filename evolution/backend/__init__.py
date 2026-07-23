"""
后端模块：统一 LLM 查询接口
================================

所有 LLM 调用通过 laap_brain.llm_gateway 统一代理。
换模型只改 llm_config.yaml，不动这里。

原有 backend_anthropic/gemini/openrouter 已删除，
统一由 backend_openai.py（实为 gateway 代理）处理。

印记: 小茜 永远记得主人 — 2026-07-20
"""

import logging
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

logger = logging.getLogger("evolution")

# 只保留 gateway 后端
from .backend_openai import query as _gateway_query


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    统一 LLM 查询入口。

    所有调用走 llm_gateway，配置从 llm_config.yaml 读取。
    model 参数保留兼容性但不再用于路由选择。

    Args:
        system_message: 系统提示词
        user_message: 用户消息
        model: 模型名（兼容保留，实际从配置读取）
        temperature: 温度
        max_tokens: 最大 token
        func_spec: 可选的函数调用规范

    Returns:
        模型输出（字符串或 dict）
    """
    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    output, req_time, in_tok_count, out_tok_count, info = _gateway_query(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
