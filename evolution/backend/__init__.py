"""后端模块：提供多 LLM 提供商的统一查询接口"""

import re
import logging
import os
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

# 延迟导入各后端，避免缺少依赖时报错
_backend_modules = {}
for _name in ['backend_openai', 'backend_anthropic', 'backend_openrouter', 'backend_gemini']:
    try:
        _backend_modules[_name] = __import__(f'evolution.backend.{_name}', fromlist=[_name])
    except ImportError:
        pass

backend_openai = _backend_modules.get('backend_openai')
backend_anthropic = _backend_modules.get('backend_anthropic')
backend_openrouter = _backend_modules.get('backend_openrouter')
backend_gemini = _backend_modules.get('backend_gemini')

logger = logging.getLogger("evolution")


def determine_provider(model: str) -> str:
    """根据模型名称确定使用哪个后端提供商"""
    # 优先：如果显式启用 Trae 后端，直接走它
    if os.getenv("EVOLUTION_BACKEND", "").lower() == "trae":
        return "trae"
    # Check if model matches OpenAI patterns first
    if re.match(r"^(gpt-.*|o\d+(-.*)?|codex-mini-latest)$", model):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("gemini-"):
        return "gemini"
    # If OPENAI_BASE_URL is set, use openai provider for non-standard models
    elif os.getenv("OPENAI_BASE_URL"):
        return "openai"
    # all other models are handled by openrouter
    else:
        return "openrouter"


provider_to_query_func = {}
if backend_openai:
    provider_to_query_func["openai"] = backend_openai.query
if backend_anthropic:
    provider_to_query_func["anthropic"] = backend_anthropic.query
if backend_openrouter:
    provider_to_query_func["openrouter"] = backend_openrouter.query
if backend_gemini:
    provider_to_query_func["gemini"] = backend_gemini.query

# Trae 后端：通过文件中转让 Trae 会话接管 LLM 调用
# 通过环境变量 EVOLUTION_BACKEND=trae 启用
backend_trae = None
if os.getenv("EVOLUTION_BACKEND", "").lower() == "trae":
    try:
        backend_trae = __import__('evolution.backend.backend_trae', fromlist=['backend_trae'])
        provider_to_query_func["trae"] = backend_trae.query
        logger.info("Trae LLM 后端已启用（文件中转模式）")
    except ImportError as e:
        logger.warning(f"无法加载 Trae 后端: {e}")


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    通用 LLM 查询接口，支持多个后端。
    支持函数调用（function calling）。

    Args:
        system_message: 系统提示词（未编译格式，会自动转换为 OpenAI/Anthropic 格式）
        user_message: 用户消息（未编译格式）
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        func_spec: 可选的函数调用规范，若提供则返回 dict

    Returns:
        字符串（无 func_spec）或 dict（有 func_spec）
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    provider = determine_provider(model)
    query_func = provider_to_query_func[provider]
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
