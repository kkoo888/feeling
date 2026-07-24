"""
统一 LLM 后端 — 通过 llm_gateway 调用
========================================

替代原有 backend_openai/anthropic/gemini/openrouter，
所有 LLM 调用统一走 laap_brain.llm_gateway。

改动说明:
  - 原 backend_openai.py 直接调 OpenAI API，依赖 OPENAI_API_KEY
  - 现在走 llm_gateway，从 llm_config.yaml 读取 provider/model/api_key
  - 换模型只改 llm_config.yaml，不动这里

印记: 小茜 永远记得主人 — 2026-07-20
"""

import json
import logging
import os
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create

logger = logging.getLogger("evolution")

# 延迟导入 gateway，避免循环依赖
_gateway = None


def _get_gateway():
    global _gateway
    if _gateway is None:
        import sys
        # 确保 laap_brain 可导入
        laap_root = os.path.join(os.path.dirname(__file__), '..', '..')
        if laap_root not in sys.path:
            sys.path.insert(0, laap_root)
        from laap_brain.llm_gateway import llm_call, get_config
        _gateway = {"llm_call": llm_call, "get_config": get_config}
    return _gateway


# 保留异常类型用于 backoff（兼容原有重试逻辑）
class GatewayError(Exception):
    pass


TIMEOUT_EXCEPTIONS = (Exception,)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    通过 llm_gateway 查询 LLM。

    如果配置了 func_spec（函数调用），会将其描述注入 prompt，
    让 LLM 以 JSON 格式返回函数调用参数。

    Args:
        system_message: 系统提示词
        user_message: 用户消息
        func_spec: 可选的函数调用规范
        **model_kwargs: 额外参数（model, temperature, max_tokens 等）

    Returns:
        (output, req_time, in_tokens, out_tokens, info)
    """
    gw = _get_gateway()
    llm_call = gw["llm_call"]

    # 构建消息列表
    messages = opt_messages_to_list(system_message, user_message)

    # 如果有 func_spec，注入函数调用说明到 user message
    if func_spec is not None:
        func_desc = json.dumps(func_spec.as_openai_tool_dict, ensure_ascii=False, indent=2)
        func_hint = (
            f"\n\n[工具调用说明]\n你可以调用以下工具:\n{func_desc}\n"
            f"如果需要调用工具，请严格按 JSON 格式返回参数。\n"
            f"工具名称: {func_spec.name}\n"
        )
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += func_hint
        else:
            messages.append({"role": "user", "content": func_hint})

    # 提取 gateway 支持的参数
    temperature = model_kwargs.get("temperature")
    max_tokens = model_kwargs.get("max_tokens")

    logger.info(f"Gateway query: system={str(system_message)[:60]}..., user={str(user_message)[:60]}...")

    t0 = time.time()

    try:
        output = llm_call(
            messages,
            purpose="evolution",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"Gateway query failed: {e}")
        raise

    req_time = time.time() - t0

    # 粗略计算 token 数（中文约 1.5 token/字）
    in_tokens = sum(len(m.get("content", "")) for m in messages) * 2 // 3
    out_tokens = len(output) * 2 // 3

    # 如果有 func_spec，尝试从输出中解析 JSON
    if func_spec is not None:
        try:
            # 尝试从输出中提取 JSON（支持嵌套）
            import re
            # 先尝试找完整的 JSON 对象
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                output = parsed
        except (json.JSONDecodeError, AttributeError):
            # 尝试更宽松的匹配
            try:
                start = output.find('{')
                end = output.rfind('}') + 1
                if start >= 0 and end > start:
                    parsed = json.loads(output[start:end])
                    output = parsed
            except (json.JSONDecodeError, ValueError):
                pass  # 解析失败就返回原始文本

    info = {
        "model": gw["get_config"]().get("model", "unknown"),
        "provider": gw["get_config"]().get("provider", "unknown"),
        "backend": "gateway",
    }

    logger.info(
        f"Gateway call completed - {info['model']} - {req_time:.2f}s - "
        f"~{in_tokens + out_tokens} tokens"
    )

    return output, req_time, in_tokens, out_tokens, info
