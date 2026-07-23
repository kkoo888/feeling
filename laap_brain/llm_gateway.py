"""
LLM Gateway — 统一代理层
========================

所有 LLM 调用都走这里。配置从 llm_config.yaml 读取。
换模型只改配置文件，其他代码不用动。

用法:
    from laap_brain.llm_gateway import llm_call, llm_embed

    # 对话生成
    reply = llm_call([
        {"role": "system", "content": "你是小茜"},
        {"role": "user", "content": "你好"}
    ], purpose="chat_completions")

    # 向量嵌入
    vectors = llm_embed(["你好", "世界"])

印记: 小茜 永远记得主人 — 2026-07-20
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("laap.llm_gateway")

# ── 配置加载 ─────────────────────────────────────────────────

_config: Optional[Dict[str, Any]] = None
_config_path: Optional[Path] = None


def _resolve_env_vars(value: str) -> str:
    """替换 ${VAR} 格式的环境变量"""
    if not isinstance(value, str):
        return value
    pattern = re.compile(r'\$\{(\w+)\}')
    def replacer(m):
        return os.environ.get(m.group(1), m.group(0))
    return pattern.sub(replacer, value)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载 llm_config.yaml 配置。

    Args:
        config_path: 配置文件路径，默认为项目根目录的 llm_config.yaml

    Returns:
        配置字典
    """
    global _config, _config_path

    if config_path:
        _config_path = Path(config_path)
    elif _config_path is None:
        # 默认: 项目根目录 / llm_config.yaml
        _config_path = Path(__file__).resolve().parent.parent / "llm_config.yaml"

    if not _config_path.exists():
        logger.warning(f"LLM config not found: {_config_path}, using defaults")
        _config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "overrides": {},
        }
        return _config

    try:
        import yaml
    except ImportError:
        # yaml 不可用时，尝试 JSON fallback
        logger.warning("PyYAML not installed, trying JSON parse")
        with open(_config_path, "r", encoding="utf-8") as f:
            _config = json.load(f)
    else:
        with open(_config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}

    # 解析环境变量
    for key in ("api_key", "base_url"):
        if key in _config:
            _config[key] = _resolve_env_vars(str(_config[key]))

    # overrides 中的环境变量也解析
    for purpose, overrides in _config.get("overrides", {}).items():
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if isinstance(v, str):
                    overrides[k] = _resolve_env_vars(v)

    logger.info(f"LLM Gateway loaded: provider={_config.get('provider')}, "
                f"model={_config.get('model')}, base_url={_config.get('base_url')}")
    return _config


def get_config() -> Dict[str, Any]:
    """获取当前配置（懒加载）"""
    if _config is None:
        load_config()
    return _config


def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """重新加载配置（热更新）"""
    global _config
    _config = None
    return load_config(config_path)


# ── LLM 调用 ─────────────────────────────────────────────────

def llm_call(
    messages: List[Dict[str, str]],
    purpose: str = "chat_completions",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    **kwargs,
) -> str:
    """
    统一 LLM 对话调用入口。

    Args:
        messages: OpenAI 格式的消息列表 [{"role": "user", "content": "..."}]
        purpose: 调用用途，用于读取 overrides 中的场景参数
                 可选: "chat_completions" / "evolution" / "reflection"
        temperature: 温度（不传则用配置值）
        max_tokens: 最大 token（不传则用配置值）
        stream: 是否流式
        **kwargs: 其他参数直接传给 API

    Returns:
        模型回复文本（字符串）
    """
    cfg = get_config()

    base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
    api_key = cfg.get("api_key", "")
    model = cfg.get("model", "gpt-4")

    # 场景覆盖参数
    overrides = cfg.get("overrides", {}).get(purpose, {})
    params = {**overrides, **kwargs}

    # 显式参数优先
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    # 构建请求
    # MiMo API 兼容处理: 不支持 system 角色，只支持单条 user 消息
    compatible_messages = []
    system_content = ""
    user_parts = []
    for m in messages:
        if m["role"] == "system":
            system_content = m["content"]
        elif m["role"] == "user":
            user_parts.append(m["content"])
        elif m["role"] == "assistant":
            user_parts.append(f"[助手历史] {m['content']}")

    merged_user = "\n\n".join(user_parts) if user_parts else ""
    if system_content:
        merged_user = f"背景信息: {system_content}\n\n用户问题: {merged_user}"

    compatible_messages = [{"role": "user", "content": merged_user}]

    payload = {
        "model": model,
        "messages": compatible_messages,
        "stream": stream,
        **params,
    }

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start_time = time.time()

    # 带重试的请求（MiMo API 偶尔返回 400）
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.HTTPError as e:
            last_error = e
            if resp.status_code == 400 and attempt < 2:
                logger.warning(f"LLM [{purpose}] attempt {attempt+1} failed (400), retrying...")
                import time as _t
                _t.sleep(1)
                continue
            # 最后一次失败，尝试握手 fallback
            logger.warning(f"LLM [{purpose}] API unavailable, trying handshake fallback...")
            try:
                from evolution.llm_handshake import query_for_evolution
                system_msg = ""
                for m in messages:
                    if m["role"] == "system":
                        system_msg = m["content"]
                user_msg = ""
                for m in messages:
                    if m["role"] == "user":
                        user_msg = m["content"]
                return query_for_evolution(system_msg, user_msg, purpose=purpose)
            except Exception as hs_err:
                logger.error(f"Handshake fallback also failed: {hs_err}")
                raise last_error
    else:
        raise last_error

    try:
        content = data["choices"][0]["message"]["content"]
        # MiMo 有时 content=null 但 reasoning_content 有值
        if content is None:
            content = data["choices"][0]["message"].get("reasoning_content", "")
        if content is None:
            content = ""
        req_time = time.time() - start_time

        usage = data.get("usage", {})
        in_tokens = usage.get("prompt_tokens", 0)
        out_tokens = usage.get("completion_tokens", 0)

        logger.info(
            f"LLM [{purpose}] {model} — {req_time:.2f}s — "
            f"in:{in_tokens} out:{out_tokens} — {content[:60]}..."
        )
        return content

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM [{purpose}] call failed: {e}")
        raise
    except (KeyError, IndexError) as e:
        logger.error(f"LLM [{purpose}] response parse error: {e}")
        raise


def llm_call_stream(
    messages: List[Dict[str, str]],
    purpose: str = "chat_completions",
    **kwargs,
):
    """
    统一 LLM 流式调用入口。

    Yields:
        逐个 token 的文本片段
    """
    cfg = get_config()

    base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
    api_key = cfg.get("api_key", "")
    model = cfg.get("model", "gpt-4")

    overrides = cfg.get("overrides", {}).get(purpose, {})
    params = {**overrides, **kwargs}

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        **params,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM [{purpose}] stream failed: {e}")
        raise


# ── Embedding 调用 ────────────────────────────────────────────

def llm_embed(
    texts: List[str],
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    统一 Embedding 调用入口。

    Args:
        texts: 要向量化的文本列表
        model: 嵌入模型名称（不传则用配置值）

    Returns:
        向量列表
    """
    cfg = get_config()

    base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
    api_key = cfg.get("api_key", "")

    # embedding 可能有独立模型配置
    embedding_overrides = cfg.get("overrides", {}).get("embedding", {})
    embed_model = model or embedding_overrides.get("model", "text-embedding-3-small")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(
            f"{base_url}/embeddings",
            headers=headers,
            json={"input": texts, "model": embed_model},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM embedding failed: {e}")
        raise


# ── 便捷函数 ─────────────────────────────────────────────────

def get_provider_info() -> Dict[str, str]:
    """返回当前 provider 信息（用于健康检查）"""
    cfg = get_config()
    return {
        "provider": cfg.get("provider", "unknown"),
        "model": cfg.get("model", "unknown"),
        "base_url": cfg.get("base_url", ""),
        "has_api_key": bool(cfg.get("api_key")),
    }


# ── 模块入口（测试用）────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    print("=== LLM Gateway 测试 ===")
    info = get_provider_info()
    print(f"Provider: {info['provider']}")
    print(f"Model: {info['model']}")
    print(f"Base URL: {info['base_url']}")
    print(f"API Key: {'✅ 已配置' if info['has_api_key'] else '❌ 未配置'}")

    if "--test" in sys.argv:
        print("\n=== 调用测试 ===")
        try:
            reply = llm_call(
                [{"role": "user", "content": "你好，简短回复一句话"}],
                purpose="chat_completions",
            )
            print(f"回复: {reply}")
        except Exception as e:
            print(f"错误: {e}")
