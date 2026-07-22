"""
Trae LLM 后端 - 通过文件中转让 Trae（通过 MCP 调用进化系统的调用方）
接管所有 LLM 调用。

工作原理：
1. 进化系统的 backend.query() 被调用
2. 本模块把 prompt 写到 BRIDGE_DIR/pending_<call_id>.json
3. 阻塞轮询 BRIDGE_DIR/response_<call_id>.json
4. Trae 在会话里读 pending 文件，处理，写 response 文件
5. 本模块读到 response，返回给进化系统

对进化系统来说完全透明——和等 OpenAI HTTP 响应一样。
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("evolution.backend.trae")

# 中转目录：默认在项目根的 .trae_llm_bridge/
BRIDGE_DIR = Path(os.getenv(
    "TRAE_LLM_BRIDGE_DIR",
    Path(__file__).resolve().parents[2] / ".trae_llm_bridge",
))

# 单次 LLM 调用的超时（秒）：默认 5 分钟
TIMEOUT = int(os.getenv("TRAE_LLM_TIMEOUT", "600"))

# 轮询间隔（秒）
POLL_INTERVAL = 0.5


def _ensure_bridge_dir():
    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec=None,
    **model_kwargs,
) -> tuple[Any, float, int, int, dict]:
    """
    把 LLM 请求中转给 Trae。

    Args 和其他 backend 一致，返回 (output, req_time, in_tok, out_tok, info)。
    """
    _ensure_bridge_dir()

    call_id = uuid.uuid4().hex[:12]
    pending_path = BRIDGE_DIR / f"pending_{call_id}.json"
    response_path = BRIDGE_DIR / f"response_{call_id}.json"
    done_path = BRIDGE_DIR / f"done_{call_id}.json"  # 标记已消费

    # 构造请求
    request = {
        "call_id": call_id,
        "timestamp": time.time(),
        "system_message": system_message,
        "user_message": user_message,
        "model": model_kwargs.get("model", "trae"),
        "temperature": model_kwargs.get("temperature"),
        "max_tokens": model_kwargs.get("max_tokens"),
        "func_spec": None,
    }

    # 如果有 func_spec，序列化
    if func_spec is not None:
        request["func_spec"] = {
            "name": func_spec.name,
            "description": getattr(func_spec, "description", ""),
            "schema": getattr(func_spec, "json_schema", None) or getattr(func_spec, "parameters", None),
        }

    # 写 pending 文件
    with open(pending_path, "w", encoding="utf-8") as f:
        json.dump(request, f, ensure_ascii=False, indent=2)

    logger.info(f"[trae-bridge] LLM 请求 #{call_id} 已提交，等待 Trae 处理...")

    # 阻塞轮询等响应
    t0 = time.time()
    while True:
        if response_path.exists():
            try:
                with open(response_path, "r", encoding="utf-8") as f:
                    resp = json.load(f)
                # 标记已消费
                try:
                    done_path.write_text("1", encoding="utf-8")
                    pending_path.unlink(missing_ok=True)
                    response_path.unlink(missing_ok=True)
                except Exception:
                    pass
                req_time = time.time() - t0
                logger.info(
                    f"[trae-bridge] LLM 响应 #{call_id} 已收到 ({req_time:.2f}s)"
                )
                # 兼容 OpenAI 返回格式
                output = resp.get("output")
                if output is None:
                    # 也支持 content 字段
                    output = resp.get("content", "")
                in_tok = int(resp.get("in_tokens", 0))
                out_tok = int(resp.get("out_tokens", 0))
                info = {
                    "provider": "trae",
                    "call_id": call_id,
                    "model": resp.get("model", "trae"),
                }
                return output, req_time, in_tok, out_tok, info
            except json.JSONDecodeError as e:
                logger.warning(f"[trae-bridge] 响应 JSON 解析失败: {e}，继续等")
        # 超时检查
        if time.time() - t0 > TIMEOUT:
            try:
                pending_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise TimeoutError(
                f"Trae LLM 请求 #{call_id} 超时（{TIMEOUT}s）。"
                f" 可能原因：Trae 会话未在线、未轮询 pending 文件、或处理时间过长。"
            )
        time.sleep(POLL_INTERVAL)


def list_pending() -> list:
    """列出所有待处理的 LLM 请求（供 Trae 查询）"""
    _ensure_bridge_dir()
    pending = []
    for p in BRIDGE_DIR.glob("pending_*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                req = json.load(f)
            # 跳过已经有 response 的
            call_id = p.stem.replace("pending_", "")
            resp_path = BRIDGE_DIR / f"response_{call_id}.json"
            if not resp_path.exists():
                pending.append(req)
        except Exception:
            continue
    return pending


def submit_response(call_id: str, output: str, in_tokens: int = 0, out_tokens: int = 0, model: str = "trae") -> bool:
    """提交 LLM 响应（供 Trae 调用）"""
    _ensure_bridge_dir()
    resp_path = BRIDGE_DIR / f"response_{call_id}.json"
    pending_path = BRIDGE_DIR / f"pending_{call_id}.json"
    if not pending_path.exists():
        return False
    resp = {
        "call_id": call_id,
        "output": output,
        "content": output,  # 兼容字段
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "model": model,
        "timestamp": time.time(),
    }
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=2)
    return True


def clear_all():
    """清空所有中转文件"""
    _ensure_bridge_dir()
    for p in BRIDGE_DIR.glob("*.json"):
        try:
            p.unlink()
        except Exception:
            pass
