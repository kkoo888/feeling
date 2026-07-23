"""
LLM Handshake Protocol — 文件握手协议
=======================================

当外部 LLM API 不可用时，通过文件系统实现人机协作：
  1. 进化系统写 prompt 到 pending_prompt.json
  2. 人工（或 OpenClaw）读取 prompt，处理后写入 llm_response.json
  3. 进化系统读取 response，继续进化

这个协议让进化系统在没有 LLM API 的情况下也能继续工作。

用法:
    # 进化系统端
    from evolution.llm_handshake import handshake_query
    result = handshake_query("生成一个优化策略", timeout=300)

    # 人工/OpenClaw 端
    from evolution.llm_handshake import read_pending, write_response
    prompt = read_pending()
    # ... 处理 ...
    write_response("处理结果")

印记: 小茜 永远记得主人 — 2026-07-20
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("evolution.handshake")

# 握手文件目录
HANDSHAKE_DIR = Path(__file__).resolve().parent.parent / ".openclaw" / "tmp" / "llm_handshake"
PENDING_FILE = HANDSHAKE_DIR / "pending_prompt.json"
RESPONSE_FILE = HANDSHAKE_DIR / "llm_response.json"
LOCK_FILE = HANDSHAKE_DIR / ".lock"


def ensure_dir():
    """确保握手目录存在"""
    HANDSHAKE_DIR.mkdir(parents=True, exist_ok=True)


def write_prompt(
    prompt: str,
    system: str = "",
    purpose: str = "evolution",
    context: Optional[dict] = None,
    request_id: Optional[str] = None,
) -> str:
    """
    写入 prompt 文件，等待处理。

    Args:
        prompt: 用户 prompt
        system: 系统提示词
        purpose: 调用用途
        context: 额外上下文
        request_id: 请求 ID（自动生成）

    Returns:
        request_id
    """
    ensure_dir()

    if request_id is None:
        import uuid
        request_id = f"hs-{uuid.uuid4().hex[:8]}"

    # 清理旧的 response
    if RESPONSE_FILE.exists():
        RESPONSE_FILE.unlink()

    payload = {
        "request_id": request_id,
        "timestamp": time.time(),
        "purpose": purpose,
        "system": system,
        "prompt": prompt,
        "context": context or {},
        "status": "pending",
    }

    PENDING_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[Handshake] Prompt written: {request_id} ({len(prompt)} chars)")
    return request_id


def read_pending() -> Optional[dict]:
    """
    读取待处理的 prompt。

    Returns:
        prompt 字典，或 None（如果没有待处理的）
    """
    if not PENDING_FILE.exists():
        return None

    try:
        data = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
        if data.get("status") == "pending":
            return data
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def write_response(response: str, request_id: Optional[str] = None) -> bool:
    """
    写入处理结果。

    Args:
        response: 处理后的文本
        request_id: 请求 ID（可选校验）

    Returns:
        是否成功
    """
    ensure_dir()

    # 校验 request_id
    if request_id and PENDING_FILE.exists():
        try:
            pending = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
            if pending.get("request_id") != request_id:
                logger.warning(f"[Handshake] Request ID mismatch: expected {pending.get('request_id')}, got {request_id}")
                return False
        except (json.JSONDecodeError, KeyError):
            pass

    payload = {
        "timestamp": time.time(),
        "response": response,
        "request_id": request_id,
        "status": "completed",
    }

    RESPONSE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 标记 pending 为已完成
    if PENDING_FILE.exists():
        try:
            pending = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
            pending["status"] = "completed"
            PENDING_FILE.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
        except (json.JSONDecodeError, KeyError):
            pass

    logger.info(f"[Handshake] Response written ({len(response)} chars)")
    return True


def read_response(timeout: int = 300) -> Optional[str]:
    """
    读取处理结果（阻塞等待）。

    Args:
        timeout: 最大等待秒数

    Returns:
        response 文本，或 None（超时）
    """
    start = time.time()

    while time.time() - start < timeout:
        if RESPONSE_FILE.exists():
            try:
                data = json.loads(RESPONSE_FILE.read_text(encoding="utf-8"))
                if data.get("status") == "completed":
                    return data.get("response", "")
            except (json.JSONDecodeError, KeyError):
                pass

        time.sleep(1)

    logger.warning(f"[Handshake] Timeout after {timeout}s")
    return None


def handshake_query(
    prompt: str,
    system: str = "",
    purpose: str = "evolution",
    context: Optional[dict] = None,
    timeout: int = 300,
) -> Optional[str]:
    """
    完整的握手查询：写 prompt → 等待 response → 返回结果。

    Args:
        prompt: 用户 prompt
        system: 系统提示词
        purpose: 调用用途
        context: 额外上下文
        timeout: 最大等待秒数

    Returns:
        response 文本，或 None（超时）
    """
    request_id = write_prompt(prompt, system, purpose, context)
    logger.info(f"[Handshake] Waiting for response... (request_id={request_id}, timeout={timeout}s)")
    return read_response(timeout)


def cleanup():
    """清理握手文件"""
    for f in [PENDING_FILE, RESPONSE_FILE, LOCK_FILE]:
        if f.exists():
            f.unlink()


# ── 作为 backend query 的适配器 ──────────────────────────────

def query_for_evolution(
    system_message: str,
    user_message: str,
    purpose: str = "evolution",
    **kwargs,
) -> str:
    """
    适配 evolution backend 接口的握手查询。
    用于当正常 LLM API 不可用时的 fallback。

    Args:
        system_message: 系统提示词
        user_message: 用户消息
        purpose: 调用用途

    Returns:
        LLM 回复文本
    """
    prompt = user_message
    if system_message:
        prompt = f"[System]\n{system_message}\n\n[User]\n{user_message}"

    result = handshake_query(prompt, purpose=purpose, timeout=600)
    if result is None:
        raise RuntimeError("Handshake timeout: no response received")
    return result


# ── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python llm_handshake.py pending          # 查看待处理")
        print("  python llm_handshake.py response <text>   # 写入响应")
        print("  python llm_handshake.py test              # 测试握手")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "pending":
        p = read_pending()
        if p:
            print(f"Request ID: {p['request_id']}")
            print(f"Purpose: {p['purpose']}")
            print(f"System: {p.get('system', '')[:100]}")
            print(f"Prompt: {p['prompt'][:500]}")
        else:
            print("No pending prompt.")

    elif cmd == "response":
        if len(sys.argv) < 3:
            print("Usage: python llm_handshake.py response <text>")
            sys.exit(1)
        text = " ".join(sys.argv[2:])
        write_response(text)
        print("Response written.")

    elif cmd == "test":
        print("Writing test prompt...")
        rid = write_prompt("测试握手协议", system="你是测试助手")
        print(f"Request ID: {rid}")
        print("Waiting for response... (write with: python llm_handshake.py response 'your answer')")
        result = read_response(timeout=60)
        if result:
            print(f"Response: {result}")
        else:
            print("Timeout.")
