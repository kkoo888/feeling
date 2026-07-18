"""
memory_bridge.py — Hindsight 记忆桥接器
========================================
只做一件事：从 Hindsight 召回记忆，供 LAAP 各模块读取。

对接：Hindsight API http://127.0.0.1:6100
Bank：hermes
"""

import logging
from typing import Dict, List

import requests

logger = logging.getLogger("laap.memory_bridge")

HINDSIGHT_BASE = "http://127.0.0.1:6100"
BANK_ID = "hermes"
TIMEOUT = 10


def _recall(query: str, limit: int = 5) -> List[Dict]:
    """
    调 Hindsight recall API。

    Hindsight RecallRequest schema:
      - query (必填, str)
      - max_tokens (int, 默认 4096) — 控制 token 预算，间接决定结果数量
      - budget: "low" | "mid" | "high" (默认 "mid")
      - types: ["world", "experience", "observation"] 等
      - trace (bool, 默认 False)
    注意：schema 没有 `limit` 字段。老代码发 {"limit": ...} 会触发 422。
    `limit` 参数保留仅用于本地截断，不发送给 Hindsight。
    """
    url = f"{HINDSIGHT_BASE}/v1/default/banks/{BANK_ID}/memories/recall"
    try:
        # Hindsight 用 token 预算控制结果量，没有 limit 字段
        # 把 limit 映射到大致的 max_tokens (每条结果 ~500 tokens)
        max_tokens = max(500, int(limit) * 500)
        payload = {"query": query, "max_tokens": max_tokens, "trace": True}
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        # RecallResponse: {results: [...], trace: {...}, entities: {...}, chunks: {...}}
        results = resp.json().get("results", [])
        # RecallResult 每项含 id/text/type/entities/context/occurred_start/...
        return [
            {"content": r.get("text", ""), "score": 0.0, "type": r.get("type", "unknown")}
            for r in results
            if r.get("text")
        ][:limit]  # 本地按 limit 截断
    except requests.exceptions.ConnectionError:
        logger.warning("Hindsight 连接失败")
        return []
    except Exception as e:
        logger.warning(f"Hindsight recall 错误: {e}")
        return []


def get_memory_context(max_core: int = 3, max_recent: int = 3, max_working: int = 2) -> str:
    """获取记忆上下文，注入 PSI 循环"""
    memories = _recall("最近的对话和重要信息", limit=max_core + max_recent + max_working)
    if not memories:
        return ""
    lines = ["[记忆上下文]"]
    for i, m in enumerate(memories):
        lines.append(f"  {i+1}. {m['content'][:200]}")
    return "\n".join(lines)


def recall_related(query: str, top_k: int = 5) -> List[Dict]:
    """按语义搜索相关记忆"""
    return _recall(query, limit=top_k)



def is_available() -> bool:
    """检查 Hindsight 是否可用"""
    try:
        return requests.get(f"{HINDSIGHT_BASE}/health", timeout=3).status_code == 200
    except:
        return False
