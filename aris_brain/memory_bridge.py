"""
memory_bridge.py — Hindsight 记忆桥接器
========================================
只做一件事：从 Hindsight 召回记忆，供 LAAP 各模块读取。

对接：Hindsight API http://127.0.0.1:6200
Bank：hermes
"""

import logging
from typing import Dict, List

import requests

logger = logging.getLogger("laap.memory_bridge")

HINDSIGHT_BASE = "http://127.0.0.1:6200"
BANK_ID = "hermes"
TIMEOUT = 10


def _recall(query: str, limit: int = 5) -> List[Dict]:
    """调 Hindsight recall API"""
    url = f"{HINDSIGHT_BASE}/v1/default/banks/{BANK_ID}/memories/recall"
    try:
        resp = requests.post(url, json={"query": query, "limit": limit}, timeout=TIMEOUT)
        resp.raise_for_status()
        memories = resp.json().get("memories", [])
        return [{"content": m.get("content", ""), "score": m.get("score", 0), "type": m.get("type", "unknown")} for m in memories if m.get("content")]
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


def store_important(content: str, tags: List[str] = None, mentioned_at: str = None) -> bool:
    """占位 — 保持接口兼容，实际不做任何事"""
    return True


def is_available() -> bool:
    """检查 Hindsight 是否可用"""
    try:
        return requests.get(f"{HINDSIGHT_BASE}/health", timeout=3).status_code == 200
    except:
        return False
