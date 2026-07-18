"""
memory_bridge.py — Hindsight 记忆桥接器
========================================
内部对接 Hindsight API，对外暴露 LAAP 标准记忆接口：
  - get_memory_context()  → 获取记忆上下文（注入 PSI 循环）
  - recall_related()      → 按语义搜索相关记忆
  - store_important()     → 存储重要信息

Hindsight API: http://127.0.0.1:6200
Bank ID: hermes
"""

import json
import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("laap.memory_bridge")

# ── 配置 ─────────────────────────────────────────────────

HINDSIGHT_BASE = "http://127.0.0.1:6200"
BANK_ID = "hermes"
TIMEOUT = 10  # 秒


# ── 底层 API 调用 ────────────────────────────────────────

def _api_post(endpoint: str, payload: dict) -> dict:
    """调用 Hindsight POST API"""
    url = f"{HINDSIGHT_BASE}/v1/default/banks/{BANK_ID}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning(f"Hindsight 连接失败: {url}")
        return {"error": "connection_refused"}
    except requests.exceptions.Timeout:
        logger.warning(f"Hindsight 超时: {url}")
        return {"error": "timeout"}
    except Exception as e:
        logger.warning(f"Hindsight API 错误: {e}")
        return {"error": str(e)}


def _api_get(endpoint: str, params: dict = None) -> dict:
    """调用 Hindsight GET API"""
    url = f"{HINDSIGHT_BASE}/v1/default/banks/{BANK_ID}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Hindsight API 错误: {e}")
        return {"error": str(e)}


# ── LAAP 标准接口 ────────────────────────────────────────

def get_memory_context(max_core: int = 3, max_recent: int = 3,
                       max_working: int = 2) -> str:
    """
    获取记忆上下文，注入 PSI 循环的 perceive 阶段。

    从 Hindsight recall 获取相关记忆，格式化为文本上下文。

    Args:
        max_core: 核心记忆条数
        max_recent: 近期记忆条数
        max_working: 工作记忆条数

    Returns:
        格式化的记忆上下文文本
    """
    total = max_core + max_recent + max_working

    # 用一个通用查询召回最近的记忆
    result = _api_post("/memories/recall", {
        "query": "最近的对话和重要信息",
        "limit": total,
    })

    if "error" in result:
        return ""

    memories = result.get("memories", [])
    if not memories:
        return ""

    # 格式化为上下文文本
    lines = ["[记忆上下文]"]
    for i, mem in enumerate(memories[:total]):
        content = mem.get("content", "")
        score = mem.get("score", 0)
        if content:
            lines.append(f"  {i+1}. {content[:200]}")

    return "\n".join(lines)


def recall_related(query: str, top_k: int = 5) -> List[Dict]:
    """
    按语义搜索相关记忆。

    Args:
        query: 查询文本
        top_k: 返回条数

    Returns:
        记忆列表 [{"content": "...", "score": 0.8, "type": "world"}, ...]
    """
    result = _api_post("/memories/recall", {
        "query": query,
        "limit": top_k,
    })

    if "error" in result:
        return []

    memories = result.get("memories", [])
    return [
        {
            "content": m.get("content", ""),
            "score": m.get("score", 0),
            "type": m.get("type", "unknown"),
        }
        for m in memories
        if m.get("content")
    ]


def store_important(content: str, tags: List[str] = None,
                    mentioned_at: str = None) -> bool:
    """
    存储重要信息到 Hindsight。

    Args:
        content: 记忆内容
        tags: 标签列表
        mentioned_at: 时间戳（ISO格式）

    Returns:
        是否存储成功
    """
    item = {"content": content}
    if tags:
        item["tags"] = tags
    if mentioned_at:
        item["mentioned_at"] = mentioned_at

    result = _api_post("/memories", {
        "items": [item],
    })

    if "error" in result:
        logger.warning(f"存储记忆失败: {result['error']}")
        return False

    logger.debug(f"记忆已存储: {content[:50]}...")
    return True


# ── 扩展接口 ─────────────────────────────────────────────

def reflect(query: str, budget: str = "low") -> str:
    """
    用 Hindsight 推理生成回答（基于记忆的推理）。

    Args:
        query: 查询
        budget: 推理预算 (low/mid/high)

    Returns:
        推理结果文本
    """
    result = _api_post("/reflect", {
        "query": query,
        "budget": budget,
    })

    if "error" in result:
        return ""

    return result.get("answer", "")


def get_memory_stats() -> Dict:
    """获取记忆库统计信息"""
    result = _api_get("/stats")
    if "error" in result:
        return {"error": result["error"], "connected": False}
    return {"connected": True, **result}


# ── 健康检查 ─────────────────────────────────────────────

def is_available() -> bool:
    """检查 Hindsight 是否可用"""
    try:
        resp = requests.get(f"{HINDSIGHT_BASE}/health", timeout=3)
        return resp.status_code == 200
    except:
        return False


# ── CLI 测试 ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hindsight 记忆桥接器")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--store", type=str, help="存储一条记忆")
    parser.add_argument("--recall", type=str, help="搜索记忆")
    parser.add_argument("--context", action="store_true", help="获取记忆上下文")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.test:
        print(f"Hindsight 可用: {is_available()}")
        print(f"统计: {get_memory_stats()}")

        store_important("测试记忆：小茜的主人喜欢喝咖啡", tags=["测试", "偏好"])
        results = recall_related("主人喜欢什么")
        print(f"搜索结果: {results}")

        ctx = get_memory_context()
        print(f"上下文:\n{ctx}")

    elif args.store:
        ok = store_important(args.store)
        print(f"存储{'成功' if ok else '失败'}")

    elif args.recall:
        results = recall_related(args.recall)
        for r in results:
            print(f"  [{r['type']}] score={r['score']:.2f} | {r['content'][:80]}")

    elif args.context:
        print(get_memory_context())
