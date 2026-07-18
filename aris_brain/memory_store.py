"""
memory_store.py — LAAP 三层记忆存储（Hindsight 后端）
=====================================================
对外暴露：
  - MemoryFragment: 记忆片段数据类
  - MemoryStore: 三层记忆存储（核心/情景/工作）

内部对接 memory_bridge.py → Hindsight API
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("laap.memory_store")


@dataclass
class MemoryFragment:
    """一条记忆片段"""
    content: str
    layer: str = "episodic"         # core | episodic | working
    importance: float = 0.5         # 0.0 ~ 1.0
    topics: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    activated: bool = False


class MemoryStore:
    """
    三层记忆存储。

    - 核心记忆 (core): 身份、人格、长期关系
    - 情景记忆 (episodic): 对话经历、事件
    - 工作记忆 (working): 当前上下文

    内部通过 memory_bridge 对接 Hindsight。
    """

    def __init__(self):
        self._local_cache: Dict[str, List[MemoryFragment]] = {
            "core": [],
            "episodic": [],
            "working": [],
        }
        self._total_store = 0
        self._total_retrievals = 0

        # 检查 Hindsight 可用性
        try:
            from memory_bridge import is_available
            self._hindsight_ok = is_available()
            if self._hindsight_ok:
                logger.info("MemoryStore: Hindsight 后端已连接")
            else:
                logger.warning("MemoryStore: Hindsight 不可用，使用本地缓存")
        except Exception:
            self._hindsight_ok = False
            logger.warning("MemoryStore: memory_bridge 加载失败，使用本地缓存")

    def store(self, fragment: MemoryFragment):
        """存储一条记忆"""
        layer = fragment.layer if fragment.layer in self._local_cache else "episodic"
        self._local_cache[layer].append(fragment)
        self._total_store += 1

        # 同步到 Hindsight（占位，实际由 Hindsight 自己处理）

        # 工作记忆只保留最近 20 条
        if len(self._local_cache["working"]) > 20:
            self._local_cache["working"] = self._local_cache["working"][-20:]

    def get_stats(self) -> Dict:
        """获取记忆统计"""
        return {
            "core": len(self._local_cache["core"]),
            "episodic": len(self._local_cache["episodic"]),
            "working": len(self._local_cache["working"]),
            "total": sum(len(v) for v in self._local_cache.values()),
            "hindsight_connected": self._hindsight_ok,
        }

    def get_memory_embedding(self, query: str = "", layer: str = "core",
                              top_k: int = 3) -> np.ndarray:
        """
        获取记忆嵌入向量 (384-dim)。

        先尝试从 Hindsight recall 获取相关记忆，
        如果不可用则从本地缓存生成简单向量。
        """
        self._total_retrievals += 1

        # 尝试从 Hindsight 获取
        if self._hindsight_ok:
            try:
                from memory_bridge import recall_related
                results = recall_related(query or "记忆", top_k=top_k)
                if results:
                    # 用内容哈希生成伪向量（384维）
                    return self._texts_to_embedding([r["content"] for r in results])
            except Exception as e:
                logger.debug(f"Hindsight recall 失败: {e}")

        # Fallback: 从本地缓存
        fragments = self._local_cache.get(layer, [])
        if not fragments:
            return np.zeros(384, dtype=np.float32)

        texts = [f.content for f in fragments[-top_k:]]
        return self._texts_to_embedding(texts)

    def _texts_to_embedding(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转为 384 维嵌入向量。
        简单实现：用字符哈希生成伪向量。
        生产环境应替换为真实 embedding 模型。
        """
        vec = np.zeros(384, dtype=np.float32)
        for text in texts:
            for i, ch in enumerate(text[:384]):
                vec[i % 384] += ord(ch) / 65536.0
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def recall(self, query: str, top_k: int = 5) -> List[MemoryFragment]:
        """搜索相关记忆"""
        if self._hindsight_ok:
            try:
                from memory_bridge import recall_related
                results = recall_related(query, top_k=top_k)
                return [
                    MemoryFragment(
                        content=r["content"],
                        layer=r.get("type", "episodic"),
                        importance=r.get("score", 0.5),
                    )
                    for r in results
                ]
            except Exception:
                pass

        # Fallback: 从本地缓存搜索
        all_fragments = []
        for fragments in self._local_cache.values():
            all_fragments.extend(fragments)

        # 简单关键词匹配
        scored = []
        for f in all_fragments:
            overlap = len(set(query) & set(f.content))
            if overlap > 0:
                scored.append((overlap, f))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:top_k]]
