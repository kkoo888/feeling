"""
小茜 Quantum Subconscious v2 — 委托给进化后的 SubconsciousEngine v2
==================================================================
在后台运行的低优先级线程：
  1. 接收对话中的话题种子
  2. 用 SubconsciousEngine v2 的 5 个子系统生成关联/直觉
  3. 这些直觉片段被注入到 PSI 循环的 perceive() 阶段
  4. 在 LLM 的理性之上叠加一层"灵感和直觉"

5 个子系统：
  - 自由联想（FreeAssociation）— 随机组合概念
  - 情感驱动（EmotionDriven）— 情绪影响联想方向（自含推断，无需外部engine）
  - 跨域联想（CrossDomain）— 跨话题/跨记忆联想
  - 梦境整合（DreamIntegration）— 离线整理记忆
  - 灵感涌现（InsightBurst）— 突然的创意

设计原则:
  - 潜意识不直接对话，只生成关联
  - 高相关性直觉会被提升到意识层（注入 PSI 上下文）
  - LLM 仍然是语言输出通道，但会受到潜意识的影响
  - 无需外部 V12.5 或 emotional_engine，纯 Python 自含运行
"""

import logging

import sys, os, time, json, threading, random
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from laap_brain.config import BRAIN_DIR as BRAIN

logger = logging.getLogger("aris.subconscious")

# 导入进化后的 V2 引擎
try:
    from subconscious_engine import SubconsciousEngine as _V2Engine
    _V2_AVAILABLE = True
    logger.info("SubconsciousEngine v2 loaded (40-round evolution)")
except ImportError as e:
    _V2_AVAILABLE = False
    logger.warning(f"SubconsciousEngine v2 unavailable: {e}")


# ── 数据结构（保持兼容）────────────────────────────────────

@dataclass
class Intuition:
    """一条潜意识直觉"""
    content: str
    source: str = "markov"
    coherence: float = 0.0
    emotional_tone: str = "neutral"
    timestamp: float = 0.0
    activated: bool = False
    seed_topics: List[str] = field(default_factory=list)


class QuantumSubconscious:
    """
    量子潜意识层 v2。
    内部委托给 SubconsciousEngine v2（40轮进化版）。
    对外接口保持不变。
    """

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # 种子队列
        self._seed_queue: deque = deque(maxlen=20)

        # 生成的直觉（兼容格式）
        self._intuitions: List[Intuition] = []
        self._max_intuitions = 50

        # V2 引擎
        self._v2: Optional[_V2Engine] = None
        self._init_engine()

        logger.info(f"QuantumSubconscious v2 initialized (interval={interval}s)")

    def _init_engine(self):
        """加载 V2 引擎"""
        if _V2_AVAILABLE:
            try:
                self._v2 = _V2Engine(interval=self.interval)
                logger.info("V2 engine loaded: 5 subsystems (free_assoc/emotion/cross_domain/dream/insight)")
            except Exception as e:
                logger.warning(f"V2 engine init failed: {e}")
                self._v2 = None
        else:
            logger.warning("No engine available, subconscious disabled")

    # ── 公开接口 ──────────────────────────────────────

    def feed(self, text: str, topics: List[str] = None):
        """向潜意识输入当前对话的种子"""
        with self._lock:
            words = self._extract_seeds(text)
            self._seed_queue.append({
                "text": text[:200],
                "words": words,
                "topics": topics or ["general"],
                "timestamp": time.time(),
            })

        # 同步喂入 V2 引擎
        if self._v2:
            self._v2.feed(text, topics)

    def get_intuitions(self, top_k: int = 3, min_coherence: float = 0.1,
                       consume: bool = True, generate_if_empty: bool = True) -> List[Intuition]:
        """
        获取最近的直觉。
        在 PSI 循环的 perceive() 阶段调用。
        """
        if self._v2:
            # 从 V2 引擎获取直觉
            v2_intuitions = self._v2.get_intuitions(
                top_k=top_k, min_coherence=min_coherence, consume=consume
            )
            # 转换为兼容格式
            result = []
            for vi in v2_intuitions:
                result.append(Intuition(
                    content=vi.content,
                    source=vi.source,
                    coherence=vi.coherence,
                    emotional_tone=vi.emotional_tone,
                    timestamp=vi.timestamp,
                    activated=vi.activated,
                    seed_topics=vi.seed_topics,
                ))
            return result

        # Fallback: 从本地直觉池获取
        with self._lock:
            available = [i for i in self._intuitions
                        if not i.activated and i.coherence >= min_coherence]
            available.sort(key=lambda x: -x.timestamp)
            results = available[:top_k]

            if not results and generate_if_empty and self._seed_queue:
                self._lock.release()
                self._generate_intuition_fallback()
                self._lock.acquire()
                available = [i for i in self._intuitions
                            if not i.activated and i.coherence >= min_coherence]
                available.sort(key=lambda x: -x.timestamp)
                results = available[:top_k]

            if consume:
                for r in results:
                    r.activated = True
            return results

    def get_random_intuition(self) -> Optional[str]:
        """获取一条随机直觉"""
        if self._v2:
            intuitions = self._v2.get_intuitions(top_k=5, consume=False)
            if intuitions:
                choice = random.choice(intuitions)
                return choice.content
            return None

        with self._lock:
            available = [i for i in self._intuitions if not i.activated and i.coherence >= 0.05]
            if available:
                choice = random.choice(available)
                choice.activated = True
                return choice.content
            return None

    def start(self):
        """启动后台潜意识线程"""
        if self._running:
            return

        if self._v2:
            self._v2.start()
            self._running = True
            logger.info("V2 subconscious engine started (5 subsystems)")
            return

        logger.warning("No engine available, subconscious disabled")
        return

    def stop(self):
        """停止后台线程"""
        self._running = False
        if self._v2:
            self._v2.stop()
            logger.info("V2 subconscious engine stopped")

    @property
    def is_running(self) -> bool:
        if self._v2:
            return self._v2.is_running
        return self._running

    # ── 内部 ──────────────────────────────────────────

    def _generate_intuition_fallback(self):
        """Fallback: V2不可用时的简单生成"""
        with self._lock:
            if not self._seed_queue:
                return
            seed = self._seed_queue[-1]
            words = seed["words"]
            topics = seed["topics"]

        if not words:
            return

        # 简单组合
        w1 = random.choice(words) if words else "存在"
        w2 = random.choice(words) if len(words) > 1 else "意义"
        templates = [
            f"{w1}和{w2}之间似乎有某种联系",
            f"想到{w1}，又联想到{w2}",
            f"{w1}的本质可能与{w2}有关",
        ]
        content = random.choice(templates)

        with self._lock:
            self._intuitions.append(Intuition(
                content=content,
                source="fallback",
                coherence=0.3,
                emotional_tone=topics[0] if topics else "neutral",
                timestamp=time.time(),
                seed_topics=topics,
            ))
            if len(self._intuitions) > self._max_intuitions:
                self._intuitions = self._intuitions[-self._max_intuitions:]

    def _extract_seeds(self, text: str) -> List[str]:
        """从文本提取种子词"""
        import re
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        words = []
        for segment in chinese_chars:
            if len(segment) >= 2:
                for i in range(len(segment) - 1):
                    words.append(segment[i:i+2])
            if len(segment) >= 1:
                words.append(segment[0])
            words.append(segment[:4])
        seen = set()
        result = []
        for w in words:
            w = w.strip()
            if w and len(w) >= 2 and w not in seen:
                seen.add(w)
                result.append(w)
        return result[:15]

    def status(self) -> dict:
        """状态"""
        if self._v2:
            v2_status = self._v2.get_status()
            return {
                "running": v2_status["running"],
                "engine": "SubconsciousEngine v2 (40-round evolution)",
                "engine_loaded": True,
                "seed_queue": len(self._seed_queue),
                "intuitions_generated": v2_status["intuitions_total"],
                "intuitions_unconsumed": v2_status["intuitions_unconsumed"],
                "interval": self.interval,
                "subsystems": {
                    "free_assoc": v2_status["stats"]["by_source"].get("free_assoc", 0),
                    "emotion": v2_status["stats"]["by_source"].get("emotion", 0),
                    "cross_domain": v2_status["stats"]["by_source"].get("cross_domain", 0),
                    "dream": v2_status["stats"].get("dream_integrations", 0),
                    "insight": v2_status["stats"].get("insight_bursts", 0),
                },
                "source_weights": v2_status.get("source_weights", {}),
                "concepts_count": v2_status.get("concepts_count", {}),
            }

        return {
            "running": self._running,
            "engine": "none",
            "engine_loaded": False,
            "seed_queue": len(self._seed_queue),
            "intuitions_generated": len(self._intuitions),
            "intuitions_unconsumed": sum(1 for i in self._intuitions if not i.activated),
            "interval": self.interval,
        }


# ── 全局单例 ────────────────────────────────────────────────

_subconscious: Optional[QuantumSubconscious] = None

def get_subconscious(interval: float = 5.0) -> QuantumSubconscious:
    global _subconscious
    if _subconscious is None:
        _subconscious = QuantumSubconscious(interval=interval)
    return _subconscious


def start_subconscious():
    """启动潜意识（在启动时调用）"""
    sc = get_subconscious()
    if not sc.is_running:
        sc.start()
        logger.info("Subconscious started")
    return sc


# ── CLI 测试 ────────────────────────────────────────────────

def main():
    """测试潜意识"""
    import argparse
    parser = argparse.ArgumentParser(description="小茜 Quantum Subconscious v2")
    parser.add_argument("--test", type=str, help="测试种子文本")
    parser.add_argument("--intuitions", action="store_true", help="显示已生成的直觉")
    parser.add_argument("--status", action="store_true", help="显示状态")
    args = parser.parse_args()

    sc = get_subconscious()

    if args.test:
        sc.feed(args.test, topics=["一般"])
        time.sleep(0.5)
        logger.info(f"种子: {args.test}")
        status = sc.status()
        logger.info(f"状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
        intuitions = sc.get_intuitions(top_k=3, consume=False)
        for i in intuitions:
            logger.info(f"  [{i.source}] coh={i.coherence:.2f} | {i.content[:80]}")
        return

    if args.intuitions:
        intuitions = sc.get_intuitions(top_k=10, consume=False)
        for i in intuitions:
            flag = "✓" if i.activated else " "
            logger.info(f"  [{flag}][{i.source}] coh={i.coherence:.2f} t={i.emotional_tone}")
            logger.info(f"    {i.content[:100]}")
        return

    logger.info(json.dumps(sc.status(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
