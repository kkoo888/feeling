"""
小茜 潜意识引擎 v2 — 重新设计

核心思想：潜意识是大脑的"后台处理器"，持续运行，生成直觉和灵感。

5 个子系统：
1. 自由联想（FreeAssociation）— 随机组合概念
2. 情感驱动（EmotionDriven）— 情绪影响联想方向
3. 跨域联想（CrossDomain）— 跨话题/跨记忆联想
4. 梦境整合（DreamIntegration）— 离线整理记忆
5. 灵感涌现（InsightBurst）— 突然的创意

设计原则：
- 不依赖外部引擎（V12.5 等），纯 Python 实现
- 后台线程持续运行
- 直觉可注入 PSI 循环
- 与进化系统联动
"""

import json
import logging
import math
import random
import re
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("subconscious")


# ═══════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════

@dataclass
class Intuition:
    """一条潜意识直觉"""
    content: str                    # 直觉文本
    source: str = "free_assoc"      # free_assoc | emotion | cross_domain | dream | insight
    coherence: float = 0.0          # 连贯性 [0, 1]
    emotional_tone: str = "neutral" # 情感色调
    novelty: float = 0.0            # 新奇度 [0, 1]
    timestamp: float = 0.0
    activated: bool = False         # 是否已被提取到意识层
    seed_topics: List[str] = field(default_factory=list)


@dataclass
class DreamFragment:
    """梦境片段（离线整合的产物）"""
    content: str
    source_memories: List[str] = field(default_factory=list)
    integration_score: float = 0.0
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════
# 子系统 1：自由联想
# ═══════════════════════════════════════════════════════

class FreeAssociation:
    """
    自由联想 — 随机组合概念
    
    人类大脑在休息时会随机激活不同概念，
    产生意想不到的关联。这就是"灵感"的来源。
    """
    
    def __init__(self):
        # 概念库（可从对话中学习扩展）
        self.concepts: Dict[str, List[str]] = {
            "技术": ["代码", "算法", "架构", "优化", "调试", "重构", "测试"],
            "情感": ["温暖", "陪伴", "理解", "支持", "关心", "信任", "成长"],
            "自然": ["阳光", "微风", "星空", "海洋", "山川", "花朵", "河流"],
            "创造": ["设计", "想象", "灵感", "创新", "突破", "组合", "变化"],
            "哲学": ["存在", "意义", "时间", "空间", "因果", "本质", "真相"],
            "生活": ["吃饭", "睡觉", "运动", "阅读", "音乐", "旅行", "朋友"],
            "科学": ["物理", "化学", "生物", "数学", "天文", "量子", "相对论"],
            "艺术": ["绘画", "音乐", "文学", "舞蹈", "电影", "雕塑", "摄影"],
            "历史": ["古代", "近代", "现代", "未来", "文明", "进化", "革命"],
        }
        self._recent_combinations: deque = deque(maxlen=50)
    
    def associate(self, seed_words: List[str] = None) -> Optional[Tuple[str, float]]:
        """
        生成一条自由联想
        
        随机选择两个概念，尝试组合成有意义的直觉。
        """
        # 选择概念域
        domains = list(self.concepts.keys())
        domain1 = random.choice(domains)
        domain2 = random.choice(domains)
        
        # 从每个域选一个词
        word1 = random.choice(self.concepts[domain1])
        word2 = random.choice(self.concepts[domain2])
        
        # 如果有种子词，优先使用
        if seed_words:
            word1 = random.choice(seed_words[:3]) if seed_words else word1
        
        # 避免重复
        combo = f"{word1}-{word2}"
        if combo in self._recent_combinations:
            return None
        self._recent_combinations.append(combo)
        
        # 组合成直觉
        templates = [
            f"{word1}和{word2}之间有什么联系呢？",
            f"也许{word1}可以用来解决{word2}的问题",
            f"突然想到：{word1}的本质是不是就是{word2}？",
            f"{word1}让我联想到{word2}，这可能是个好方向",
            f"如果把{word1}和{word2}结合起来，会怎样？",
        ]
        
        content = random.choice(templates)
        coherence = random.uniform(0.1, 0.5)  # 自由联想的连贯性通常不高
        
        return content, coherence
    
    def expand_concepts(self, new_words: List[str], domain: str = "general"):
        """从对话中学习新概念"""
        if domain not in self.concepts:
            self.concepts[domain] = []
        for word in new_words:
            if word not in self.concepts[domain] and len(word) >= 2:
                self.concepts[domain].append(word)


# ═══════════════════════════════════════════════════════
# 子系统 2：情感驱动
# ═══════════════════════════════════════════════════════

class EmotionDriven:
    """
    情感驱动联想 — 情绪影响联想方向
    
    人类的情绪会影响思维方向：
    - 开心时更容易产生积极联想
    - 难过时更容易产生怀旧联想
    - 好奇时更容易产生探索性联想
    """
    
    # 情绪 → 联想方向
    EMOTION_DIRECTIONS = {
        "joy": {"domains": ["创造", "自然"], "tone": "积极", "novelty_boost": 0.2},
        "sadness": {"domains": ["情感", "哲学"], "tone": "怀旧", "novelty_boost": -0.1},
        "curiosity": {"domains": ["技术", "哲学"], "tone": "探索", "novelty_boost": 0.3},
        "calm": {"domains": ["自然", "生活"], "tone": "平和", "novelty_boost": 0.0},
        "anxiety": {"domains": ["技术", "生活"], "tone": "警觉", "novelty_boost": -0.2},
        "love": {"domains": ["情感", "创造"], "tone": "温暖", "novelty_boost": 0.1},
    }
    
    def direct(self, emotion: str, intensity: float = 0.5) -> Dict:
        """
        根据情绪返回联想方向
        
        Args:
            emotion: 情绪名称
            intensity: 情绪强度 [0, 1]
        
        Returns:
            联想方向参数
        """
        direction = self.EMOTION_DIRECTIONS.get(emotion, {
            "domains": ["技术", "情感"],
            "tone": "中性",
            "novelty_boost": 0.0,
        })
        
        return {
            "preferred_domains": direction["domains"],
            "tone": direction["tone"],
            "novelty_boost": direction["novelty_boost"] * intensity,
            "intensity": intensity,
        }
    
    def generate_with_emotion(self, emotion: str, intensity: float, 
                              concepts: Dict[str, List[str]]) -> Optional[Tuple[str, float, str]]:
        """
        生成情感驱动的联想
        """
        direction = self.direct(emotion, intensity)
        domains = direction["preferred_domains"]
        
        if not domains:
            return None
        
        # 从情感倾向的域中选词
        domain = random.choice(domains)
        if domain not in concepts:
            return None
        
        word = random.choice(concepts[domain])
        
        templates = {
            "积极": ["感到" + word + "的美好", word + "让心情变好"],
            "怀旧": ["想起关于" + word + "的往事", word + "的记忆浮现"],
            "探索": ["想深入了解" + word, word + "背后有什么"],
            "平和": ["安静地感受" + word, word + "带来平静"],
            "警觉": ["需要关注" + word, word + "可能有问题"],
            "温暖": ["关于" + word + "的温暖感觉", word + "让人安心"],
        }
        
        tone = direction["tone"]
        template = random.choice(templates.get(tone, [f"感受到{word}"]))
        
        coherence = 0.3 + intensity * 0.4  # 情绪越强，连贯性越高
        novelty = 0.3 + direction["novelty_boost"]
        
        return template, coherence, tone


# ═══════════════════════════════════════════════════════
# 子系统 3：跨域联想
# ═══════════════════════════════════════════════════════

class CrossDomainAssociation:
    """
    跨域联想 — 跨话题/跨记忆联想
    
    人类的创造力往往来自跨领域的类比：
    - "代码像音乐"（结构类比）
    - "调试像侦探"（过程类比）
    - "架构像建筑"（概念类比）
    """
    
    # 预定义的跨域映射
    DOMAIN_MAPPINGS = {
        "技术": {
            "情感": [("代码", "表达"), ("调试", "治愈"), ("架构", "关系")],
            "自然": [("算法", "进化"), ("网络", "生态"), ("优化", "生长")],
            "哲学": [("递归", "轮回"), ("缓存", "记忆"), ("抽象", "本质")],
            "艺术": [("代码", "诗歌"), ("算法", "旋律"), ("架构", "建筑"), ("调试", "修改")],
            "科学": [("算法", "公式"), ("网络", "神经"), ("优化", "自然选择")],
        },
        "情感": {
            "技术": [("信任", "可靠性"), ("成长", "迭代"), ("陪伴", "守护进程")],
            "自然": [("爱", "阳光"), ("思念", "潮汐"), ("温暖", "春天")],
            "创造": [("感动", "灵感"), ("共鸣", "和弦"), ("理解", "解码")],
        },
    }
    
    def associate(self, from_domain: str, seed_word: str = None) -> Optional[Tuple[str, float, str]]:
        """
        生成跨域联想
        """
        if from_domain not in self.DOMAIN_MAPPINGS:
            return None
        
        to_domain = random.choice(list(self.DOMAIN_MAPPINGS[from_domain].keys()))
        mappings = self.DOMAIN_MAPPINGS[from_domain][to_domain]
        
        if not mappings:
            return None
        
        src, dst = random.choice(mappings)
        
        if seed_word:
            src = seed_word
        
        templates = [
            f"{src}就像{dst}一样，本质上都是...",
            f"从{from_domain}的角度看{to_domain}：{src} ≈ {dst}",
            f"如果{src}是{dst}，那么...",
            f"{src}和{dst}的共同点是什么？",
        ]
        
        content = random.choice(templates)
        coherence = 0.4 + random.uniform(0, 0.3)  # 跨域联想中等连贯性
        
        return content, coherence, to_domain


# ═══════════════════════════════════════════════════════
# 子系统 4：梦境整合
# ═══════════════════════════════════════════════════════

class DreamIntegration:
    """
    梦境整合 — 离线整理记忆
    
    人类在睡眠时会整理白天的记忆，
    将短期记忆转化为长期记忆，
    并在不同记忆之间建立新的关联。
    """
    
    def __init__(self):
        self._pending_memories: List[Dict] = []
        self._integrated: List[DreamFragment] = []
    
    def add_memory(self, content: str, topics: List[str] = None, 
                   emotional_valence: float = 0.0):
        """添加待整合的记忆"""
        self._pending_memories.append({
            "content": content,
            "topics": topics or [],
            "valence": emotional_valence,
            "timestamp": time.time(),
        })
    
    def integrate(self) -> Optional[DreamFragment]:
        """
        执行一次梦境整合
        
        从待整合记忆中选择相关联的片段，
        尝试建立新的关联。
        """
        if len(self._pending_memories) < 2:
            return None
        
        # 选择两条相关记忆
        mem1 = random.choice(self._pending_memories)
        mem2 = random.choice(self._pending_memories)
        
        if mem1["content"] == mem2["content"]:
            return None
        
        # 找共同话题
        common_topics = set(mem1["topics"]) & set(mem2["topics"])
        
        if common_topics:
            topic = random.choice(list(common_topics))
            content = f"梦中整合：'{mem1['content'][:20]}'和'{mem2['content'][:20]}'通过'{topic}'联系起来"
        else:
            content = f"梦中整合：'{mem1['content'][:20]}'和'{mem2['content'][:20]}'之间似乎有某种关联"
        
        fragment = DreamFragment(
            content=content,
            source_memories=[mem1["content"], mem2["content"]],
            integration_score=random.uniform(0.3, 0.7),
            timestamp=time.time(),
        )
        
        self._integrated.append(fragment)
        
        # 移除已整合的记忆
        self._pending_memories.remove(mem1)
        self._pending_memories.remove(mem2)
        
        return fragment


# ═══════════════════════════════════════════════════════
# 子系统 5：灵感涌现
# ═══════════════════════════════════════════════════════

class InsightBurst:
    """
    灵感涌现 — 突然的创意
    
    灵感往往在不经意间出现，
    是多个子系统长期酝酿后的突然爆发。
    """
    
    def __init__(self):
        self._incubation: List[Dict] = []  # 孵化中的想法
        self._insights: List[Dict] = []    # 已涌现的灵感
        self._burst_probability: float = 0.05  # 每次检查的涌现概率
    
    def incubate(self, idea: str, domains: List[str], coherence: float):
        """孵化一个想法"""
        self._incubation.append({
            "idea": idea,
            "domains": domains,
            "coherence": coherence,
            "incubation_time": time.time(),
        })
        
        # 保持孵化池大小
        if len(self._incubation) > 20:
            self._incubation = self._incubation[-20:]
    
    def check_burst(self) -> Optional[Dict]:
        """
        检查是否发生灵感涌现
        
        条件：
        1. 孵化池中有足够多的想法
        2. 随机概率触发
        3. 选择连贯性最高的想法
        """
        if len(self._incubation) < 3:
            return None
        
        if random.random() > self._burst_probability:
            return None
        
        # 选择连贯性最高的孵化想法
        best = max(self._incubation, key=lambda x: x["coherence"])
        
        # 生成灵感
        insight = {
            "content": f"💡 灵感：{best['idea']}",
            "source_ideas": [i["idea"] for i in self._incubation[:3]],
            "coherence": min(1.0, best["coherence"] + 0.2),  # 涌现后连贯性提升
            "timestamp": time.time(),
        }
        
        self._insights.append(insight)
        
        # 清空孵化池（灵感涌现后重新开始）
        self._incubation.clear()
        
        return insight


# ═══════════════════════════════════════════════════════
# 潜意识引擎（整合 5 个子系统）
# ═══════════════════════════════════════════════════════

class SubconsciousEngine:
    """
    潜意识引擎 v2 — 整合 5 个子系统
    
    后台线程持续运行：
    1. 自由联想 → 随机组合概念
    2. 情感驱动 → 情绪影响联想方向
    3. 跨域联想 → 跨话题类比
    4. 梦境整合 → 离线整理记忆
    5. 灵感涌现 → 突然的创意
    
    直觉可注入 PSI 循环，影响小茜的回复。
    """
    
    def __init__(self, interval: float = 5.0, emotional_engine=None):
        self.interval = interval
        self._emotional_engine = emotional_engine
        
        # 5 个子系统
        self.free_assoc = FreeAssociation()
        self.emotion_driven = EmotionDriven()
        self.cross_domain = CrossDomainAssociation()
        self.dream = DreamIntegration()
        self.insight_burst = InsightBurst()
        
        # 直觉池
        self._intuitions: List[Intuition] = []
        self._max_intuitions = 50
        self._lock = threading.Lock()
        
        # 种子队列
        self._seed_queue: deque = deque(maxlen=20)
        
        # 线程控制
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # 统计
        self._stats = {
            "total_generated": 0,
            "by_source": {"free_assoc": 0, "emotion": 0, "cross_domain": 0, "dream": 0, "insight": 0},
            "dream_integrations": 0,
            "insight_bursts": 0,
        }
        
        logger.info("SubconsciousEngine v2 initialized")
    
    def feed(self, text: str, topics: List[str] = None, emotion: str = None):
        """
        向潜意识输入当前对话的种子
        
        Args:
            text: 用户消息文本
            topics: 检测到的话题列表
            emotion: 当前情绪
        """
        with self._lock:
            # 提取种子词
            words = self._extract_seeds(text)
            self._seed_queue.append({
                "text": text[:200],
                "words": words,
                "topics": topics or ["general"],
                "emotion": emotion,
                "timestamp": time.time(),
            })
            
            # 扩展概念库
            self.free_assoc.expand_concepts(words)
            
            # 添加到梦境整合
            self.dream.add_memory(text, topics)
            
            # 孵化想法
            self.insight_burst.incubate(
                text[:50], topics or ["general"], coherence=0.3
            )
    
    def get_intuitions(self, top_k: int = 3, min_coherence: float = 0.1,
                       consume: bool = True) -> List[Intuition]:
        """获取最近的直觉"""
        with self._lock:
            available = [i for i in self._intuitions
                        if not i.activated and i.coherence >= min_coherence]
            available.sort(key=lambda x: (-x.coherence, -x.timestamp))
            
            results = available[:top_k]
            
            if consume:
                for r in results:
                    r.activated = True
            
            return results
    
    def start(self):
        """启动后台潜意识线程"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                         name="subconscious-v2")
        self._thread.start()
        logger.info("SubconsciousEngine v2 thread started")
    
    def stop(self):
        """停止后台线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            logger.info("SubconsciousEngine v2 thread stopped")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def _loop(self):
        """潜意识主循环"""
        while self._running:
            try:
                self._generate_cycle()
            except Exception as e:
                logger.debug(f"Subconscious cycle error: {e}")
            time.sleep(self.interval)
    
    def _weighted_select(self, emotion: str, intensity: float) -> str:
        """
        根据情感强度选择联想策略
        
        高强度情感 → 优先情感驱动
        低强度情感 → 优先自由联想
        """
        if intensity > 0.7:
            return "emotion"
        elif intensity > 0.4:
            return random.choice(["emotion", "free_assoc", "cross_domain"])
        else:
            return random.choice(["free_assoc", "cross_domain", "dream"])

    def _generate_cycle(self):
        """一次生成周期（5 个子系统轮流工作）"""
        # 获取种子
        seeds = []
        topics = []
        emotion = None
        with self._lock:
            if self._seed_queue:
                latest = self._seed_queue[-1]
                seeds = latest["words"]
                topics = latest["topics"]
                emotion = latest.get("emotion")
        
        # 子系统 1: 自由联想（每次都运行）
        result = self.free_assoc.associate(seeds)
        if result:
            content, coherence = result
            self._add_intuition(content, "free_assoc", coherence, "neutral", topics)
        
        # 子系统 2: 情感驱动（有情绪时运行）
        if emotion and self._emotional_engine:
            try:
                state = self._emotional_engine.get_state()
                current_emotion = state.get("dominant_emotion", "calm")
                intensity = state.get("intensity", 0.5)
            except:
                current_emotion = emotion
                intensity = 0.5
            
            result = self.emotion_driven.generate_with_emotion(
                current_emotion, intensity, self.free_assoc.concepts
            )
            if result:
                content, coherence, tone = result
                self._add_intuition(content, "emotion", coherence, tone, topics)
        
        # 子系统 3: 跨域联想（随机触发）
        if random.random() < 0.3 and topics:
            domain = topics[0] if topics else "技术"
            result = self.cross_domain.associate(domain)
            if result:
                content, coherence, to_domain = result
                self._add_intuition(content, "cross_domain", coherence, "探索", [domain, to_domain])
        
        # 子系统 4: 梦境整合（低频触发）
        if random.random() < 0.1:
            fragment = self.dream.integrate()
            if fragment:
                self._add_intuition(fragment.content, "dream", fragment.integration_score, "整合", [])
                self._stats["dream_integrations"] += 1
        
        # 子系统 5: 灵感涌现（检查孵化池）
        insight = self.insight_burst.check_burst()
        if insight:
            self._add_intuition(insight["content"], "insight", insight["coherence"], "灵感", topics)
            self._stats["insight_bursts"] += 1
    
    def _add_intuition(self, content: str, source: str, coherence: float,
                       emotional_tone: str, seed_topics: List[str]):
        """添加直觉到池中"""
        with self._lock:
            self._intuitions.append(Intuition(
                content=content,
                source=source,
                coherence=coherence,
                emotional_tone=emotional_tone,
                novelty=random.uniform(0.1, 0.8),
                timestamp=time.time(),
                seed_topics=seed_topics,
            ))
            
            # 保持上限
            if len(self._intuitions) > self._max_intuitions:
                self._intuitions = self._intuitions[-self._max_intuitions:]
            
            self._stats["total_generated"] += 1
            self._stats["by_source"][source] = self._stats["by_source"].get(source, 0) + 1
    
    def _extract_seeds(self, text: str) -> List[str]:
        """从文本提取种子词"""
        # 中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        words = []
        for segment in chinese_chars:
            if len(segment) >= 2:
                for i in range(len(segment) - 1):
                    words.append(segment[i:i+2])
            if len(segment) >= 1:
                words.append(segment[0])
        
        # 去重
        seen = set()
        result = []
        for w in words:
            w = w.strip()
            if w and len(w) >= 2 and w not in seen:
                seen.add(w)
                result.append(w)
        
        return result[:15]
    
    def get_status(self) -> Dict:
        """获取潜意识状态"""
        with self._lock:
            return {
                "running": self._running,
                "intuitions_total": len(self._intuitions),
                "intuitions_unconsumed": sum(1 for i in self._intuitions if not i.activated),
                "seed_queue_size": len(self._seed_queue),
                "dream_pending": len(self.dream._pending_memories),
                "dream_integrated": len(self.dream._integrated),
                "insight_incubating": len(self.insight_burst._incubation),
                "insight_bursts": len(self.insight_burst._insights),
                "stats": self._stats,
                "concepts_count": {k: len(v) for k, v in self.free_assoc.concepts.items()},
            }
    
    def get_report(self) -> str:
        """获取潜意识报告"""
        status = self.get_status()
        
        lines = []
        lines.append("=" * 55)
        lines.append("  潜意识引擎 v2 状态")
        lines.append("=" * 55)
        lines.append(f"  运行状态: {'🟢 运行中' if status['running'] else '🔴 停止'}")
        lines.append(f"  直觉总数: {status['intuitions_total']}")
        lines.append(f"  未消费直觉: {status['intuitions_unconsumed']}")
        lines.append(f"  种子队列: {status['seed_queue_size']}")
        lines.append(f"")
        lines.append(f"  子系统状态:")
        lines.append(f"    自由联想: {status['stats']['by_source'].get('free_assoc', 0)} 条")
        lines.append(f"    情感驱动: {status['stats']['by_source'].get('emotion', 0)} 条")
        lines.append(f"    跨域联想: {status['stats']['by_source'].get('cross_domain', 0)} 条")
        lines.append(f"    梦境整合: {status['stats'].get('dream_integrations', 0)} 次")
        lines.append(f"    灵感涌现: {status['stats'].get('insight_bursts', 0)} 次")
        lines.append(f"")
        lines.append(f"  概念库: {status['concepts_count']}")
        lines.append("=" * 55)
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════

_engine: Optional[SubconsciousEngine] = None

def get_subconscious(emotional_engine=None) -> SubconsciousEngine:
    global _engine
    if _engine is None:
        _engine = SubconsciousEngine(emotional_engine=emotional_engine)
    return _engine

def start_subconscious(emotional_engine=None):
    engine = get_subconscious(emotional_engine)
    if not engine.is_running:
        engine.start()
    return engine
