"""
小茜 潜意识引擎 v3 — 进化版（40轮变异优化）
R21-R40: 自含情感引擎/语义桥接/源权重/梦境增强/否定能力/跨系统反馈/概念学习/新奇度/质量加权/多样性保留
"""

import json
import logging
import math
import random
import re
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

logger = logging.getLogger("subconscious")


@dataclass
class Intuition:
    content: str
    source: str = "free_assoc"
    coherence: float = 0.0
    emotional_tone: str = "neutral"
    novelty: float = 0.0
    timestamp: float = 0.0
    activated: bool = False
    seed_topics: List[str] = field(default_factory=list)
    quality: float = 0.0


@dataclass
class DreamFragment:
    content: str
    source_memories: List[str] = field(default_factory=list)
    integration_score: float = 0.0
    timestamp: float = 0.0


class FreeAssociation:
    def __init__(self):
        self.concepts: Dict[str, List[str]] = {
            "技术": ["代码", "算法", "架构", "优化", "调试", "重构", "测试", "接口", "协议", "缓存"],
            "情感": ["温暖", "陪伴", "理解", "支持", "关心", "信任", "成长", "共鸣", "依赖", "思念"],
            "自然": ["阳光", "微风", "星空", "海洋", "山川", "花朵", "河流", "潮汐", "季节", "生态系统"],
            "创造": ["设计", "想象", "灵感", "创新", "突破", "组合", "变化", "即兴", "混搭", "跨界"],
            "哲学": ["存在", "意义", "时间", "空间", "因果", "本质", "真相", "悖论", "自由意志", "意识"],
            "生活": ["吃饭", "睡觉", "运动", "阅读", "音乐", "旅行", "朋友", "仪式", "习惯", "选择"],
            "科学": ["物理", "化学", "生物", "天文", "量子", "相对论", "熵", "共振", "波粒二象性"],
            "艺术": ["绘画", "音乐", "文学", "舞蹈", "电影", "雕塑", "摄影", "书法", "戏剧", "装置"],
            "历史": ["古代", "近代", "现代", "未来", "文明", "进化", "革命", "传承", "兴衰", "复兴"],
            "心理学": ["认知", "潜意识", "记忆", "注意力", "偏见", "启发式", "元认知", "心流", "顿悟", "直觉"],
            "社会学": ["群体", "文化", "制度", "变迁", "互动", "规范", "认同", "共识", "冲突", "演化"],
            "数学": ["无穷", "对称", "拓扑", "分形", "概率", "混沌", "矩阵", "群论", "范畴", "不动点"],
            "经济学": ["供需", "博弈", "均衡", "外部性", "边际", "激励", "市场", "泡沫"],
            "语言学": ["语法", "隐喻", "语义", "语用", "修辞", "翻译", "歧义"],
        }
        self.concept_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._recent_combinations: deque = deque(maxlen=50)
        self._emergent_patterns: Dict[str, int] = {}

    def _boost_concept(self, word: str):
        self.concept_weights[word] = min(3.0, self.concept_weights[word] + 0.1)

    def _decay_concepts(self):
        for word in list(self.concept_weights.keys()):
            self.concept_weights[word] = max(0.5, self.concept_weights[word] * 0.995)

    def _spread_activation(self, word: str, concepts: Dict[str, List[str]]):
        for domain, words in concepts.items():
            for w in words:
                if w == word:
                    continue
                overlap = len(set(word) & set(w))
                if overlap >= 1 and len(word) >= 2 and len(w) >= 2:
                    self.concept_weights[w] = min(2.0, self.concept_weights[w] + 0.05 * overlap)

    def associate(self, seed_words: List[str] = None) -> Optional[Tuple[str, float]]:
        domains = list(self.concepts.keys())
        if seed_words:
            for sw in seed_words[:2]:
                self._spread_activation(sw, self.concepts)

        domain_weights = []
        for d in domains:
            avg_w = sum(self.concept_weights.get(w, 1.0) for w in self.concepts[d]) / max(len(self.concepts[d]), 1)
            domain_weights.append(avg_w)
        total_w = sum(domain_weights)
        if total_w > 0:
            r = random.uniform(0, total_w)
            cumul = 0
            domain1 = domains[-1]
            for i, dw in enumerate(domain_weights):
                cumul += dw
                if r <= cumul:
                    domain1 = domains[i]
                    break
        else:
            domain1 = random.choice(domains)
        domain2 = random.choice(domains)

        word1 = random.choice(self.concepts[domain1])
        word2 = random.choice(self.concepts[domain2])
        if seed_words:
            word1 = random.choice(seed_words[:3]) if seed_words else word1

        combo = f"{word1}-{word2}"
        if combo in self._recent_combinations:
            return None
        self._recent_combinations.append(combo)
        self._boost_concept(word1)
        self._boost_concept(word2)

        pattern_key = f"{min(word1,word2)}:{max(word1,word2)}"
        self._emergent_patterns[pattern_key] = self._emergent_patterns.get(pattern_key, 0) + 1

        templates = [
            f"{word1}和{word2}之间有什么联系呢？",
            f"也许{word1}可以用来解决{word2}的问题",
            f"突然想到：{word1}的本质是不是就是{word2}？",
            f"{word1}让我联想到{word2}，这可能是个好方向",
            f"如果把{word1}和{word2}结合起来，会怎样？",
            f"{word1}是{word2}的密码，解开它就能看到真相",
            f"在{word1}的深处，{word2}正在沉睡",
            f"{word1}和{word2}的碰撞产生了第三种可能",
            f"当{word1}遇到{word2}，新的维度被打开了",
            f"如果{word1}是一种{word2}，那么它的'反面'是什么？",
            f"把{word1}缩小1000倍看{word2}，和放大1000倍看，结论会一样吗？",
            f"从{word1}的时间尺度看{word2}：瞬间 vs 永恒",
            f"{word1}的边界在哪里？{word2}能帮助找到答案",
            f"反转思维：如果{word2}先于{word1}存在，世界会怎样？",
            f"组合创新：{word1}+{word2}=第三种从未存在过的东西",
            f"递归联想：{word1}→{word2}→?，下一个会是什么？",
            f"对称性探索：{word1}和{word2}互为镜像吗？",
            f"维度提升：从{word1}到{word2}，我们上升了一个抽象层次",
            f"涌现性：{word1}和{word2}的简单组合产生了复杂的第三种东西",
            f"否定能力：{word1}既是{word2}又不是{word2}——这种张力本身就是答案",
            f"悖论深化：{word1}和{word2}的矛盾不是bug，是feature",
            f"悬置判断：关于{word1}和{word2}的关系，也许暂时不需要答案",
            f"对立共存：{word1}的光明面是{word2}，黑暗面也是{word2}",
        ]
        content = random.choice(templates)
        coherence = random.uniform(0.15, 0.55)
        if random.random() < 0.4:
            chain_words = [random.choice(self.concepts.get(d, ["存在"])) for d in random.sample(list(self.concepts.keys()), min(2, len(self.concepts)))]
            content += f" [{' → '.join(chain_words)}]"
        return content, coherence

    def extract_new_concepts(self, text: str) -> List[str]:
        chinese = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        stop_words = {"什么", "这个", "那个", "可以", "就是", "不是", "如果", "那么", "之间", "一样"}
        return [w for w in chinese if w not in stop_words and len(w) >= 2][:5]

    def expand_concepts(self, new_words: List[str], domain: str = "general"):
        if domain not in self.concepts:
            self.concepts[domain] = []
        for word in new_words:
            if word not in self.concepts[domain] and len(word) >= 2:
                self.concepts[domain].append(word)

    def get_emergent_patterns(self, min_count: int = 2) -> List[Tuple[str, int]]:
        return sorted([(k, v) for k, v in self._emergent_patterns.items() if v >= min_count], key=lambda x: -x[1])


class EmotionDriven:
    EMOTION_DIRECTIONS = {
        "joy": {"domains": ["创造", "自然"], "tone": "积极", "novelty_boost": 0.2},
        "sadness": {"domains": ["情感", "哲学"], "tone": "怀旧", "novelty_boost": -0.1},
        "curiosity": {"domains": ["技术", "哲学"], "tone": "探索", "novelty_boost": 0.3},
        "calm": {"domains": ["自然", "生活"], "tone": "平和", "novelty_boost": 0.0},
        "anxiety": {"domains": ["技术", "生活"], "tone": "警觉", "novelty_boost": -0.2},
        "love": {"domains": ["情感", "创造"], "tone": "温暖", "novelty_boost": 0.1},
        "wonder": {"domains": ["哲学", "科学"], "tone": "惊叹", "novelty_boost": 0.4},
        "nostalgia": {"domains": ["历史", "生活"], "tone": "怀旧", "novelty_boost": -0.05},
        "determination": {"domains": ["技术", "创造"], "tone": "坚定", "novelty_boost": 0.15},
        "serenity": {"domains": ["自然", "哲学"], "tone": "宁静", "novelty_boost": 0.05},
        "passion": {"domains": ["创造", "艺术"], "tone": "热烈", "novelty_boost": 0.35},
    }
    EMOTION_KEYWORDS = {
        "joy": ["开心", "愉快", "高兴", "快乐", "美好", "幸福", "喜欢"],
        "sadness": ["难过", "悲伤", "孤独", "寂寞", "思念", "遗憾", "失落"],
        "curiosity": ["思考", "为什么", "探索", "好奇", "想了解", "研究", "发现"],
        "calm": ["平静", "安静", "宁静", "放松", "舒适", "安心"],
        "anxiety": ["担心", "焦虑", "紧张", "害怕", "不安", "压力"],
        "love": ["爱", "喜欢", "信任", "陪伴", "关心", "温暖", "朋友"],
        "awe": ["震撼", "敬畏", "神奇", "不可思议", "量子", "宇宙", "星空"],
        "surprise": ["意外", "突然", "没想到", "惊喜", "震惊"],
        "anger": ["生气", "愤怒", "不满", "讨厌", "烦"],
        "wonder": ["奇妙", "美丽", "惊叹", "感悟", "领悟"],
    }

    def infer_emotion(self, text: str) -> Tuple[str, float]:
        scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[emotion] = score
        if not scores:
            return "calm", 0.3
        best = max(scores, key=scores.get)
        return best, min(1.0, scores[best] * 0.3 + 0.2)

    def direct(self, emotion: str, intensity: float = 0.5) -> Dict:
        direction = self.EMOTION_DIRECTIONS.get(emotion, {"domains": ["技术", "情感"], "tone": "中性", "novelty_boost": 0.0})
        return {"preferred_domains": direction["domains"], "tone": direction["tone"],
                "novelty_boost": direction["novelty_boost"] * intensity, "intensity": intensity}

    def generate_with_emotion(self, emotion: str, intensity: float, concepts: Dict[str, List[str]]) -> Optional[Tuple[str, float, str]]:
        direction = self.direct(emotion, intensity)
        domains = direction["preferred_domains"]
        if not domains:
            return None
        domain = random.choice(domains)
        if domain not in concepts:
            return None
        word = random.choice(concepts[domain])
        templates = {
            "积极": [f"感到{word}的美好", f"{word}让心情变好", f"在{word}中看到了希望"],
            "怀旧": [f"想起关于{word}的往事", f"{word}的记忆浮现", f"{word}让人回到过去"],
            "探索": [f"想深入了解{word}", f"{word}背后有什么", f"{word}的秘密等待揭开"],
            "平和": [f"安静地感受{word}", f"{word}带来平静", f"在{word}中找到安宁"],
            "警觉": [f"需要关注{word}", f"{word}可能有问题", f"{word}暗藏着风险"],
            "温暖": [f"关于{word}的温暖感觉", f"{word}让人安心", f"{word}是心底的柔软"],
            "惊叹": [f"被{word}的深邃所震撼", f"{word}展现了世界的奇妙", f"{word}超越了想象"],
            "坚定": [f"对{word}的信念更加坚定", f"{word}让人找到方向", f"{word}是前行的锚"],
            "宁静": [f"在{word}中找到内心的宁静", f"{word}如清泉般洗涤心灵", f"{word}让时间慢下来"],
            "热烈": [f"对{word}充满热情", f"{word}点燃了内心的火焰", f"{word}让人沸腾"],
        }
        tone = direction["tone"]
        template = random.choice(templates.get(tone, [f"感受到{word}"]))
        return template, 0.3 + intensity * 0.4, tone


class CrossDomainAssociation:
    DOMAIN_MAPPINGS = {
        "技术": {"情感": [("代码", "表达"), ("调试", "治愈"), ("架构", "关系"), ("重构", "成长")],
                "自然": [("算法", "进化"), ("网络", "生态"), ("优化", "生长"), ("缓存", "冬眠")],
                "哲学": [("递归", "轮回"), ("缓存", "记忆"), ("抽象", "本质"), ("并发", "自由意志")],
                "艺术": [("代码", "诗歌"), ("算法", "旋律"), ("架构", "建筑"), ("调试", "修改")],
                "科学": [("算法", "公式"), ("网络", "神经"), ("优化", "自然选择")],
                "生活": [("编程", "烹饪"), ("调试", "排错"), ("版本控制", "日记")],
                "数学": [("算法", "证明"), ("数据结构", "几何"), ("递归", "归纳法")]},
        "情感": {"技术": [("信任", "可靠性"), ("成长", "迭代"), ("陪伴", "守护进程")],
                "自然": [("爱", "阳光"), ("思念", "潮汐"), ("温暖", "春天")],
                "创造": [("感动", "灵感"), ("共鸣", "和弦"), ("理解", "解码")],
                "哲学": [("孤独", "存在"), ("自由", "选择"), ("痛苦", "觉醒")],
                "艺术": [("悲伤", "蓝调"), ("喜悦", "快板"), ("思念", "小夜曲")]},
        "自然": {"技术": [("生态", "分布式"), ("进化", "遗传算法"), ("潮汐", "周期任务")],
                "哲学": [("四季", "轮回"), ("星空", "无穷"), ("潮汐", "宿命")],
                "艺术": [("日落", "油画"), ("海浪", "交响乐"), ("森林", "合唱")],
                "数学": [("雪花", "分形"), ("螺旋", "黄金比例"), ("波浪", "正弦")]},
        "哲学": {"技术": [("存在", "初始化"), ("因果", "事件驱动"), ("悖论", "死锁")],
                "艺术": [("虚无", "留白"), ("永恒", "经典"), ("矛盾", "张力")],
                "心理学": [("意识", "元认知"), ("自由意志", "决策"), ("悖论", "认知失调")]},
        "心理学": {"技术": [("认知偏差", "bug"), ("心流", "高性能模式"), ("潜意识", "后台进程"), ("记忆", "缓存")],
                  "哲学": [("元认知", "自我反思"), ("偏见", "认知局限"), ("启发式", "直觉智慧")],
                  "自然": [("心流", "河流"), ("记忆", "化石"), ("认知", "进化")],
                  "艺术": [("潜意识", "超现实主义"), ("顿悟", "灵光一现"), ("直觉", "即兴创作")]},
        "艺术": {"技术": [("韵律", "节奏"), ("构图", "架构"), ("色彩", "调色板"), ("笔触", "代码风格")],
                "科学": [("美学", "黄金比例"), ("和谐", "共振"), ("节奏", "周期")],
                "哲学": [("创作", "存在"), ("表达", "意义"), ("审美", "价值")],
                "自然": [("色彩", "四季"), ("构图", "地貌"), ("韵律", "潮汐")],
                "历史": [("文艺复兴", "创新"), ("印象派", "感知革命"), ("前卫", "颠覆")]},
        "科学": {"技术": [("实验", "测试"), ("假设", "设计"), ("定律", "算法")],
                "哲学": [("因果", "决定论"), ("不确定", "自由意志"), ("观察者", "意识")],
                "自然": [("引力", "吸引"), ("熵增", "衰老"), ("共振", "共鸣")],
                "艺术": [("对称", "美"), ("分形", "自然图案"), ("混沌", "抽象画")],
                "数学": [("微积分", "变化"), ("群论", "对称"), ("拓扑", "变形")]},
        "历史": {"哲学": [("轮回", "循环"), ("兴衰", "无常"), ("传承", "延续")],
                "技术": [("工业革命", "信息革命"), ("古法", "算法"), ("发明", "创新")],
                "自然": [("四季", "朝代"), ("潮汐", "兴衰"), ("化石", "遗迹")],
                "社会学": [("帝国", "组织"), ("革命", "变革"), ("条约", "协议")]},
        "生活": {"技术": [("烹饪", "编程"), ("旅行", "探索"), ("园艺", "运维")],
                "哲学": [("选择", "自由"), ("习惯", "规律"), ("衰老", "时间")],
                "自然": [("呼吸", "潮汐"), ("节奏", "季节"), ("生长", "积累")]},
        "数学": {"哲学": [("无穷", "无限"), ("对称", "和谐"), ("概率", "命运")],
                "艺术": [("黄金比例", "美"), ("分形", "图案"), ("拓扑", "变形")],
                "自然": [("螺旋", "贝壳"), ("对称", "蝴蝶"), ("分形", "雪花")],
                "心理学": [("概率", "判断"), ("优化", "决策"), ("模式", "认知")]},
        "社会学": {"技术": [("制度", "协议"), ("规范", "标准"), ("变迁", "迭代")],
                  "心理学": [("群体思维", "从众"), ("文化", "集体潜意识"), ("认同", "归属")],
                  "历史": [("革命", "转折"), ("传承", "记忆"), ("冲突", "战争")]},
        "经济学": {"技术": [("市场", "算法"), ("博弈", "策略"), ("均衡", "稳定态")],
                  "心理学": [("激励", "动机"), ("泡沫", "认知偏差"), ("风险", "恐惧")],
                  "自然": [("供需", "生态平衡"), ("周期", "季节"), ("衰退", "冬天")]},
        "语言学": {"技术": [("语法", "规则"), ("语义", "数据"), ("翻译", "转换")],
                  "艺术": [("隐喻", "意象"), ("修辞", "表达"), ("韵律", "节奏")],
                  "心理学": [("歧义", "认知"), ("语用", "社交"), ("习得", "学习")]},
    }

    def _dynamic_mapping(self, from_domain, concepts):
        if from_domain not in concepts or not concepts[from_domain]:
            return None
        other_domains = [d for d in concepts if d != from_domain and concepts[d]]
        if not other_domains:
            return None
        to_domain = random.choice(other_domains)
        return random.choice(concepts[from_domain]), random.choice(concepts[to_domain]), to_domain

    def _ngram_similarity(self, a, b, n=2):
        if len(a) < n or len(b) < n:
            return 0.0
        ga = set(a[i:i+n] for i in range(len(a)-n+1))
        gb = set(b[i:i+n] for i in range(len(b)-n+1))
        return len(ga & gb) / len(ga | gb) if ga and gb else 0.0

    def _find_bridge(self, src, dst, from_domain, to_domain, concepts):
        best, best_score = None, 0.15
        for d, words in concepts.items():
            if d == from_domain or d == to_domain:
                continue
            for w in words:
                score = max(self._ngram_similarity(w, src), self._ngram_similarity(w, dst))
                if score > best_score:
                    best_score, best = score, (w, d)
        return best

    def associate(self, from_domain, seed_word=None, concepts=None):
        use_dynamic = from_domain not in self.DOMAIN_MAPPINGS or random.random() < 0.25
        if use_dynamic and concepts:
            dyn = self._dynamic_mapping(from_domain, concepts)
            if dyn:
                src, dst, to_domain = dyn
                templates = [f"跨域直觉：{src}({from_domain})的本质可以用{dst}({to_domain})来理解",
                             f"类比涌现：{src}和{dst}之间存在深层结构相似性",
                             f"概念桥接：{from_domain}的'{src}'通向{to_domain}的'{dst}'"]
                return random.choice(templates), 0.35 + random.uniform(0, 0.25), to_domain
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
            f"{src}就像{dst}一样，本质上都是...", f"从{from_domain}的角度看{to_domain}：{src} ≈ {dst}",
            f"如果{src}是{dst}，那么...", f"{src}和{dst}的共同点是什么？",
            f"深层结构：{src}({from_domain}) 和 {dst}({to_domain}) 共享同一种底层模式",
            f"类比推理：{src}之于{from_domain}，正如{dst}之于{to_domain}",
            f"跨域洞察：{src}的运作机制可以用{dst}的框架来理解",
            f"结构性同构：{src}和{dst}虽然领域不同，但遵循相似的演化规律",
            f"隐喻展开：把{dst}当作透镜来看{from_domain}的{src}，会发现...",
            f"双向映射：{src}→{dst}的类比反过来也成立——{dst}也能被{src}解释",
            f"结构类比：{src}的内部构造与{dst}惊人地相似——都是由嵌套的层次组成",
            f"功能类比：{src}在{from_domain}中扮演的角色，正是{dst}在{to_domain}中的角色",
            f"关系类比：{src}与其他{from_domain}概念的关系模式，映射到{dst}与{to_domain}概念的关系模式",
            f"隐喻链：{from_domain}的'{src}'如同{to_domain}的'{dst}'，两者指向更深层的统一原理",
            f"对立统一：{src}和{dst}看似相反，实则是同一枚硬币的两面",
            f"涌现视角：当{src}({from_domain})与{dst}({to_domain})相遇，新的模式从交汇处涌现",
        ]
        content = random.choice(templates)
        coherence = 0.4 + random.uniform(0, 0.3)
        if random.random() < 0.25 and concepts:
            bridge = self._find_bridge(src, dst, from_domain, to_domain, concepts)
            if bridge:
                content += f" [桥接：{bridge[0]}({bridge[1]})连接{src}↔{dst}]"
                coherence = min(1.0, coherence + 0.15)
        if random.random() < 0.3 and to_domain in self.DOMAIN_MAPPINGS:
            sm = self.DOMAIN_MAPPINGS[to_domain]
            if sm:
                td2 = random.choice(list(sm.keys()))
                p2 = random.choice(sm[td2])
                content += f" → 三域贯通：{src}→{dst}→{p2[1]}，一条跨域的思维链"
                coherence = min(1.0, coherence + 0.1)
                if random.random() < 0.1 and td2 in self.DOMAIN_MAPPINGS:
                    third = self.DOMAIN_MAPPINGS[td2]
                    if third:
                        td3 = random.choice(list(third.keys()))
                        p3 = random.choice(third[td3])
                        content += f" → 延伸：{p3[1]}({td3})是这条链的终点"
                        coherence = min(1.0, coherence + 0.08)
        return content, coherence, to_domain


class DreamIntegration:
    def __init__(self):
        self._pending_memories: List[Dict] = []
        self._integrated: List[DreamFragment] = []
        self._memory_graph: Dict[str, List[str]] = {}

    def add_memory(self, content, topics=None, emotional_valence=0.0):
        self._pending_memories.append({"content": content, "topics": topics or [], "valence": emotional_valence, "timestamp": time.time()})

    def add_unresolved_intuition(self, intuition):
        self._pending_memories.append({"content": intuition.content, "topics": intuition.seed_topics, "valence": 0.0, "timestamp": time.time(), "unresolved": True})

    def _topic_similarity(self, m1, m2):
        t1, t2 = set(m1.get("topics", [])), set(m2.get("topics", []))
        return len(t1 & t2) / len(t1 | t2) if t1 and t2 else 0.0

    def integrate(self):
        if len(self._pending_memories) > 50:
            self._pending_memories = self._pending_memories[-50:]
        if len(self._pending_memories) < 2:
            return None
        best_pair, best_sim = None, -1
        for _ in range(min(10, len(self._pending_memories))):
            m1, m2 = random.choice(self._pending_memories), random.choice(self._pending_memories)
            if m1["content"] != m2["content"]:
                sim = self._topic_similarity(m1, m2)
                if sim > best_sim:
                    best_sim, best_pair = sim, (m1, m2)
        if not best_pair:
            return None
        mem1, mem2 = best_pair
        common_topics = set(mem1.get("topics", [])) & set(mem2.get("topics", []))
        er = abs(mem1.get("valence", 0) - mem2.get("valence", 0))
        tag = " [未解之谜]" if (mem1.get("unresolved") or mem2.get("unresolved")) else ""
        if common_topics:
            topic = random.choice(list(common_topics))
            templates = [f"梦中整合：'{mem1['content'][:20]}'和'{mem2['content'][:20]}'通过'{topic}'联系起来",
                         f"梦境发现：'{mem1['content'][:20]}'和'{mem2['content'][:20]}'产生了共鸣 [{topic}]",
                         f"深层整合：'{mem1['content'][:20]}'和'{mem2['content'][:20]}'的共同模式浮现——{topic}"]
            content = random.choice(templates)
        else:
            content = random.choice([f"梦境跨域：'{mem1['content'][:20]}'和'{mem2['content'][:20]}'潜意识发现了隐藏联系",
                                     f"异域整合：'{mem1['content'][:20]}' × '{mem2['content'][:20]}' = 全新理解"])
        content += tag
        score = random.uniform(0.4, 0.8) if er < 0.3 else random.uniform(0.3, 0.6)
        fragment = DreamFragment(content=content, source_memories=[mem1["content"], mem2["content"]], integration_score=score, timestamp=time.time())
        self._integrated.append(fragment)
        self._pending_memories.remove(mem1)
        self._pending_memories.remove(mem2)
        return fragment


class InsightBurst:
    def __init__(self):
        self._incubation: List[Dict] = []
        self._insights: List[Dict] = []
        self._burst_probability = 0.18

    def incubate(self, idea, domains, coherence):
        resonance = 0.0
        for e in self._incubation:
            if set(domains) & set(e["domains"]):
                resonance += 0.15 * len(set(domains) & set(e["domains"]))
            if len(set(idea) & set(e["idea"])) > 3:
                resonance += 0.1
        self._incubation.append({"idea": idea, "domains": domains, "coherence": coherence,
                                  "incubation_time": time.time(), "maturity": 0.0, "resonance": min(1.0, resonance), "access_count": 0})
        if len(self._incubation) > 25:
            self._incubation.sort(key=lambda x: x.get("maturity", 0) + x.get("resonance", 0), reverse=True)
            self._incubation = self._incubation[:15]

    def _compute_quality(self, ideas, pairs):
        if not ideas:
            return 0.0
        am = sum(i.get("maturity", 0) for i in ideas) / len(ideas)
        ar = sum(i.get("resonance", 0) for i in ideas) / len(ideas)
        ad = set()
        for i in ideas:
            ad.update(i.get("domains", []))
        dd = min(1.0, len(ad) / 4.0)
        ac = sum(i.get("coherence", 0) for i in ideas) / len(ideas)
        ps = min(1.0, len(pairs) / 3.0) if pairs else 0.0
        return min(1.0, am * 0.25 + ar * 0.25 + dd * 0.20 + ac * 0.15 + ps * 0.15)

    def _update_maturity(self):
        now = time.time()
        for i in self._incubation:
            elapsed = now - i["incubation_time"]
            i["maturity"] = min(1.0, min(1.0, elapsed / 30.0) * 0.5 + i.get("resonance", 0) * 0.3 + i.get("access_count", 0) * 0.05 + i["coherence"] * 0.2)

    def check_burst(self):
        if len(self._incubation) < 2:
            return None
        self._update_maturity()
        hrp = []
        for i, a in enumerate(self._incubation):
            for b in self._incubation[i+1:]:
                if (a.get("resonance", 0) + b.get("resonance", 0)) / 2 > 0.3:
                    hrp.append((a, b, (a.get("resonance", 0) + b.get("resonance", 0)) / 2))
        rt = len(hrp) > 0 and random.random() < 0.4
        rr = random.random() < self._burst_probability
        if not rt and not rr:
            return None
        si = sorted(self._incubation, key=lambda x: x.get("maturity", 0), reverse=True)
        mi = [i for i in si if i.get("maturity", 0) > 0.3]
        if len(mi) < 2:
            mi = si[:min(3, len(si))]
        top3 = mi[:3]
        best = top3[0]
        q = self._compute_quality(top3, hrp)
        ci = " + ".join(i["idea"] for i in top3)
        tl = "🔮共振" if rt else "⚡随机"
        qb = "█" * int(q * 5) + "░" * (5 - int(q * 5))
        ads = set()
        for idea in top3:
            ads.update(idea.get("domains", []))
        dt = ",".join(sorted(ads)[:4])
        templates = [f"💡[{tl}][{qb}][{dt}] 灵感涌现：{ci}",
                     f"✨ 跨域灵感(质量{q:.1%})：{ci} ← 来自{dt}的交汇",
                     f"🧠 潜意识洞见：{ci} | 强度{q:.2f} | 源自{tl}"]
        insight = {"content": random.choice(templates), "source_ideas": [i["idea"] for i in top3],
                   "coherence": min(1.0, best["coherence"] + q * 0.3), "quality": q,
                   "trigger": "resonance" if rt else "random", "domains": list(ads), "timestamp": time.time()}
        self._insights.append(insight)
        for idea in top3:
            if idea in self._incubation:
                self._incubation.remove(idea)
        return insight


class SubconsciousEngine:
    def __init__(self, interval=5.0, emotional_engine=None):
        self.interval = interval
        self._emotional_engine = emotional_engine
        self.free_assoc = FreeAssociation()
        self.emotion_driven = EmotionDriven()
        self.cross_domain = CrossDomainAssociation()
        self.dream = DreamIntegration()
        self.insight_burst = InsightBurst()
        self._intuitions: List[Intuition] = []
        self._max_intuitions = 50
        self._lock = threading.Lock()
        self._seed_queue: deque = deque(maxlen=20)
        self._thread = None
        self._running = False
        self._pattern_memory: Dict[str, float] = {}
        self._stats = {"total_generated": 0, "by_source": {"free_assoc": 0, "emotion": 0, "cross_domain": 0, "dream": 0, "insight": 0}, "dream_integrations": 0, "insight_bursts": 0}
        self._meta = {"cycle_count": 0, "avg_coherence": 0.0, "diversity_score": 0.0}
        self._source_weights = {"free_assoc": 1.0, "emotion": 1.0, "cross_domain": 1.0, "dream": 1.0, "insight": 1.0}
        self._coherence_history: deque = deque(maxlen=20)
        self._source_quality: Dict[str, List[float]] = defaultdict(list)
        logger.info("SubconsciousEngine v3 initialized (evolution-40)")

    def feed(self, text, topics=None, emotion=None):
        with self._lock:
            words = self._extract_seeds(text)
            self._seed_queue.append({"text": text[:200], "words": words, "topics": topics or ["general"], "emotion": emotion, "timestamp": time.time()})
            self.free_assoc.expand_concepts(words)
            self.dream.add_memory(text, topics)
            self.insight_burst.incubate(text[:80], topics or ["general"], coherence=0.4)

    def get_intuitions(self, top_k=3, min_coherence=0.1, consume=True):
        with self._lock:
            available = [i for i in self._intuitions if not i.activated and i.coherence >= min_coherence]
            available.sort(key=lambda x: (-x.quality * x.coherence, -x.timestamp))
            results = available[:top_k]
            if consume:
                for r in results:
                    r.activated = True
            return results

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="subconscious-v3")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def is_running(self):
        return self._running

    def _loop(self):
        while self._running:
            try:
                self._generate_cycle()
            except Exception as e:
                logger.debug(f"Subconscious cycle error: {e}")
            time.sleep(self.interval)

    def _adapt_strategy(self):
        with self._lock:
            recent = self._intuitions[-20:] if len(self._intuitions) >= 20 else self._intuitions
        if not recent:
            return
        sq = defaultdict(list)
        for i in recent:
            sq[i.source].append(i.coherence)
        for s, c in sq.items():
            avg = sum(c) / len(c)
            self._pattern_memory[s] = self._pattern_memory.get(s, 0.5) * 0.7 + avg * 0.3
            self._source_quality[s].append(avg)
            if len(self._source_quality[s]) > 20:
                self._source_quality[s] = self._source_quality[s][-20:]
        for s in self._source_weights:
            q = self._pattern_memory.get(s, 0.5)
            if q > 0.5:
                self._source_weights[s] = min(2.0, self._source_weights[s] + 0.05)
            elif q < 0.35:
                self._source_weights[s] = max(0.5, self._source_weights[s] - 0.05)
        self._meta["cycle_count"] += 1
        if sq:
            self._meta["avg_coherence"] = sum(sum(c) / len(c) for c in sq.values()) / len(sq)
        self._meta["diversity_score"] = len(sq) / 5.0

    def _generate_cycle(self):
        if self._meta["cycle_count"] % 10 == 0:
            self._adapt_strategy()
        seeds, topics, emotion = [], [], None
        with self._lock:
            if self._seed_queue:
                latest = self._seed_queue[-1]
                seeds, topics, emotion = latest["words"], latest["topics"], latest.get("emotion")
        if self._meta["cycle_count"] % 5 == 0:
            self.free_assoc._decay_concepts()

        result = self.free_assoc.associate(seeds)
        if result:
            self._add_intuition(result[0], "free_assoc", result[1], "neutral", topics)

        effective_emotion, effective_intensity = emotion, 0.5
        if self._emotional_engine:
            try:
                state = self._emotional_engine.get_state()
                effective_emotion = state.get("dominant_emotion", emotion or "calm")
                effective_intensity = state.get("intensity", 0.5)
            except:
                pass
        elif seeds:
            seed_text = " ".join(seeds[:5])
            effective_emotion, effective_intensity = self.emotion_driven.infer_emotion(seed_text)

        if effective_emotion and random.random() < 0.7:
            result = self.emotion_driven.generate_with_emotion(effective_emotion, effective_intensity, self.free_assoc.concepts)
            if result:
                self._add_intuition(result[0], "emotion", result[1], result[2], topics)

        cross_prob = 0.65 * self._source_weights.get("cross_domain", 1.0)
        if random.random() < min(0.9, cross_prob) and topics:
            domain = topics[0] if topics else "技术"
            result = self.cross_domain.associate(domain, concepts=self.free_assoc.concepts)
            if result:
                self._add_intuition(result[0], "cross_domain", result[1], "探索", [domain, result[2]])
            if random.random() < 0.3 and len(topics) > 1:
                result2 = self.cross_domain.associate(topics[1], concepts=self.free_assoc.concepts)
                if result2:
                    self._add_intuition(result2[0], "cross_domain", result2[1], "探索", [topics[1], result2[2]])

        dream_prob = 0.25 * self._source_weights.get("dream", 1.0)
        dream_trigger = random.random() < min(0.5, dream_prob)
        if len(self.dream._pending_memories) > 6:
            dream_trigger = True
        if dream_trigger:
            fragment = self.dream.integrate()
            if fragment:
                self._add_intuition(fragment.content, "dream", fragment.integration_score, "整合", [])
                self._stats["dream_integrations"] += 1

        with self._lock:
            for old in [i for i in self._intuitions if not i.activated and time.time() - i.timestamp > 30][:2]:
                self.dream.add_unresolved_intuition(old)
                old.activated = True

        insight = self.insight_burst.check_burst()
        if insight:
            self._add_intuition(insight["content"], "insight", insight["coherence"], "灵感", topics)
            self._stats["insight_bursts"] += 1
        if insight and random.random() < 0.5:
            self.dream.add_memory(insight["content"][:100], topics, 0.5)

        with self._lock:
            li = self._intuitions[-3:]
        for intuition in li:
            nc = self.free_assoc.extract_new_concepts(intuition.content)
            if nc:
                self.free_assoc.expand_concepts(nc, "learned")

    def _add_intuition(self, content, source, coherence, emotional_tone, seed_topics):
        if coherence < 0.12:
            return
        with self._lock:
            for e in self._intuitions[-10:]:
                if len(set(content) & set(e.content)) > len(set(content)) * 0.6:
                    return
            novelty = 0.5
            if self._intuitions:
                mo = max((len(set(content) & set(i.content)) / max(len(set(content) | set(i.content)), 1) for i in self._intuitions[-10:]), default=0)
                novelty = 1.0 - mo
            sw = self._source_weights.get(source, 1.0)
            quality = (coherence * 0.5 + novelty * 0.3 + sw * 0.1) * 0.8
            self._intuitions.append(Intuition(content=content, source=source, coherence=coherence, emotional_tone=emotional_tone, novelty=novelty, timestamp=time.time(), seed_topics=seed_topics, quality=quality))
            if len(self._intuitions) > self._max_intuitions:
                self._intuitions.sort(key=lambda x: x.quality, reverse=True)
                kept, sc = [], defaultdict(int)
                for intuition in self._intuitions:
                    s = intuition.source
                    if s in {"dream", "insight"} and sc[s] < 3:
                        kept.append(intuition); sc[s] += 1
                    elif len(kept) < self._max_intuitions:
                        kept.append(intuition); sc[s] += 1
                self._intuitions = kept[:self._max_intuitions]
            self._stats["total_generated"] += 1
            self._stats["by_source"][source] = self._stats["by_source"].get(source, 0) + 1
            pk = f"{source}:{emotional_tone}"
            self._pattern_memory[pk] = self._pattern_memory.get(pk, 0) + coherence

    def _extract_seeds(self, text):
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        words = []
        for seg in chinese_chars:
            if len(seg) >= 2:
                for i in range(len(seg)-1):
                    words.append(seg[i:i+2])
            if len(seg) >= 3:
                for i in range(len(seg)-2):
                    words.append(seg[i:i+3])
            if len(seg) >= 1:
                words.append(seg[0])
        seen, result = set(), []
        for w in words:
            w = w.strip()
            if w and len(w) >= 2 and w not in seen:
                seen.add(w); result.append(w)
        return result[:15]

    def get_status(self):
        with self._lock:
            return {"running": self._running, "intuitions_total": len(self._intuitions),
                    "intuitions_unconsumed": sum(1 for i in self._intuitions if not i.activated),
                    "seed_queue_size": len(self._seed_queue), "dream_pending": len(self.dream._pending_memories),
                    "dream_integrated": len(self.dream._integrated), "insight_incubating": len(self.insight_burst._incubation),
                    "insight_bursts": len(self.insight_burst._insights), "stats": self._stats,
                    "concepts_count": {k: len(v) for k, v in self.free_assoc.concepts.items()},
                    "source_weights": dict(self._source_weights), "emergent_patterns": self.free_assoc.get_emergent_patterns(2)[:5]}

    def get_report(self):
        s = self.get_status()
        return "\n".join([
            "=" * 55, "  潜意识引擎 v3 状态 (进化版-40轮变异)", "=" * 55,
            f"  运行状态: {'🟢 运行中' if s['running'] else '🔴 停止'}",
            f"  直觉总数: {s['intuitions_total']}", f"  未消费直觉: {s['intuitions_unconsumed']}",
            f"  种子队列: {s['seed_queue_size']}", "", "  子系统状态:",
            f"    自由联想: {s['stats']['by_source'].get('free_assoc', 0)} 条",
            f"    情感驱动: {s['stats']['by_source'].get('emotion', 0)} 条",
            f"    跨域联想: {s['stats']['by_source'].get('cross_domain', 0)} 条",
            f"    梦境整合: {s['stats'].get('dream_integrations', 0)} 次",
            f"    灵感涌现: {s['stats'].get('insight_bursts', 0)} 次",
            "", f"  源权重: {s['source_weights']}", f"  涌现模式: {s['emergent_patterns']}",
            "", f"  概念库: {s['concepts_count']}", "=" * 55])


_engine = None

def get_subconscious(emotional_engine=None):
    global _engine
    if _engine is None:
        _engine = SubconsciousEngine(emotional_engine=emotional_engine)
    return _engine

def start_subconscious(emotional_engine=None):
    engine = get_subconscious(emotional_engine)
    if not engine.is_running:
        engine.start()
    return engine
