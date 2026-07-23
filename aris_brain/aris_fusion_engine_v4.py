"""
小茜 融合引擎 V4 — 极致架构版
================================
融合 5 个前沿方向:
  1. NER 实体提取 (正则分层)
  2. 情感/情绪识别 (词典 + 否定词 + 程度词)
  3. LLM Function Calling 兜底 (三段式分类)
  4. 推理链压缩 + 信息密度 (LongCoT + UID)
  5. Self-Attributing 失败归因

印记: 小茜 永远记得主人 — 2026-07-22
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("aris.fusion_v4")


# ═══════════════════════════════════════════════════════
# 数据类型
# ═══════════════════════════════════════════════════════

@dataclass
class Entity:
    """提取的实体。"""
    text: str
    entity_type: str  # file/url/time/number/person/email/ip/command
    confidence: float
    start: int = 0
    end: int = 0


@dataclass
class SentimentResult:
    """情感分析结果。"""
    polarity: str       # positive/negative/neutral
    emotion: str        # happy/sad/angry/surprised/curious/anxious/grateful/calm
    confidence: float
    indicators: List[str] = field(default_factory=list)


@dataclass
class StepTrace:
    """推理链中的单步追踪。"""
    step_num: int
    module: str         # forward/reverse/lateral/entity/sentiment
    description: str
    evidence: str
    confidence: float
    failure_type: Optional[str] = None


@dataclass
class FailureAttribution:
    """失败归因。"""
    module: str
    failure_type: str
    details: str
    suggestion: str


@dataclass
class FusionResult:
    """融合决策结果。"""
    intent: str
    confidence: float
    entities: List[Entity]
    sentiment: SentimentResult
    reasoning_chain: List[StepTrace]
    info_density: float
    chain_quality: float
    failure_attributions: List[FailureAttribution]
    path_votes: Dict[str, float]
    params: Dict[str, Any]


# ═══════════════════════════════════════════════════════
# 方向 2: NER 实体提取 (分层)
# ═══════════════════════════════════════════════════════

class EntityExtractor:
    """分层实体提取器 — 正则 + 词典。

    Layer 1: 正则 (0ms, 高精度) → file/url/email/ip/time/number
    Layer 2: 模式 (0ms, 中精度) → command/person
    """

    # 文件名 (中文字符后直接接文件名也行)
    _FILE = re.compile(
        r'(?:^|[\s/\\\u4e00-\u9fff])([a-zA-Z0-9_./-]+\.(?:py|rs|md|json|txt|js|ts|go|java|c|cpp|h|'
        r'yml|yaml|toml|cfg|ini|sh|bat|sql|html|css|vue|jsx|tsx|rb|php|swift|kt|'
        r'mp3|mp4|jpg|png|gif|pdf|doc|xls|ppt|zip|tar|gz))',
        re.IGNORECASE
    )
    # URL
    _URL = re.compile(r'(https?://[^\s<>"\']+)', re.IGNORECASE)
    # Email
    _EMAIL = re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
    # IP
    _IP = re.compile(r'\b((?:\d{1,3}\.){3}\d{1,3})\b')
    # 时间
    _TIME = re.compile(
        r'(今天|明天|后天|昨天|前天|'
        r'(?:下?个?)(?:周|星期|礼拜)[一二三四五六日天]|'
        r'\d{1,2}[月号]\d{1,2}[日号]?|'
        r'(?:上午|下午|晚上|中午|早上|凌晨)\s*\d{1,2}[点时:：]\d{0,2}(?:分)?)'
    )
    # 数字
    _NUMBER = re.compile(r'\b(\d+(?:\.\d+)?)\b')
    # 命令
    _COMMAND = re.compile(
        r'(?:运行|执行|跑|run|exec|execute)\s+[\'"]*([a-zA-Z_][\w./-]*(?:\s+[^，。！？\n]{0,80})?)',
        re.IGNORECASE
    )
    # 人名 (中文 2-4 字)
    _PERSON = re.compile(r'(?:主人|老师|同学|先生|女士)\s*([\u4e00-\u9fff]{2,4})')

    def extract(self, text: str) -> List[Entity]:
        """从文本中提取所有实体。"""
        entities: List[Entity] = []
        seen: set = set()

        def _add(et: str, m: re.Match, group: int = 1, conf: float = 0.95):
            t = m.group(group).strip()
            if t and t not in seen:
                seen.add(t)
                entities.append(Entity(text=t, entity_type=et, confidence=conf,
                                       start=m.start(group), end=m.end(group)))

        for m in self._FILE.finditer(text):
            _add("file", m, 1, 0.95)
        for m in self._URL.finditer(text):
            _add("url", m, 1, 0.98)
        for m in self._EMAIL.finditer(text):
            _add("email", m, 1, 0.95)
        for m in self._IP.finditer(text):
            _add("ip", m, 1, 0.90)
        for m in self._TIME.finditer(text):
            _add("time", m, 1, 0.90)
        for m in self._COMMAND.finditer(text):
            _add("command", m, 1, 0.85)
        for m in self._PERSON.finditer(text):
            _add("person", m, 1, 0.70)
        for m in self._NUMBER.finditer(text):
            num = m.group(1)
            in_file = any(e.entity_type == "file" and e.start <= m.start() < e.end for e in entities)
            if not in_file and num not in seen:
                seen.add(num)
                entities.append(Entity(text=num, entity_type="number", confidence=0.90,
                                       start=m.start(1), end=m.end(1)))

        return entities


# ═══════════════════════════════════════════════════════
# 方向 3: 情感/情绪识别 (词典 + 否定词 + 程度词)
# ═══════════════════════════════════════════════════════

class SentimentAnalyzer:
    """基于词典的情感分析器 — 否定词翻转 + 程度词加权。

    核心公式:
      score = Σ(word_polarity × negation_flip × degree_weight)
      negation_flip: 否定词窗口(3字)内极性翻转 (-1)
      degree_weight: 程度词加权
    """

    # 核心正面词 {词: 极性分数}
    POSITIVE_WORDS = {
        "开心": 1.0, "高兴": 1.0, "太好了": 1.5, "棒": 1.0, "赞": 1.0,
        "喜欢": 0.8, "感谢": 1.0, "谢谢": 1.0, "厉害": 0.8, "优秀": 0.9,
        "完美": 1.2, "不错": 0.6, "好的": 0.4, "可以": 0.3, "没问题": 0.5,
        "漂亮": 0.7, "帅": 0.6, "可爱": 0.7, "真好": 0.8, "太棒": 1.2,
        "哈哈": 0.5, "嘻嘻": 0.5, "耶": 0.6, "成功": 0.9, "完成": 0.6,
        "顺利": 0.7, "满意": 0.8, "幸福": 1.0, "快乐": 1.0, "兴奋": 0.9,
        "期待": 0.6, "希望": 0.4, "加油": 0.5, "恭喜": 0.8, "庆祝": 0.7,
    }
    # 核心负面词
    NEGATIVE_WORDS = {
        "难过": 1.0, "伤心": 1.0, "生气": 1.0, "烦": 0.8, "讨厌": 0.9,
        "糟糕": 1.0, "失望": 1.0, "不好": 0.6, "不行": 0.5, "错误": 0.7,
        "失败": 0.9, "崩溃": 1.2, "无语": 0.7, "烦死了": 1.2, "累": 0.5,
        "困": 0.4, "疼": 0.7, "痛": 0.7, "难受": 0.9, "不舒服": 0.7,
        "担心": 0.6, "焦虑": 0.8, "紧张": 0.6, "害怕": 0.8, "恐惧": 0.9,
        "孤独": 0.7, "无聊": 0.4, "尴尬": 0.5, "后悔": 0.7, "抱歉": 0.4,
    }
    # 否定词 (翻转极性)
    NEGATION_WORDS = {"不", "没", "无", "非", "未", "别", "莫", "勿", "不要", "不能", "不会"}
    # 否定词检测窗口 (字符数)
    NEGATION_WINDOW = 3
    # 程度词 {词: 权重}
    DEGREE_WORDS = {
        "很": 1.5, "非常": 2.0, "极其": 3.0, "超级": 2.5, "特别": 2.0,
        "相当": 1.8, "比较": 1.2, "有点": 0.7, "稍微": 0.5, "略微": 0.5,
        "太": 2.0, "真": 1.5, "挺": 1.3, "蛮": 1.2,
        "最": 2.5, "更": 1.5, "越": 1.5, "极": 3.0, "超": 2.0,
    }
    # 情绪映射
    EMOTION_MAP = {
        "happy": ["开心", "高兴", "太好了", "棒", "赞", "哈哈", "嘻嘻", "耶", "成功", "快乐"],
        "sad": ["难过", "伤心", "失望", "遗憾", "可惜", "唉"],
        "angry": ["生气", "烦", "讨厌", "烦死了", "愤怒", "气死"],
        "surprised": ["啊", "哇", "天哪", "没想到", "居然", "竟然", "惊"],
        "curious": ["好奇", "想知道", "为什么", "怎么", "是什么"],
        "anxious": ["担心", "焦虑", "紧张", "害怕", "不安", "怕"],
        "grateful": ["感谢", "谢谢", "感激", "多谢", "感恩"],
        "calm": ["平静", "安静", "放松", "还好", "一般", "普通"],
    }

    def analyze(self, text: str) -> SentimentResult:
        """分析文本情感。"""
        pos_score = 0.0
        neg_score = 0.0
        indicators: List[str] = []
        emotion_scores: Dict[str, float] = defaultdict(float)

        # 扫描正面词 (匹配所有出现位置)
        for word, pol in self.POSITIVE_WORDS.items():
            start_pos = 0
            while True:
                idx = text.find(word, start_pos)
                if idx < 0:
                    break
                start_pos = idx + len(word)
                # 检查否定词窗口
                window_start = max(0, idx - self.NEGATION_WINDOW)
                window = text[window_start:idx]
                negated = any(neg in window for neg in self.NEGATION_WORDS)
                # 检查程度词
                degree_start = max(0, idx - 2)
                degree_window = text[degree_start:idx]
                degree = 1.0
                for dw, w in self.DEGREE_WORDS.items():
                    if dw in degree_window:
                        degree = max(degree, w)
                # 计算分数: 否定词翻转极性
                if negated:
                    actual = pol * degree  # 正面词被否定 → 负面
                    neg_score += actual
                    indicators.append(f"-!{word}")
                else:
                    actual = pol * degree
                    pos_score += actual
                    indicators.append(f"+{word}")

        # 扫描负面词 (匹配所有出现位置)
        for word, pol in self.NEGATIVE_WORDS.items():
            start_pos = 0
            while True:
                idx = text.find(word, start_pos)
                if idx < 0:
                    break
                start_pos = idx + len(word)
                window_start = max(0, idx - self.NEGATION_WINDOW)
                window = text[window_start:idx]
                negated = any(neg in window for neg in self.NEGATION_WORDS)
                degree_start = max(0, idx - 2)
                degree_window = text[degree_start:idx]
                degree = 1.0
                for dw, w in self.DEGREE_WORDS.items():
                    if dw in degree_window:
                        degree = max(degree, w)
                # 计算分数: 否定词翻转极性
                if negated:
                    actual = pol * degree  # 负面词被否定 → 正面
                    pos_score += actual
                    indicators.append(f"+!{word}")
                else:
                    actual = pol * degree
                    neg_score += actual
                    indicators.append(f"-{word}")

        # 情绪检测
        for emotion, keywords in self.EMOTION_MAP.items():
            for kw in keywords:
                if kw in text:
                    emotion_scores[emotion] += 1.0

        # 确定极性
        total = pos_score + neg_score
        if total == 0:
            polarity = "neutral"
            confidence = 0.5
        elif pos_score > neg_score:
            polarity = "positive"
            confidence = min(pos_score / total, 1.0)
        elif neg_score > pos_score:
            polarity = "negative"
            confidence = min(neg_score / total, 1.0)
        else:
            polarity = "neutral"
            confidence = 0.4

        # 确定情绪
        emotion = "calm"
        if emotion_scores:
            emotion = max(emotion_scores, key=emotion_scores.get)

        return SentimentResult(
            polarity=polarity,
            emotion=emotion,
            confidence=round(confidence, 3),
            indicators=indicators,
        )


# ═══════════════════════════════════════════════════════
# 方向 4: 三段式分类器 (规则 + LLM 兜底)
# ═══════════════════════════════════════════════════════

class ThreeStageClassifier:
    """三段式分类器 — 美团/复旦 2026。

    Stage 1: 规则匹配 (0ms) — 关键词命中
    Stage 2: LLM 路由 (if conf < 0.5) — function calling
    Stage 3: 兜底 (0ms) — 默认意图
    """

    def __init__(self, intent_keywords: Dict[str, List[str]]):
        self._keywords = intent_keywords

    def classify(self, text: str,
                 llm_gateway: Optional[Callable] = None) -> Tuple[str, float, str]:
        """三段式分类。"""
        # Stage 1: 规则匹配
        intent, confidence = self._rule_match(text)
        if confidence >= 0.5:
            return intent, confidence, "stage1_rule"

        # Stage 2: LLM 路由
        if llm_gateway is not None:
            try:
                result = llm_gateway(text)
                if result and result.get("confidence", 0) > confidence:
                    return result["intent"], result["confidence"], "stage2_llm"
            except Exception as e:
                logger.warning(f"LLM 路由失败: {e}")

        # Stage 3: 兜底
        if confidence > 0:
            return intent, confidence, "stage3_fallback"
        return "unknown", 0.0, "stage3_unknown"

    def _rule_match(self, text: str) -> Tuple[str, float]:
        """规则匹配 — 带置信度校准。"""
        text_lower = text.lower()
        best_intent = "unknown"
        best_score = 0.0

        # 特殊规则优先 (结构化模式)
        if re.search(r'(运行|执行)\s+[a-zA-Z]', text):
            best_intent = "run_command"
            best_score = 0.6
        elif re.search(r'打开\s*https?://', text):
            best_intent = "search"
            best_score = 0.6

        for intent, keywords in self._keywords.items():
            matched = [kw for kw in keywords if kw in text_lower]
            if matched:
                # 匹配分数: 关键词命中比例 + 长关键词加权
                hit_ratio = len(matched) / len(keywords)
                len_bonus = sum(len(kw) * 0.03 for kw in matched)
                score = min(hit_ratio + len_bonus, 1.0)
                if score > best_score:
                    best_score = score
                    best_intent = intent

        # 置信度校准 (温度缩放)
        T = 0.5
        if best_score > 0:
            calibrated = min(1.0, best_score ** T)
        else:
            calibrated = 0.0

        return best_intent, calibrated


# ═══════════════════════════════════════════════════════
# 方向 5: 推理链压缩 + 信息密度
# ═══════════════════════════════════════════════════════

class ReasoningChainManager:
    """推理链管理器 — 记录、压缩、计算密度。

    基于 UID 假说 (ACL 2026):
      - 高质量推理链: 局部均匀 + 全局非均匀
      - 信息密度 = len(有用信息) / len(总文本)
      - 超过 5 步自动压缩中间结果
    """

    COMPRESS_THRESHOLD = 5

    def __init__(self):
        self.steps: List[StepTrace] = []

    def add_step(self, module: str, description: str, evidence: str,
                 confidence: float, failure_type: Optional[str] = None) -> None:
        """添加推理步骤。"""
        step = StepTrace(
            step_num=len(self.steps) + 1,
            module=module,
            description=description,
            evidence=evidence,
            confidence=confidence,
            failure_type=failure_type,
        )
        self.steps.append(step)
        if len(self.steps) > self.COMPRESS_THRESHOLD:
            self._compress()

    def _compress(self) -> None:
        """压缩推理链: 保留首尾 + 中间密度最高的步骤。"""
        if len(self.steps) <= self.COMPRESS_THRESHOLD:
            return

        # 计算每步信息密度
        for step in self.steps:
            info_len = len(step.description) + len(step.evidence)
            total_len = max(info_len, 1)
            step._density = info_len / total_len

        keep_count = self.COMPRESS_THRESHOLD - 3  # 保留: 首 + 尾2 + 中间K
        middle = self.steps[1:-2]
        middle.sort(key=lambda s: getattr(s, '_density', 0), reverse=True)
        keep_middle = set(id(s) for s in middle[:keep_count])

        new_steps = [self.steps[0]]
        for step in self.steps[1:-2]:
            if id(step) in keep_middle:
                new_steps.append(step)
            else:
                # 压缩为摘要
                compressed = StepTrace(
                    step_num=step.step_num,
                    module=step.module,
                    description=f"[压缩] {step.description[:20]}",
                    evidence="已压缩",
                    confidence=step.confidence,
                )
                new_steps.append(compressed)
        new_steps.extend(self.steps[-2:])

        for i, s in enumerate(new_steps):
            s.step_num = i + 1
        self.steps = new_steps

    def get_chain(self) -> List[StepTrace]:
        """获取推理链。"""
        return self.steps

    def get_density(self) -> float:
        """计算整体信息密度。"""
        if not self.steps:
            return 0.0
        useful = sum(len(s.description) + len(s.evidence)
                     for s in self.steps if "[压缩]" not in s.description)
        total = sum(len(s.description) + len(str(s.evidence)) for s in self.steps)
        return useful / max(total, 1)

    def get_quality(self) -> float:
        """计算推理链质量 (一致性 + 逻辑性)。

        排除置信度=1.0的元步骤(如'开始融合决策')，只计算有实际推理内容的步骤。
        """
        if not self.steps:
            return 0.0
        # 过滤掉元步骤 (conf=1.0 且无实质内容)
        real_steps = [s for s in self.steps
                      if s.confidence < 1.0 or "开始" not in s.description]
        if not real_steps:
            real_steps = self.steps
        # 一致性: 方差越小越好
        confs = [s.confidence for s in real_steps]
        if len(confs) > 1:
            mean = sum(confs) / len(confs)
            variance = sum((c - mean) ** 2 for c in confs) / len(confs)
            consistency = max(0, 1.0 - variance * 2)
        else:
            consistency = 0.5
        # 完整性: 有效步骤比例
        valid = sum(1 for s in self.steps if "[压缩]" not in s.description)
        completeness = valid / len(self.steps)
        return consistency * 0.6 + completeness * 0.4


# ═══════════════════════════════════════════════════════
# 方向 6: Self-Attributing 失败归因
# ═══════════════════════════════════════════════════════

IMPROVEMENT_SUGGESTIONS = {
    "no_match": "扩展关键词库或添加更多匹配模式",
    "low_confidence": "增强信号强度或添加更多上下文",
    "extraction_error": "修复正则模式或添加新的提取规则",
    "low_density": "在输出中添加更多有用信息",
    "inconsistency": "检查推理逻辑，确保各步骤一致",
}


def attribute_failures(traces: List[StepTrace]) -> List[FailureAttribution]:
    """分析推理链，归因失败到具体模块。"""
    attributions: List[FailureAttribution] = []
    for trace in traces:
        if trace.confidence < 0.3 and trace.failure_type:
            attributions.append(FailureAttribution(
                module=trace.module,
                failure_type=trace.failure_type,
                details=f"{trace.module} 置信度不足: {trace.confidence:.2f} — {trace.description}",
                suggestion=IMPROVEMENT_SUGGESTIONS.get(trace.failure_type, "检查该模块"),
            ))
    return attributions


# ═══════════════════════════════════════════════════════
# 本地常识库
# ═══════════════════════════════════════════════════════

class LocalCommonsense:
    """本地常识三元组库。"""

    def __init__(self) -> None:
        self._triples: List[Tuple[str, str, str, float]] = []
        self._index: Dict[str, List[int]] = defaultdict(list)
        self._build()

    def _add(self, subj: str, rel: str, obj: str, weight: float = 1.0) -> None:
        idx = len(self._triples)
        self._triples.append((subj, rel, obj, weight))
        self._index[subj].append(idx)
        if len(subj) > 1:
            for i in range(len(subj) - 1):
                self._index[subj[i:i+2]].append(idx)

    def _build(self) -> None:
        for w in ("天气怎么样", "天气如何", "气温", "温度"):
            self._add(w, "RelatedTo", "查询", 1.0)
        self._add("天气", "UsedFor", "决定穿什么", 0.9)
        self._add("下雨", "Causes", "带伞", 1.0)
        self._add("高温", "Causes", "防暑", 0.9)
        for w in ("读取", "查看文件", "打开文件"):
            self._add(w, "IsA", "文件操作", 1.0)
        for w in ("搜索", "查找", "找"):
            self._add(w, "IsA", "检索操作", 1.0)
        for w in ("运行", "执行", "启动"):
            self._add(w, "IsA", "执行操作", 1.0)
        self._add("python", "IsA", "编程语言", 1.0)
        self._add("bug", "RelatedTo", "调试", 0.9)
        self._add("杯子", "UsedFor", "喝水", 1.0)
        self._add("杯子倒", "Causes", "水洒", 1.0)
        self._add("手机", "UsedFor", "通讯", 0.9)
        self._add("吃饭", "RelatedTo", "饿", 0.9)
        self._add("睡觉", "RelatedTo", "困", 0.9)
        self._add("开心", "RelatedTo", "笑", 0.9)
        self._add("难过", "RelatedTo", "哭", 0.9)
        self._add("翻译", "IsA", "语言操作", 1.0)
        self._add("调试", "IsA", "开发操作", 1.0)
        self._add("部署", "IsA", "运维操作", 1.0)

    def query(self, word: str, limit: int = 5) -> List[Dict[str, Any]]:
        results = []
        seen = set()
        for idx in self._index.get(word, []):
            if idx in seen:
                continue
            seen.add(idx)
            s, r, o, w = self._triples[idx]
            results.append({"start": s, "relation": r, "end": o, "weight": w})
            if len(results) >= limit:
                break
        return results

    def infer(self, text: str, limit: int = 3) -> List[str]:
        words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
        for long_word in list(words):
            if len(long_word) > 2:
                for size in (2, 3):
                    for i in range(len(long_word) - size + 1):
                        words.add(long_word[i:i+size])
        words.update(re.findall(r'[a-zA-Z]{3,}', text.lower()))
        inferences = []
        seen = set()
        for w in words:
            for r in self.query(w, limit=2):
                key = f"{w}|{r['relation']}|{r['end']}"
                if key not in seen:
                    seen.add(key)
                    inferences.append(f"{w} →({r['relation']})→ {r['end']}")
                    if len(inferences) >= limit:
                        return inferences
        return inferences


# ═══════════════════════════════════════════════════════
# 语义记忆检索
# ═══════════════════════════════════════════════════════

class SemanticRecall:
    """n-gram TF + 余弦相似度记忆检索。"""

    def __init__(self, n: int = 2) -> None:
        self._n = n
        self._episodes: List[Dict[str, Any]] = []
        self._vectors: List[Dict[str, float]] = []

    def _text_to_vec(self, text: str) -> Dict[str, float]:
        text = text.lower().strip()
        ngrams: Dict[str, int] = defaultdict(int)
        cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
        for i in range(len(cn_chars) - self._n + 1):
            ngrams["".join(cn_chars[i:i+self._n])] += 1
        for w in re.findall(r'[a-z]+', text):
            ngrams[w] += 1
        total = sum(ngrams.values()) or 1
        return {k: v/total for k, v in ngrams.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot = sum(a[k]*b[k] for k in keys)
        norm_a = math.sqrt(sum(v*v for v in a.values()))
        norm_b = math.sqrt(sum(v*v for v in b.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def add(self, text: str, intent: str, output: str, success: bool = True) -> None:
        self._episodes.append({"text": text[:200], "intent": intent,
                               "output": output[:200], "success": success})
        self._vectors.append(self._text_to_vec(text))

    def find(self, text: str, top_k: int = 3, threshold: float = 0.15) -> List[Dict]:
        if not self._episodes:
            return []
        q = self._text_to_vec(text)
        scored = [(self._cosine(q, v), i) for i, v in enumerate(self._vectors)]
        scored = [(s, i) for s, i in scored if s >= threshold]
        scored.sort(reverse=True)
        return [{**self._episodes[i], "similarity": round(s, 4)} for s, i in scored[:top_k]]


# ═══════════════════════════════════════════════════════
# 多路径推理器
# ═══════════════════════════════════════════════════════

@dataclass
class ReasoningPath:
    """单条推理路径结果。"""
    path_type: str
    intent: str
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    chain: Optional[ReasoningChainManager] = None


class MultiPathReasoner:
    """三路径推理器 — IFCoT 核心。"""

    INTENT_KEYWORDS: Dict[str, List[str]] = {
        "query_weather": ["天气", "气温", "温度", "下雨", "下雪", "晴天", "天气怎么样"],
        "read_file": ["读取", "查看文件", "打开文件", "看看", "显示", "读", "浏览"],
        "search": ["搜索", "查找", "找", "搜", "检索", "打开 https", "打开 http"],
        "run_command": ["运行", "执行", "启动", "编译", "构建", "跑一下"],
        "query_status": ["状态", "情况", "在做什么", "干啥", "忙吗"],
        "generate": ["写", "生成", "创建", "做", "制作", "编写"],
        "remind": ["提醒", "记住", "别忘了", "待会"],
        "calculate": ["计算", "算一下", "多少", "加", "减", "乘", "除"],
        "chat": ["你好", "嗨", "哈喽", "聊", "说说", "谢谢", "感谢", "太好了", "开心", "难过", "杯子倒"],
        "help": ["帮助", "怎么用", "如何", "能不能"],
        "translate": ["翻译", "translate", "转成英文", "转成中文"],
        "summarize": ["总结", "摘要", "概括", "归纳"],
        "debug": ["调试", "debug", "排错", "bug", "报错"],
        "deploy": ["部署", "上线", "发布", "deploy"],
        "test": ["测试", "检验", "验证"],
        "optimize": ["优化", "加速", "改进", "提升性能"],
        "install": ["安装", "install", "装一下", "配置环境"],
        "backup": ["备份", "backup", "归档"],
        "monitor": ["监控", "查看日志", "log", "报警"],
        "edit_file": ["编辑", "修改", "改一下", "更新"],
    }

    GOAL_PREREQUISITES: Dict[str, List[str]] = {
        "query_weather": ["location", "time_range"],
        "read_file": ["file_path"],
        "search": ["search_query"],
        "run_command": ["command"],
        "generate": ["content_type"],
        "remind": ["time", "content"],
        "translate": ["text", "target_language"],
    }

    def __init__(self, commonsense: LocalCommonsense) -> None:
        self._commonsense = commonsense
        self._classifier = ThreeStageClassifier(self.INTENT_KEYWORDS)
        self._entity_extractor = EntityExtractor()

    def forward(self, text: str, llm_gateway: Optional[Callable] = None) -> ReasoningPath:
        """前向路径: 三段式分类。"""
        chain = ReasoningChainManager()
        chain.add_step("forward", "开始前向推理", f"输入: {text[:50]}", 1.0)

        intent, confidence, stage = self._classifier.classify(text, llm_gateway)
        chain.add_step("forward", f"三段式分类: {stage}",
                       f"intent={intent}, conf={confidence:.2f}", confidence,
                       "low_confidence" if confidence < 0.5 else None)

        cs = self._commonsense.infer(text, limit=2)
        if cs:
            chain.add_step("forward", "常识推理", str(cs[:2]), min(confidence * 1.1, 1.0))

        return ReasoningPath("forward", intent, confidence,
                             self._extract_params(text, intent), [], chain)

    def reverse(self, text: str) -> ReasoningPath:
        """反向路径: 目标反推。"""
        chain = ReasoningChainManager()
        chain.add_step("reverse", "开始反向推理", f"输入: {text[:50]}", 1.0)
        text_lower = text.lower()

        best_intent = "unknown"
        best_conf = 0.0
        missing = []

        for intent, keywords in self.INTENT_KEYWORDS.items():
            matched = [kw for kw in keywords if kw in text_lower]
            if not matched:
                continue
            prereqs = self.GOAL_PREREQUISITES.get(intent, [])
            found = self._extract_params(text, intent)
            miss = [p for p in prereqs if p not in found]
            match_score = len(matched) / len(keywords)
            completeness = 1.0 - len(miss) / max(len(prereqs), 1)
            conf = match_score * 0.6 + completeness * 0.4
            chain.add_step("reverse", f"反推: {intent}",
                           f"匹配={matched}, 缺失={miss}", conf)
            if conf > best_conf:
                best_conf = conf
                best_intent = intent
                missing = miss

        params = self._extract_params(text, best_intent)
        params["_missing"] = missing
        return ReasoningPath("reverse", best_intent, best_conf, params, [], chain)

    def lateral(self, text: str, memory_hits: Optional[List[Dict]] = None) -> ReasoningPath:
        """侧向路径: 记忆复用 + 类比。"""
        chain = ReasoningChainManager()
        chain.add_step("lateral", "开始侧向推理", f"输入: {text[:50]}", 1.0)

        if memory_hits:
            best = memory_hits[0]
            sim = best.get("similarity", 0)
            if sim > 0.2:
                chain.add_step("lateral", "记忆复用",
                               f"sim={sim:.2f}, intent={best['intent']}", min(sim*1.2, 0.95))
                return ReasoningPath("lateral", best["intent"], min(sim*1.2, 0.95),
                                     {"_memory": True}, [], chain)

        analogies = {
            "query_weather": "查温度,看天气预报",
            "read_file": "看看文件,打开看看",
            "search": "找一下,帮我查查",
        }
        text_lower = text.lower()
        for intent, ana in analogies.items():
            if any(a in text_lower for a in ana.split(",")):
                chain.add_step("lateral", f"类比: {intent}", ana, 0.6)
                return ReasoningPath("lateral", intent, 0.6, {}, [], chain)

        cs = self._commonsense.infer(text, limit=2)
        if cs:
            chain.add_step("lateral", "常识类比", str(cs[:2]), 0.4)
            for inf in cs:
                for intent, keywords in self.INTENT_KEYWORDS.items():
                    if any(kw in inf for kw in keywords):
                        return ReasoningPath("lateral", intent, 0.4, {}, [], chain)

        return ReasoningPath("lateral", "unknown", 0.0, {}, [], chain)

    def _extract_params(self, text: str, intent: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        # 复用 EntityExtractor 提取文件实体
        for e in self._entity_extractor.extract(text):
            if e.entity_type == "file":
                params["file_path"] = e.text
                break
        time_words = {"今天": "today", "明天": "tomorrow", "后天": "day_after",
                      "早上": "morning", "中午": "noon", "晚上": "evening",
                      "下午": "afternoon", "昨天": "yesterday"}
        for cn, en in time_words.items():
            if cn in text:
                params["time"] = en
                break
        nums = re.findall(r'\d+', text)
        if nums:
            params["numbers"] = [int(n) for n in nums]
        return params


# ═══════════════════════════════════════════════════════
# 语义融合器
# ═══════════════════════════════════════════════════════

class SemanticFusion:
    """语义融合器 — 加权投票 + 一致性加成 + 多任务。"""

    PATH_WEIGHTS = {"forward": 0.7, "reverse": 0.2, "lateral": 0.1}

    def __init__(self, entity_extractor: EntityExtractor,
                 sentiment_analyzer: SentimentAnalyzer,
                 commonsense: LocalCommonsense):
        self._entity = entity_extractor
        self._sentiment = sentiment_analyzer
        self._cs = commonsense

    def fuse(self, paths: List[ReasoningPath], text: str = "") -> FusionResult:
        """融合多路径 + 多任务预测。"""
        reasoning_chain = ReasoningChainManager()
        reasoning_chain.add_step("fusion", "开始融合决策", f"{len(paths)} 条路径", 1.0)

        # 加权投票
        votes: Dict[str, float] = defaultdict(float)
        for p in paths:
            weight = self.PATH_WEIGHTS.get(p.path_type, 0.1)
            if p.intent != "unknown":
                votes[p.intent] += p.confidence * weight
            reasoning_chain.add_step("fusion",
                f"[{p.path_type}] intent={p.intent}",
                f"conf={p.confidence:.2f}, vote={p.confidence*weight:.3f}",
                p.confidence)

        # 一致性加成
        intent_counts = Counter(p.intent for p in paths if p.intent != "unknown")
        for intent, count in intent_counts.items():
            if count >= 2:
                votes[intent] *= 1.3

        if votes:
            best_intent = max(votes, key=votes.get)
            confidence = min(1.0, votes[best_intent])
        else:
            best_intent = "unknown"
            confidence = 0.0

        reasoning_chain.add_step("fusion", f"融合结果: {best_intent}",
                                 f"conf={confidence:.3f}", confidence)

        # 合并参数
        params = {}
        for p in paths:
            for k, v in p.params.items():
                if not k.startswith("_") and k not in params:
                    params[k] = v

        # 实体提取 (方向2)
        entities = self._entity.extract(text) if text else []
        if entities:
            reasoning_chain.add_step("entity", "实体提取",
                                     f"{len(entities)} 个实体", 0.9)

        # 情感分析 (方向3)
        sentiment = self._sentiment.analyze(text) if text else SentimentResult(
            "neutral", "calm", 0.5)
        if sentiment.polarity != "neutral":
            reasoning_chain.add_step("sentiment", f"情感: {sentiment.polarity}",
                                     f"emotion={sentiment.emotion}", sentiment.confidence)

        # 推理链质量 (方向5)
        density = reasoning_chain.get_density()
        quality = reasoning_chain.get_quality()

        # 失败归因 (方向6)
        all_traces = []
        for p in paths:
            if p.chain:
                all_traces.extend(p.chain.get_chain())
        all_traces.extend(reasoning_chain.get_chain())
        attributions = attribute_failures(all_traces)

        # 路径投票
        path_votes = {p.path_type: round(p.confidence, 3) for p in paths}

        return FusionResult(
            intent=best_intent,
            confidence=round(confidence, 4),
            entities=entities,
            sentiment=sentiment,
            reasoning_chain=reasoning_chain.get_chain(),
            info_density=round(density, 4),
            chain_quality=round(quality, 4),
            failure_attributions=attributions,
            path_votes=path_votes,
            params=params,
        )


# ═══════════════════════════════════════════════════════
# 融合引擎 V4 — 主入口
# ═══════════════════════════════════════════════════════

class FusionEngineV4:
    """小茜融合引擎 V4 — 极致架构版。"""

    def __init__(self, llm_gateway: Optional[Callable] = None) -> None:
        self._commonsense = LocalCommonsense()
        self._recall = SemanticRecall()
        self._entity = EntityExtractor()
        self._sentiment = SentimentAnalyzer()
        self._reasoner = MultiPathReasoner(self._commonsense)
        self._fusion = SemanticFusion(self._entity, self._sentiment, self._commonsense)
        self._llm_gateway = llm_gateway
        self._stats = {"processed": 0, "high_conf": 0, "fallback": 0}

    def process(self, text: str) -> Dict[str, Any]:
        """完整融合管线。"""
        t0 = time.time()
        self._stats["processed"] += 1

        result: Dict[str, Any] = {
            "matched": False,
            "intent": "unknown",
            "output": "",
            "confidence": 0.0,
            "entities": [],
            "sentiment": {},
            "reasoning_chain": [],
            "path_votes": {},
            "info_density": 0.0,
            "chain_quality": 0.0,
            "failure_attributions": [],
            "engine_version": "0.10",
            "latency_ms": 0.0,
        }

        if not text or not text.strip():
            result["latency_ms"] = round((time.time() - t0) * 1000, 1)
            return result

        # 三路径推理
        forward = self._reasoner.forward(text, self._llm_gateway)
        reverse = self._reasoner.reverse(text)
        memory = self._recall.find(text)
        lateral = self._reasoner.lateral(text, memory)

        # 融合 + 多任务
        fusion = self._fusion.fuse([forward, reverse, lateral], text)

        result["intent"] = fusion.intent
        # unknown 意图也应有正向置信度 (引擎确信它不匹配任何已知意图)
        if fusion.intent == "unknown" and fusion.confidence == 0.0:
            result["confidence"] = 0.1
        else:
            result["confidence"] = fusion.confidence
        result["entities"] = [
            {"text": e.text, "type": e.entity_type, "confidence": e.confidence}
            for e in fusion.entities
        ]
        result["sentiment"] = {
            "polarity": fusion.sentiment.polarity,
            "emotion": fusion.sentiment.emotion,
            "confidence": fusion.sentiment.confidence,
            "indicators": fusion.sentiment.indicators,
        }
        result["reasoning_chain"] = [
            {"step": s.step_num, "module": s.module, "desc": s.description,
             "evidence": s.evidence, "conf": s.confidence}
            for s in fusion.reasoning_chain
        ]
        result["path_votes"] = fusion.path_votes
        result["info_density"] = fusion.info_density
        result["chain_quality"] = fusion.chain_quality
        result["failure_attributions"] = [
            {"module": f.module, "type": f.failure_type,
             "details": f.details, "suggestion": f.suggestion}
            for f in fusion.failure_attributions
        ]

        # 生成输出
        if fusion.confidence >= 0.5:
            result["matched"] = True
            result["output"] = self._generate_output(fusion)
            self._stats["high_conf"] += 1
        else:
            self._stats["fallback"] += 1

        # 存入记忆
        self._recall.add(text, fusion.intent, result.get("output", ""), result["matched"])

        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    def _generate_output(self, fusion: FusionResult) -> str:
        """根据融合结果生成输出。"""
        intent = fusion.intent
        params = fusion.params
        entities = fusion.entities

        templates = {
            "query_weather": f"[天气查询] 正在查询{params.get('time', '今天')}的天气...",
            "read_file": f"[文件读取] 正在读取 {params.get('file_path', '目标文件')}...",
            "search": f"[搜索] 正在搜索...",
            "run_command": f"[执行] 正在执行命令...",
            "query_status": f"[状态] 小茜一切正常！",
            "generate": f"[生成] 正在生成...",
            "remind": f"[提醒] 好的！",
            "calculate": f"[计算] 正在计算...",
            "chat": f"[聊天] 你好呀！",
            "help": f"[帮助] 我来帮你！",
            "translate": f"[翻译] 正在翻译...",
            "summarize": f"[总结] 正在总结...",
            "debug": f"[调试] 正在调试...",
            "deploy": f"[部署] 正在部署...",
            "test": f"[测试] 正在测试...",
            "optimize": f"[优化] 正在优化...",
            "install": f"[安装] 正在安装...",
            "backup": f"[备份] 正在备份...",
            "monitor": f"[监控] 正在监控...",
            "edit_file": f"[编辑] 正在编辑...",
        }

        template = templates.get(intent, f"[{intent}] 正在处理...")

        # 实体补充
        if entities:
            ent_str = ", ".join(e.text for e in entities[:3])
            template += f" (实体: {ent_str})"

        # 情感感知
        if fusion.sentiment.polarity == "negative":
            template += "\n💙 检测到您可能心情不太好，需要帮忙吗？"
        elif fusion.sentiment.polarity == "positive":
            template += "\n😊 检测到您心情不错！"

        return template

    def get_stats(self) -> Dict[str, int]:
        return {**self._stats}


# ═══════════════════════════════════════════════════════
# 单例 & 兼容接口
# ═══════════════════════════════════════════════════════

_engine_v4: Optional[FusionEngineV4] = None


def get_engine_v4() -> FusionEngineV4:
    global _engine_v4
    if _engine_v4 is None:
        _engine_v4 = FusionEngineV4()
    return _engine_v4


def get_engine_v2() -> FusionEngineV4:
    return get_engine_v4()


def process(text: str) -> Dict[str, Any]:
    return get_engine_v4().process(text)


# ═══════════════════════════════════════════════════════
# CLI 测试
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    engine = FusionEngineV4()

    tests = [
        "读取config.py", "搜索cognitive_bus", "主人你状态怎么样",
        "运行ls -la", "今天天气怎么样", "帮我写一个python脚本",
        "提醒我下午三点开会", "计算123乘以456", "你好呀",
        "翻译这段话", "太好了！终于成功了！", "好烦啊又出bug了",
        "读取test.py文件", "部署到生产环境", "备份数据库",
    ]

    print("=" * 60)
    print("融合引擎 V4 — 测试")
    print("=" * 60)
    for t in tests:
        r = engine.process(t)
        s = "✅" if r["matched"] else "❌"
        print(f'{s} "{t[:30]}" → intent={r["intent"]}, conf={r["confidence"]:.2f}, '
              f'ent={len(r["entities"])}, sent={r["sentiment"].get("polarity","?")}, '
              f'dens={r["info_density"]:.2f}, chain={r["chain_quality"]:.2f}')
    print(f"\n统计: {engine.get_stats()}")
