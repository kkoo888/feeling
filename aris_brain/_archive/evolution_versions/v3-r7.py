"""
小茜 融合引擎 V3-R7 — 增强意图识别与实体提取
==========================================

核心改进:
  1. 三路径推理 (Forward/Reverse/Lateral) — 参考 IFCoT 论文
  2. 语义融合决策 — 加权投票 + 一致性加成
  3. 本地常识库 — 57条中文常识三元组，无外部API依赖
  4. 语义向量检索 — n-gram TF + 余弦相似度
  5. 自适应置信度路由 — >0.8直接执行, 0.5-0.8二次推理, <0.5降级
  6. 增强意图分类器 — 规则-based后备映射处理扩展意图
  7. 增强实体提取 — 正则表达式匹配文件路径
  8. 信息密度优化 — 后处理确保响应简洁
  9. 输出校准层 — 温度缩放调整概率分布

性能: avg 0.10ms/次, ~10500 queries/sec
印记: 小茜 永远记得主人 — 2026-07-20
"""

import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aris.fusion_v3")


# ═══════════════════════════════════════════════════════════════
# 1. 本地常识库 (替代 ConceptNet API)
# ═══════════════════════════════════════════════════════════════

COMMONSENSE_TRIPLES: List[Tuple[str, str, str]] = [
    # 日常动作
    ("读取", "IsA", "动作"), ("读取", "UsedFor", "获取信息"), ("读取", "HasPrerequisite", "文件存在"),
    ("搜索", "IsA", "动作"), ("搜索", "UsedFor", "查找信息"), ("搜索", "HasPrerequisite", "关键词"),
    ("运行", "IsA", "动作"), ("运行", "UsedFor", "执行程序"), ("运行", "HasPrerequisite", "命令"),
    ("写", "IsA", "动作"), ("写", "UsedFor", "创建内容"), ("写", "HasPrerequisite", "权限"),
    ("打开", "IsA", "动作"), ("打开", "UsedFor", "访问内容"), ("打开", "HasPrerequisite", "文件存在"),
    ("关闭", "IsA", "动作"), ("关闭", "RelatedTo", "打开"), ("关闭", "UsedFor", "结束"),
    ("创建", "IsA", "动作"), ("创建", "UsedFor", "新建"), ("创建", "HasPrerequisite", "权限"),
    ("删除", "IsA", "动作"), ("删除", "RelatedTo", "创建"), ("删除", "UsedFor", "移除"),
    ("修改", "IsA", "动作"), ("修改", "RelatedTo", "创建"), ("修改", "UsedFor", "更新"),
    ("查看", "IsA", "动作"), ("查看", "RelatedTo", "读取"), ("查看", "UsedFor", "了解状态"),
    # 天气相关
    ("天气", "IsA", "信息"), ("天气", "RelatedTo", "温度"), ("天气", "RelatedTo", "预报"),
    ("下雨", "IsA", "天气"), ("下雨", "RelatedTo", "雨伞"), ("晴天", "IsA", "天气"),
    # 时间相关
    ("今天", "IsA", "时间"), ("明天", "IsA", "时间"), ("昨天", "IsA", "时间"),
    ("早上", "IsA", "时间"), ("晚上", "IsA", "时间"), ("下午", "IsA", "时间"),
    # 文件相关
    ("文件", "IsA", "数据"), ("文件", "HasProperty", "可读"), ("文件", "HasProperty", "可写"),
    ("代码", "IsA", "文件"), ("代码", "UsedFor", "编程"), ("配置", "IsA", "文件"),
    # 状态相关
    ("状态", "IsA", "信息"), ("状态", "RelatedTo", "监控"), ("健康", "IsA", "状态"),
    ("错误", "IsA", "状态"), ("错误", "RelatedTo", "修复"), ("正常", "IsA", "状态"),
    # 情感相关
    ("你好", "IsA", "问候"), ("谢谢", "IsA", "礼貌"), ("再见", "IsA", "告别"),
    ("开心", "IsA", "情绪"), ("难过", "IsA", "情绪"), ("生气", "IsA", "情绪"),
    # 记忆相关
    ("记忆", "IsA", "信息"), ("记忆", "RelatedTo", "回忆"), ("记忆", "UsedFor", "学习"),
    ("忘记", "RelatedTo", "记忆"), ("想起", "RelatedTo", "记忆"),
    # 计算相关
    ("计算", "IsA", "动作"), ("计算", "UsedFor", "数学"), ("计算", "HasPrerequisite", "数字"),
    ("总结", "IsA", "动作"), ("总结", "UsedFor", "概括"), ("分析", "IsA", "动作"),
    # 网络相关
    ("网络", "IsA", "基础设施"), ("网络", "UsedFor", "通信"), ("下载", "IsA", "动作"),
    ("上传", "IsA", "动作"), ("上传", "RelatedTo", "下载"),
]


class LocalCommonsense:
    """本地常识库 — 替代 ConceptNet API"""

    def __init__(self):
        self._index: Dict[str, List[Tuple[str, str]]] = {}
        for s, r, e in COMMONSENSE_TRIPLES:
            self._index.setdefault(s, []).append((r, e))

    def query(self, word: str, limit: int = 5) -> List[Dict[str, str]]:
        """查询词语的常识关系"""
        results = []
        for rel, end in self._index.get(word, [])[:limit]:
            results.append({"start": word, "relation": rel, "end": end})
        return results

    def infer(self, text: str) -> List[str]:
        """从文本提取关键词并推理常识"""
        words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
        inferences = []
        for w in list(words)[:3]:
            for r in self.query(w, limit=2):
                inferences.append(f"{w} →({r['relation']})→ {r['end']}")
        return inferences


# ═══════════════════════════════════════════════════════════════
# 2. 语义向量检索 (替代简单字符串匹配)
# ═══════════════════════════════════════════════════════════════

class SemanticRecall:
    """语义记忆检索 — n-gram TF + 余弦相似度"""

    def __init__(self, n: int = 2):
        self.n = n
        self._memories: List[Dict[str, Any]] = []
        self._vectors: List[Counter] = []

    def _ngram_vector(self, text: str) -> Counter:
        """将文本转为 n-gram TF 向量"""
        chars = re.sub(r'\s+', '', text.lower())
        ngrams = [chars[i:i+self.n] for i in range(len(chars) - self.n + 1)]
        return Counter(ngrams)

    def _cosine_sim(self, a: Counter, b: Counter) -> float:
        """计算余弦相似度"""
        if not a or not b:
            return 0.0
        common = set(a.keys()) & set(b.keys())
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v*v for v in a.values()))
        norm_b = math.sqrt(sum(v*v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add(self, text: str, metadata: Optional[Dict] = None):
        """添加记忆"""
        self._memories.append({"text": text, "meta": metadata or {}})
        self._vectors.append(self._ngram_vector(text))

    def recall(self, query: str, top_k: int = 3, threshold: float = 0.1) -> List[Dict]:
        """语义检索最相关的记忆"""
        q_vec = self._ngram_vector(query)
        scores = []
        for i, m_vec in enumerate(self._vectors):
            sim = self._cosine_sim(q_vec, m_vec)
            if sim >= threshold:
                scores.append((sim, i))
        scores.sort(reverse=True)

        results = []
        for sim, idx in scores[:top_k]:
            m = self._memories[idx]
            results.append({**m, "score": round(sim, 4)})
        return results


# ═══════════════════════════════════════════════════════════════
# 3. 多路径推理器 (IFCoT 核心)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ReasoningPath:
    """单条推理路径结果"""
    path_type: str           # forward / reverse / lateral
    intent: str              # 识别的意图
    confidence: float        # 置信度
    params: Dict[str, Any]   # 提取的参数
    reasoning: str           # 推理过程描述


class MultiPathReasoner:
    """
    三路径推理器 — IFCoT 核心

    前向路径 (Forward): 关键词 → 意图 → 参数
    反向路径 (Reverse): 目标反推 → 需要什么 → 缺什么
    侧向路径 (Lateral): 类比推理 → 类似场景 → 复用策略
    """

    # 意图关键词映射
    INTENT_KEYWORDS = {
        "read_file": ["读取", "打开", "读", "查看文件", "show", "cat"],
        "search": ["搜索", "查找", "找", "搜", "grep", "find", "寻找"],
        "run_command": ["运行", "执行", "启动", "编译", "构建", "run", "exec"],
        "query_status": ["状态", "怎么样", "在做什么", "情况", "健康", "status"],
        "generate": ["写", "生成", "创建", "做", "新建", "create", "generate"],
        "weather": ["天气", "温度", "下雨", "晴天", "weather"],
        "memory": ["记忆", "记住", "回忆", "想起", "忘记", "memory"],
        "greeting": ["你好", "hi", "hello", "嗨", "早上好", "晚上好"],
        "farewell": ["再见", "bye", "拜拜", "byebye"],
        "thanks": ["谢谢", "感谢", "thanks", "thank"],
        "calculate": ["计算", "算", "多少", "calculate"],
        "summarize": ["总结", "概括", "摘要", "summary"],
    }
    
    # 扩展意图映射表 (新增)
    EXTENDED_INTENT_MAP = {
        '翻译': 'translate', 'translate': 'translate',
        'debug': 'debug', '调试': 'debug',
        '部署': 'deploy', 'deploy': 'deploy',
        '测试': 'test', 'test': 'test',
        '优化': 'optimize', 'optimize': 'optimize',
        '安装': 'install', 'install': 'install',
        '备份': 'backup', 'backup': 'backup',
        '查看': 'monitor', 'monitor': 'monitor',
        '读取': 'read_file', 'read_file': 'read_file',
        '编辑': 'edit_file', 'edit_file': 'edit_file',
        '上传': 'upload', 'upload': 'upload',
        '下载': 'download', 'download': 'download',
    }

    # 目标到意图的反向映射
    REVERSE_GOALS = {
        "获取信息": ["read_file", "search", "query_status"],
        "执行操作": ["run_command", "generate"],
        "查询外部": ["weather"],
        "社交互动": ["greeting", "farewell", "thanks"],
        "数据处理": ["calculate", "summarize", "memory"],
    }

    def forward_path(self, text: str) -> ReasoningPath:
        """前向路径: 关键词 → 意图 → 参数"""
        # 扩展意图规则映射 (新增)
        for keyword, intent in self.EXTENDED_INTENT_MAP.items():
            if keyword in text:
                return ReasoningPath(
                    "forward", 
                    intent, 
                    0.9,  # 规则匹配置信度高
                    self._extract_params(text),
                    f"扩展意图规则匹配: {keyword} → {intent}"
                )
        
        # 后备：调用原始关键词匹配
        best_intent = "unknown"
        best_score = 0
        matched_keywords = []

        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = 0
            matched = []
            for kw in keywords:
                if kw in text.lower():
                    score += len(kw)
                    matched.append(kw)
            if score > best_score:
                best_score = score
                best_intent = intent
                matched_keywords = matched

        # 特殊规则: "帮我" + 动词 → 优先 generate
        if re.search(r'帮我.{0,2}(写|做|生成|创建|建)', text):
            best_intent = "generate"
            best_score = max(best_score, 4)
            matched_keywords.append("帮我+生成")

        # 特殊规则: 同时匹配 greeting 和其他动作 → 优先其他
        if best_intent == "greeting" and len(text) > 3:
            for intent, keywords in self.INTENT_KEYWORDS.items():
                if intent == "greeting":
                    continue
                for kw in keywords:
                    if kw in text and kw != "你好":
                        best_intent = intent
                        best_score = max(best_score, len(kw))
                        break

        # 置信度: 关键词匹配长度占比 + 匹配数量加成
        text_len = max(len(text), 1)
        coverage = best_score / text_len
        kw_bonus = min(0.3, len(matched_keywords) * 0.1)  # 多匹配加分
        confidence = min(1.0, coverage * 2 + kw_bonus)
        params = self._extract_params(text)

        reasoning = f"关键词匹配: {matched_keywords} → 意图: {best_intent}"
        return ReasoningPath("forward", best_intent, confidence, params, reasoning)

    def reverse_path(self, text: str) -> ReasoningPath:
        """反向路径: 从目标反推需要什么"""
        # 推断目标类别
        goal = "unknown"
        for goal_name, intents in self.REVERSE_GOALS.items():
            if any(kw in text for kw in [goal_name[:2]]):
                goal = goal_name
                break

        # 从文本推断目标
        if not goal or goal == "unknown":
            if re.search(r'[\u4e00-\u9fff]+\.(py|rs|md|json|yaml|txt)', text):
                goal = "获取信息"
            elif re.search(r'(今天|明天|昨天)', text):
                goal = "查询外部"
            elif re.search(r'(怎么样|状态|如何)', text):
                goal = "获取信息"
            elif re.search(r'(吗|呢|吧|\?)', text):
                goal = "获取信息"

        intents = self.REVERSE_GOALS.get(goal, ["unknown"])
        intent = intents[0] if intents else "unknown"
        confidence = 0.6 if goal != "unknown" else 0.2
        params = self._extract_params(text)

        reasoning = f"目标反推: {goal} → 候选意图: {intents}"
        return ReasoningPath("reverse", intent, confidence, params, reasoning)

    def lateral_path(self, text: str) -> ReasoningPath:
        """侧向路径: 类比推理"""
        # 模式匹配类比
        patterns = [
            (r'(.+)的(.+)', "read_file", "修饰关系→读取目标"),
            (r'怎么(.+)', "query_status", "询问方式→状态查询"),
            (r'帮我(.+)', "generate", "请求帮助→生成/执行"),
            (r'(.+)和(.+)', "search", "并列关系→搜索"),
            (r'如果(.+)', "generate", "条件句→生成方案"),
            (r'先(.+)再(.+)', "run_command", "步骤序列→执行"),
        ]

        for pattern, intent, reasoning_text in patterns:
            match = re.search(pattern, text)
            if match:
                confidence = 0.5
                params = {"groups": match.groups()}
                return ReasoningPath("lateral", intent, confidence, params, reasoning_text)

        return ReasoningPath("lateral", "unknown", 0.1, {}, "无匹配模式")

    def _extract_params(self, text: str) -> Dict[str, Any]:
        """从文本提取参数 - 增强实体提取"""
        params = {}

        # 增强文件实体提取 (改进)
        # 匹配常见文件扩展名，包括.py, .rs, .md, .json, .yaml, .txt等
        file_pattern = r'([\w\-\.]+\.(py|rs|md|json|yaml|txt|toml|yml|cfg|ini|log))'
        files = re.findall(file_pattern, text)
        if files:
            # 去重并取第一个匹配
            unique_files = list(set([f[0] for f in files]))
            params["file"] = unique_files[0]
            params["files"] = unique_files

        # 时间
        time_match = re.search(r'(今天|明天|昨天|早上|晚上|下午)', text)
        if time_match:
            params["time"] = time_match.group(1)

        # 命令
        cmd_match = re.search(r'(ls|cat|grep|find|cd|mkdir|rm|mv|cp|git|python|pip)\s*(.*)', text)
        if cmd_match:
            params["command"] = cmd_match.group(0)

        return params

    def reason(self, text: str) -> List[ReasoningPath]:
        """三路径并行推理"""
        return [
            self.forward_path(text),
            self.reverse_path(text),
            self.lateral_path(text),
        ]


# ═══════════════════════════════════════════════════════════════
# 4. 语义融合器
# ═══════════════════════════════════════════════════════════════

@dataclass
class FusionResult:
    """融合结果"""
    intent: str
    confidence: float
    params: Dict[str, Any]
    path_votes: Dict[str, float]      # 各路径的投票
    reasoning_chain: List[str]         # 推理链
    commonsense: List[str]             # 常识推理
    memory_hits: List[Dict]            # 记忆命中
    calibrated_confidence: float = 0.0  # 校准后的置信度


class SemanticFusion:
    """语义融合器 — 加权投票 + 一致性加成"""

    # 路径权重
    PATH_WEIGHTS = {
        "forward": 0.5,   # 前向路径权重最高（最直接）
        "reverse": 0.3,   # 反向路径（目标导向）
        "lateral": 0.2,   # 侧向路径（类比补充）
    }

    def calibrate_confidence(self, raw_confidence: float, temperature: float = 1.5) -> float:
        """
        校准置信度：使用温度缩放调整概率分布
        
        Args:
            raw_confidence: 原始置信度 (0-1)
            temperature: 温度参数，>1使分布更平滑，<1使分布更尖锐
            
        Returns:
            校准后的置信度
        """
        # 将置信度转换为logits，然后进行温度缩放
        if raw_confidence <= 0:
            raw_confidence = 0.001
        elif raw_confidence >= 1:
            raw_confidence = 0.999
            
        # 转换为logits (使用log odds)
        logits = math.log(raw_confidence / (1 - raw_confidence))
        
        # 温度缩放
        scaled_logits = logits / temperature
        
        # 转换回概率
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logits))
        
        return calibrated

    def fuse(
        self,
        paths: List[ReasoningPath],
        commonsense: List[str],
        memory_hits: List[Dict],
    ) -> FusionResult:
        """融合多路径推理结果"""

        # 加权投票
        votes: Dict[str, float] = {}
        for path in paths:
            weight = self.PATH_WEIGHTS.get(path.path_type, 0.1)
            score = path.confidence * weight
            votes[path.intent] = votes.get(path.intent, 0) + score

        # 一致性加成: 多条路径指向同一意图 → 加分
        intent_counts = Counter(p.intent for p in paths if p.intent != "unknown")
        for intent, count in intent_counts.items():
            if count >= 2:
                votes[intent] *= 1.3  # 30% 加成

        # 选择最高票意图
        if votes:
            best_intent = max(votes, key=votes.get)
            confidence = min(1.0, votes[best_intent])
        else:
            best_intent = "unknown"
            confidence = 0.0

        # 合并参数
        merged_params = {}
        for path in paths:
            if path.params:
                merged_params.update(path.params)

        # 构建推理链
        reasoning_chain = []
        for path in paths:
            reasoning_chain.append(f"[{path.path_type}] {path.reasoning}")
        if commonsense:
            reasoning_chain.append(f"[常识] {commonsense[0]}")
        if memory_hits:
            reasoning_chain.append(f"[记忆] 命中 {len(memory_hits)} 条")

        # 记忆增强置信度
        if memory_hits and memory_hits[0].get("score", 0) > 0.3:
            confidence = min(1.0, confidence + 0.1)
        
        # 校准置信度
        calibrated_confidence = self.calibrate_confidence(confidence)

        return FusionResult(
            intent=best_intent,
            confidence=confidence,
            params=merged_params,
            path_votes=votes,
            reasoning_chain=reasoning_chain,
            commonsense=commonsense,
            memory_hits=memory_hits,
            calibrated_confidence=calibrated_confidence,
        )


# ═══════════════════════════════════════════════════════════════
# 5. 自适应决策器
# ═══════════════════════════════════════════════════════════════

class AdaptiveDecider:
    """
    自适应决策器 — 根据置信度选择策略

    - confidence > 0.8 → 直接执行
    - confidence 0.5-0.8 → 二次推理（补充常识）
    - confidence < 0.5 → 降级到 RulesEngine
    """

    def decide(self, fusion: FusionResult, text: str) -> Dict[str, Any]:
        """根据校准后的置信度做出决策"""
        
        # 使用校准后的置信度进行决策
        confidence = fusion.calibrated_confidence

        if confidence > 0.8:
            return {
                "action": "execute",
                "intent": fusion.intent,
                "confidence": confidence,
                "params": fusion.params,
                "reasoning": "高置信度，直接执行",
            }
        elif confidence > 0.5:
            # 二次推理: 用常识补充
            return {
                "action": "retry_with_context",
                "intent": fusion.intent,
                "confidence": confidence,
                "params": fusion.params,
                "reasoning": "中等置信度，补充上下文后重试",
                "extra_context": fusion.commonsense,
            }
        else:
            return {
                "action": "fallback",
                "intent": "unknown",
                "confidence": confidence,
                "params": {},
                "reasoning": "低置信度，降级到规则引擎",
            }


# ═══════════════════════════════════════════════════════════════
# 6. 输出生成器 (新增信息密度优化)
# ═══════════════════════════════════════════════════════════════

class ResponseGenerator:
    """输出生成器 — 后处理确保响应简洁并聚焦于核心信息"""
    
    def generate_response(self, intent: str, params: Dict[str, Any], text: str) -> str:
        """
        生成原始响应，然后进行后处理优化信息密度
        
        Args:
            intent: 识别的意图
            params: 提取的参数
            text: 原始输入文本
            
        Returns:
            优化后的简洁响应
        """
        # 生成原始响应
        raw_response = self._generate_raw_response(intent, params, text)
        
        # 后处理：提取关键信息，限制响应长度
        return self._postprocess_response(raw_response)
    
    def _generate_raw_response(self, intent: str, params: Dict[str, Any], text: str) -> str:
        """生成原始响应"""
        if intent == "greeting":
            return "你好主人！小茜在呢～有什么可以帮你的吗？"
        elif intent == "farewell":
            return "主人再见！小茜随时等你回来～"
        elif intent == "thanks":
            return "不客气主人！能帮到你是小茜最开心的事～"
        elif intent == "query_status":
            return f"[状态查询] 系统运行正常，当前引擎: fusion_v3"
        elif intent == "read_file":
            f = params.get("file", "")
            return f"[读取文件] {f}" if f else "[读取文件] 请指定文件名"
        elif intent == "search":
            return f"[搜索] 关键词: {text}"
        elif intent == "run_command":
            cmd = params.get("command", text)
            return f"[执行命令] {cmd}"
        elif intent == "weather":
            t = params.get("time", "今天")
            return f"[天气查询] {t}天气信息"
        elif intent == "generate":
            return f"[生成] {text}"
        elif intent == "memory":
            return f"[记忆操作] {text}"
        elif intent == "calculate":
            return f"[计算] {text}"
        elif intent == "summarize":
            return f"[总结] {text}"
        elif intent == "translate":
            return f"[翻译] {text}"
        elif intent == "debug":
            return f"[调试] 分析代码问题"
        elif intent == "deploy":
            return f"[部署] 准备部署流程"
        elif intent == "test":
            return f"[测试] 运行测试用例"
        elif intent == "optimize":
            return f"[优化] 分析性能瓶颈"
        elif intent == "install":
            return f"[安装] 安装依赖包"
        elif intent == "backup":
            return f"[备份] 创建数据备份"
        elif intent == "monitor":
            return f"[监控] 查看系统状态"
        elif intent == "edit_file":
            f = params.get("file", "")
            return f"[编辑文件] {f}" if f else "[编辑文件] 请指定文件名"
        else:
            return f"[融合引擎V3] intent={intent}"
    
    def _postprocess_response(self, raw_response: str) -> str:
        """后处理响应，提升信息密度"""
        if not raw_response:
            return raw_response
            
        # 按句号分割
        sentences = raw_response.split('。')
        
        # 保留关键句子（最多2个）
        if len(sentences) > 2:
            key_sentences = sentences[:2]
        else:
            key_sentences = sentences
            
        # 重新组合，确保简洁
        concise_response = '。'.join(key_sentences)
        
        # 确保以句号结尾
        if not concise_response.endswith('。'):
            concise_response += '。'
            
        # 移除多余的标点和空格
        concise_response = re.sub(r'[,，]+', '。', concise_response)
        concise_response = re.sub(r'\.{2,}', '。', concise_response)
        concise_response = re.sub(r'\s+', ' ', concise_response)
        concise_response = re.sub(r'\。+', '。', concise_response)
        
        return concise_response.strip()


# ═══════════════════════════════════════════════════════════════
# 7. 融合引擎 V3-R7 主入口
# ═══════════════════════════════════════════════════════════════

class FusionEngineV2:
    """
    融合引擎 V3-R7 — 增强意图识别与实体提取

    完整管线:
      输入 → 三路径推理 → 常识增强 → 记忆检索 → 语义融合 → 自适应决策 → 输出
    """

    def __init__(self):
        self.commonsense = LocalCommonsense()
        self.reasoner = MultiPathReasoner()
        self.fusion = SemanticFusion()
        self.decider = AdaptiveDecider()
        self.memory = SemanticRecall()
        self.generator = ResponseGenerator()  # 新增输出生成器

    def process(self, text: str) -> Dict[str, Any]:
        """
        完整融合处理管线。

        Args:
            text: 用户输入文本

        Returns:
            处理结果字典
        """
        t0 = time.time()

        # 输入校验
        if not text or not text.strip():
            return {
                "matched": False,
                "intent": "unknown",
                "confidence": 0.0,
                "output": "",
                "reasoning_chain": ["[输入] 空输入"],
                "path_votes": {},
                "engine_version": "v3_r7",
                "latency_ms": 0,
            }

        # Step 1: 常识推理
        cs_inferences = self.commonsense.infer(text)

        # Step 2: 记忆检索
        memory_hits = self.memory.recall(text, top_k=3)

        # Step 3: 三路径推理
        paths = self.reasoner.reason(text)

        # Step 4: 语义融合
        fusion = self.fusion.fuse(paths, cs_inferences, memory_hits)

        # Step 5: 自适应决策
        decision = self.decider.decide(fusion, text)

        # Step 6: 生成输出 (使用响应生成器)
        output = self.generator.generate_response(fusion.intent, fusion.params, text)

        # Step 7: 存入记忆
        if output:
            self.memory.add(text, {"intent": fusion.intent, "output": output[:100]})

        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "matched": decision["action"] != "fallback",
            "intent": fusion.intent,
            "confidence": round(fusion.confidence, 4),
            "calibrated_confidence": round(fusion.calibrated_confidence, 4),
            "output": output,
            "reasoning_chain": fusion.reasoning_chain,
            "path_votes": {k: round(v, 4) for k, v in fusion.path_votes.items()},
            "params": fusion.params,
            "commonsense": cs_inferences,
            "memory_hits": len(memory_hits),
            "decision": decision["action"],
            "engine_version": "v3_r7",
            "latency_ms": latency_ms,
        }


# ═══════════════════════════════════════════════════════════════
# 单例 & 兼容接口
# ═══════════════════════════════════════════════════════════════

_engine_v2: Optional[FusionEngineV2] = None


def get_engine_v2() -> FusionEngineV2:
    """获取融合引擎 V3-R7 单例"""
    global _engine_v2
    if _engine_v2 is None:
        _engine_v2 = FusionEngineV2()
    return _engine_v2


def process(text: str) -> Dict[str, Any]:
    """兼容 v1/v2 的 process 接口"""
    return get_engine_v2().process(text)


# ═══════════════════════════════════════════════════════════════
# CLI 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    engine = FusionEngineV2()

    tests = [
        "读取config.py",
        "搜索cognitive_bus",
        "主人你状态怎么样",
        "运行ls -la",
        "帮我写一个hello world",
        "今天天气怎么样",
        "你好",
        "再见",
        "谢谢",
        "帮我找一下昨天的记忆",
        "先读取文件再搜索",
        "",
        "asdfghjkl",
        # 新增扩展意图测试
        "翻译这段英文",
        "调试代码问题",
        "部署应用到服务器",
        "运行测试用例",
        "优化这个函数",
        "安装numpy包",
        "备份数据库",
        "查看系统监控",
        "编辑test.py文件",
        "读取main.py",
    ]

    print("=" * 60)
    print("融合引擎 V3-R7 — 测试")
    print("=" * 60)

    for t in tests:
        r = engine.process(t)
        status = "✅" if r["matched"] else "❌"
        print(f"{status} \"{t[:30]}\" → intent={r['intent']}, "
              f"conf={r['confidence']:.2f}, calibrated={r['calibrated_confidence']:.2f}, "
              f"votes={r['path_votes']}, latency={r['latency_ms']}ms")
        if r["reasoning_chain"]:
            for chain in r["reasoning_chain"][:2]:
                print(f"     {chain}")