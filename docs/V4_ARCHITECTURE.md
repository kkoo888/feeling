# 融合引擎 V4 架构设计 — 极致推演版

## 5 个方向的前沿技术分析

---

### 方向 2: NER 实体提取

**前沿**: GLiNER v2 (60M, 零样本 NER, GitHub ⭐3k+)
**论文**: GLiNER: Generalist Model for NER using Bidirectional Transformer (arXiv:2311.08526)
**核心公式**: `P(entity|text, label) = σ(fθ(text) · gφ(label))` — 双向编码器计算文本片段与标签的匹配概率

**推演结论**: GLiNER 需要模型下载，不适合纯规则场景。最优方案是**分层提取**:
```
Layer 1: 正则提取 (0ms, 高精度) → file/url/email/ip/time/number
Layer 2: 词典提取 (1ms, 中精度) → person/org/location
Layer 3: GLiNER (50ms, 高召回) → 兜底，零样本识别任意实体
```

**最简实现**: 修好正则接口 + 扩充模式，entity_accuracy 从 0→0.8+

---

### 方向 3: 情感/情绪识别

**前沿**: NTUSD 情感词典 (36k 词)、知网 HowNet、大连理工 DUTIR
**论文**: EmoBERT-Lite (IEEE 2026)、LCEM (CCL 2023)
**核心算法**:
```
score = Σ(word_polarity × negation_flip × degree_weight)
negation_flip: 否定词窗口(3字)内极性翻转
degree_weight: 程度词加权(很=1.5, 非常=2.0, 极其=3.0, 稍微=0.5)
```

**推演结论**: 完整词典 36k 词太大。最优方案是**核心词典 + 否定词 + 程度词**:
```
核心正面词: 200 个高频正面词
核心负面词: 200 个高频负面词
否定词: 30 个 (不/没/无/非/未/别/莫/勿...)
程度词: 40 个 (很/非常/极其/稍微/比较/有点...)
情绪映射: 8 种基本情绪 (开心/悲伤/愤怒/惊讶/好奇/焦虑/感激/平静)
```

**最简实现**: 纯规则，无外部依赖，sentiment_accuracy 从 0→0.7+

---

### 方向 4: LLM Function Calling 兜底

**前沿**: OpenAI Function Calling、Agent 工具调用架构
**论文**: 三段式分类 (美团/复旦 2026)
**核心架构**:
```
Stage 1: 规则匹配 (0ms) — 关键词命中 → 直接返回
Stage 2: LLM 路由 (500ms) — 置信度 < 0.5 时调 LLM function calling
Stage 3: 兜底 (0ms) — LLM 也失败 → 默认意图
```

**推演结论**: Function Calling 需要定义工具 schema。最优方案:
```python
tools = [
    {"name": "read_file", "parameters": {"file_path": "string"}},
    {"name": "search", "parameters": {"query": "string"}},
    {"name": "run_command", "parameters": {"command": "string"}},
    # ... 20+ 工具
]
# 低置信度时: LLM 从 tools 中选择最匹配的
```

**最简实现**: 置信度 < 0.5 时调 llm_gateway，返回 intent

---

### 方向 5: 推理链压缩 + 信息密度

**前沿**: LongCoT (arXiv:2604.14128)、UID 假说 (ACL 2026)
**论文**: Think Less, Know More (arXiv:2604.09150)、UCoT (ACL 2026)
**核心公式**:
```
信息密度(step_i) = H(step_i | step_{i-1}) = -log P(step_i | step_{i-1})
压缩策略: 保留首尾 + 中间密度最高的 K 步
K = min(5, chain_length)
被压缩步骤: 保留摘要 "[Step 3-7: 通过常识推理和记忆检索确认意图]"
```

**推演结论**: 推理链当前只是标签，需要真正记录每步的依据和密度。最优方案:
```
每步记录: {step_num, description, evidence, density_score}
密度计算: len(useful_info) / len(total_text)
压缩触发: chain_length > 5
压缩策略: 保留第一步 + 最后两步 + 中间密度最高的 (5-3)=2 步
```

**最简实现**: 数据结构 + 压缩函数，chain_quality 从 0→0.8+

---

### 方向 6: Self-Attributing 失败归因

**前沿**: AIDE 进化树 debug 机制、Self-Evolving Agents (ICLR 2026)
**论文**: Darwin Gödel Machine (ICLR 2026 Oral)、Harness Engineering for Self-Improvement
**核心架构**:
```
每个推理步骤记录:
  - module: 哪个模块生成的 (forward/reverse/lateral/fusion/entity/sentiment)
  - confidence: 该步骤的置信度
  - evidence: 该步骤的依据
  - failure_type: 如果失败，失败类型 (no_match/low_confidence/extraction_error)

失败归因:
  1. 找到 confidence 最低的步骤
  2. 定位到具体模块
  3. 生成改进建议
  4. 记录到进化日志
```

**推演结论**: 需要在每个推理步骤埋点。最优方案:
```python
@dataclass
class StepTrace:
    module: str           # "forward" / "entity" / "sentiment"
    confidence: float
    evidence: str
    failure_type: Optional[str]

# 失败时:
def attribute_failure(traces: List[StepTrace]) -> FailureAttribution:
    weakest = min(traces, key=lambda t: t.confidence)
    return FailureAttribution(
        module=weakest.module,
        suggestion=IMPROVEMENT_SUGGESTIONS[weakest.failure_type]
    )
```

**最简实现**: 在现有推理流程中加 trace 记录

---

## 极致架构图

```
输入文本
  │
  ├─→ [方向3] 情感分析 (并行, 0ms)
  │     词典匹配 + 否定词 + 程度词
  │     → SentimentResult {polarity, emotion, confidence}
  │
  ├─→ [方向2] 实体提取 (并行, 0ms)
  │     正则 + 词典
  │     → List[Entity] {text, type, confidence}
  │
  └─→ [IFCoT] 三路径推理 (并行)
        │
        ├─ Forward: 三段式分类
        │   Stage 1: 规则匹配 (0ms)
        │   Stage 2: [方向4] LLM 路由 (if conf < 0.5)
        │   Stage 3: 兜底
        │
        ├─ Reverse: 目标反推
        │
        ├─ Lateral: 类比 + 记忆 + 常识
        │
        └─→ [方向6] StepTrace 记录每步
              │
              ▼
        语义融合 (加权投票 + 一致性加成)
              │
              ├─→ [方向5] 推理链压缩 (>5 步自动压缩)
              │
              ├─→ [方向6] 失败归因 (if conf < threshold)
              │
              ▼
        输出: {intent, confidence, entities, sentiment,
               reasoning_chain, info_density, chain_quality,
               failure_attributions}
```

## 接口设计

```python
@dataclass
class Entity:
    text: str
    entity_type: str  # file/url/time/number/person/org/location
    confidence: float
    start: int = 0
    end: int = 0

@dataclass
class SentimentResult:
    polarity: str       # positive/negative/neutral
    emotion: str        # happy/sad/angry/surprised/curious/anxious/grateful/calm
    confidence: float
    indicators: List[str]  # 命中的情感词

@dataclass
class StepTrace:
    step_num: int
    module: str         # forward/reverse/lateral/entity/sentiment
    description: str
    evidence: str
    confidence: float
    failure_type: Optional[str] = None

@dataclass
class FailureAttribution:
    module: str
    failure_type: str
    details: str
    suggestion: str

@dataclass
class FusionResult:
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
```

## 评估维度

| 维度 | V2 | V3-R8 | V4 目标 | 权重 |
|------|-----|-------|---------|------|
| 准确性 | 1.00 | 1.00 | 1.00 | 0.25 |
| 延迟 | 0.4ms | 0.4ms | <1ms | 0.10 |
| 鲁棒性 | 1.00 | 1.00 | 1.00 | 0.10 |
| 覆盖度 | 1.00 | 1.00 | 1.00 | 0.10 |
| 融合质量 | 0.57 | 0.62 | **0.80** | 0.15 |
| 信息密度 | 0.00 | 0.42 | **0.90** | 0.08 |
| 推理链质量 | 0.00 | 0.76 | **0.90** | 0.07 |
| 置信度校准 | 0.44 | 0.44 | **0.70** | 0.05 |
| 实体提取 | 0.00 | 0.00 | **0.80** | 0.05 |
| 情感识别 | 0.00 | 0.00 | **0.70** | 0.05 |
| **综合** | 0.66 | 0.67 | **0.85** | 1.00 |
