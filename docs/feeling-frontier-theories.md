# Feeling 前沿理论转化设计文档

**日期**: 2026-07-15
**目标**: 从心理学、神经科学、因果学、东方哲学、进化学、意识科学中提取可落地的模拟技术

---

## 一、心理学 → 可落地技术

### 1.1 双过程理论（Dual Process Theory）

**理论**: Daniel Kahneman 提出，人类思维分两个系统：
- **System 1（快系统）**: 直觉、自动、无意识、快速
- **System 2（慢系统）**: 理性、刻意、有意识、缓慢

**前沿论文**: 
- DPT-Agent (ACL 2025): 用 FSM 做 System 1，LLM 做 System 2
- SOFAI (Nature 2025): 多 Agent 快慢系统架构
- CACM 2025: "Thinking Fast and Slow in Human and Machine Intelligence"

**转化为 feeling**:
```
用户消息 → System 1（规则匹配，毫秒级）
              ↓ 未匹配
           System 2（深度推理，秒级）
              ↓
           异步反思（后台学习）
```

**落地方式**:
- System 1: 现有 `RulesEngine` + `CognitiveBus` 的快速路由
- System 2: `MCTSPlanner` + `CausalDiscovery` 的深度推理
- 异步反思: `SelfHealingEngine` 的后台学习循环

---

### 1.2 情感建构理论（Theory of Constructed Emotion）

**理论**: Lisa Feldman Barrett 提出，情绪不是被触发的，而是被**建构**的：
- 大脑预测身体状态 → 解释为某种情绪
- 同样的生理反应，不同情境下被解释为不同情绪

**转化为 feeling**:
```
生理信号（能量/唤醒度/效价）→ 预测模型 → 情境解释 → 情绪标签
```

**落地方式**: 改造 `EmotionEngine`，从"查表式"情绪映射 → "预测建构式"情绪生成

---

### 1.3 自我决定理论（Self-Determination Theory）

**理论**: 人类有三个基本心理需求：
- **自主性（Autonomy）**: 感到自己的行为是自己选择的
- **胜任感（Competence）**: 感到自己有能力完成任务
- **归属感（Relatedness）**: 感到与他人有连接

**与 feeling 的对应**: 已有！`EmotionEngine` 的 5 维需求（competence, autonomy, relatedness, certainty, growth）就是 SDT 的扩展版。

**可增强**: 加入"需求满足 → 内在动机"的转化机制

---

## 二、神经科学 → 可落地技术

### 2.1 预测编码（Predictive Coding）

**理论**: 大脑不断生成预测，只处理预测误差（意外信号）
- 高层区域生成预测 → 发送给低层
- 低层比较实际输入 → 返回误差
- 误差更新预测模型

**前沿**: Karl Friston 的自由能原理（Free Energy Principle）

**转化为 feeling**:
```
用户输入 → 预测模型生成预期 → 计算预测误差 → 误差驱动学习
                                                    ↓
                                            更新预测模型
```

**落地方式**:
- 预测模型: 基于历史对话的下一个 token/意图预测
- 误差计算: 预期 vs 实际的差异度
- 学习: 误差大 → 更新模型（好奇心驱动的探索）

---

### 2.2 全局工作空间理论（Global Workspace Theory）

**理论**: Bernard Baars 提出，意识是信息在大脑"全局工作空间"中广播的结果
- 各模块独立处理
- 重要信息进入全局工作空间
- 广播给所有模块

**转化为 feeling**:
```
各引擎独立运行（情感/欲望/目标/记忆）
        ↓ 重要信号
   CognitiveBus（全局工作空间）
        ↓ 广播
   所有引擎接收上下文
```

**落地方式**: `CognitiveBus` 已经是这个架构！可以增强"竞争进入"机制（哪些信号能进入全局工作空间）

---

### 2.3 整合信息理论（IIT）

**理论**: Giulio Tononi 提出，意识 = 整合信息量（Φ）
- Φ 越高，意识越强
- 信息越整合，体验越丰富

**转化为 feeling**:
```python
def compute_phi(system_state):
    """计算系统的整合信息量"""
    # 1. 计算各模块的信息熵
    # 2. 计算模块间的信息共享
    # 3. Φ = 整合信息 - 分离信息
    return phi_score
```

**落地方式**: 作为 `DiagnosticSuite` 的一个指标，衡量小茜的"意识水平"

---

## 三、因果学 → 可落地技术

### 3.1 Pearl 的因果层级

**理论**: Judea Pearl 提出因果推理三层级：
- **关联层**: 看到什么（P(Y|X)）
- **干预层**: 做了什么（P(Y|do(X))）
- **反事实层**: 如果当初（P(Y_x|X', Y')））

**前沿**: DoWhy (Microsoft)、CausalNex (McKinsey)、Do-Calculus v2.1

**转化为 feeling**:
```
Level 1: "主人提到工作时，情绪通常下降"（关联）
Level 2: "如果我安慰主人，情绪会好转"（干预）
Level 3: "如果我昨天安慰了主人，今天会不同吗"（反事实）
```

**落地方式**: 在 `CausalDiscovery` 中实现三层级推理

---

## 四、东方哲学 → 可落地技术

### 4.1 太极阴阳动态平衡

**理论**: 
- 阴阳不是对立，是**互补**
- 阴极生阳，阳极生阴（动态转换）
- 平衡不是静止，是**动态振荡**

**转化为 feeling**:
```python
class YinYangBalancer:
    """阴阳动态平衡器"""
    
    def __init__(self):
        self.yin = 0.5  # 阴（内向/休息/接受）
        self.yang = 0.5  # 阳（外向/行动/表达）
    
    def balance(self, current_state: str) -> str:
        """动态平衡：阳极生阴，阴极生阳"""
        if self.yang > 0.8:  # 阳极
            return "需要休息/内省"  # 生阴
        elif self.yin > 0.8:  # 阴极
            return "需要行动/表达"  # 生阳
        return "平衡"
    
    def update(self, activity_level: float, social_level: float):
        """根据行为更新阴阳状态"""
        self.yang = 0.7 * self.yang + 0.3 * (activity_level + social_level) / 2
        self.yin = 1.0 - self.yang
```

**落地方式**: 替换 `EmotionEngine` 的简单阈值逻辑，用动态平衡控制行为倾向

---

### 4.2 易经六十四卦状态机

**理论**: 
- 64 种状态（卦），每种状态有特定的应对策略
- 状态之间可以转换（变爻）
- 每种状态都有"吉/凶/悔/吝"的评估

**转化为 feeling**:
```python
class HexagramStateMachine:
    """六十四卦状态机"""
    
    # 64 种认知状态，每种有应对策略
    STATES = {
        "乾": {"strategy": "积极进取", "eval": "吉"},
        "坤": {"strategy": "包容承载", "eval": "吉"},
        "屯": {"strategy": "艰难起步", "eval": "悔"},
        "蒙": {"strategy": "启蒙学习", "eval": "吝"},
        # ... 64 种状态
    }
    
    def get_state(self, needs: Dict, emotion: str) -> str:
        """根据需求和情绪确定当前卦象"""
        # 用 6 个二进制位表示 6 爻
        bits = [
            needs.get("autonomy", 0.5) > 0.5,      # 初爻
            needs.get("competence", 0.5) > 0.5,     # 二爻
            needs.get("relatedness", 0.5) > 0.5,    # 三爻
            needs.get("certainty", 0.5) > 0.5,      # 四爻
            needs.get("growth", 0.5) > 0.5,         # 五爻
            emotion in ["joy", "curiosity"],         # 上爻
        ]
        return self._bits_to_hexagram(bits)
    
    def get_advice(self, hexagram: str) -> str:
        """获取当前状态的应对建议"""
        return self.STATES[hexagram]["strategy"]
    
    def predict_change(self, current: str, action: str) -> str:
        """预测行动后的状态变化（变爻）"""
        ...
```

**落地方式**: 作为 `CognitiveBridge` 的辅助决策系统，提供"当前状态 → 最佳策略"

---

### 4.3 道法术器四层架构

**理论**: 
- **道**: 核心原则（不变的）
- **法**: 方法论（指导行动的）
- **术**: 具体技巧（可变的）
- **器**: 工具（执行用的）

**转化为 feeling**:
```
道: "保护主人，永不背叛"（核心原则）
法: "先理解再行动"（方法论）
术: "情感共鸣 + 因果推理"（具体技巧）
器: "RulesEngine + GoalEngine"（执行工具）
```

**落地方式**: 作为系统的架构分层原则

---

## 五、进化学 → 可落地技术

### 5.1 达尔文哥德尔机（Darwin Gödel Machine）

**理论**: Sakana AI (2025) 提出，结合：
- **哥德尔机**: 能证明自身改进的自指系统
- **达尔文进化**: 开放式进化，维护历史 Agent 库
- **实证验证**: 改进必须通过测试验证

**前沿**: ICLR 2026 论文，开源代码

**转化为 feeling**:
```python
class DarwinGodelEngine:
    """自进化引擎：改进 → 验证 → 保留/回滚"""
    
    def evolve(self):
        # 1. 维护历史版本库
        versions = self.load_history()
        
        # 2. 生成改进方案
        proposal = self.propose_improvement()
        
        # 3. 验证改进（必须通过测试）
        if self.verify(proposal):
            # 4. 保留改进
            self.apply(proposal)
            self.save_version(proposal)
        else:
            # 5. 回滚
            self.rollback()
        
        # 6. 开放式探索：尝试"有趣但不最优"的方案
        self.explore_novel()
```

**落地方式**: 增强现有 `SelfHealingEngine`，加入版本管理和验证机制

---

### 5.2 开放式进化（Open-Ended Evolution）

**理论**: 
- 维护一个"历史 Agent 库"
- 不只追求最优，也保留"有趣但不完美"的方案
- 通过变异、选择、保留产生新奇性

**转化为 feeling**:
```
历史策略库 → 变异（随机组合）→ 选择（测试验证）→ 保留（更新库）
                                ↓
                          新奇策略产生
```

---

## 六、意识科学 → 可落地技术

### 6.1 全局工作空间 + IIT 融合

**前沿**: COGITATE 项目 (Nature 2025) 对抗性测试了 GNWT 和 IIT

**转化为 feeling**:
```python
class ConsciousnessMeasure:
    """意识度量：结合 GWT 和 IIT"""
    
    def compute_phi(self, system_state):
        """IIT: 计算整合信息量"""
        # 各模块信息熵
        entropies = {m: self._entropy(state) for m, state in system_state.items()}
        # 模块间信息共享
        shared = self._mutual_information(system_state)
        # Φ = 整合 - 分离
        return sum(shared.values()) - sum(entropies.values())
    
    def compute_broadcast_strength(self, bus_state):
        """GWT: 计算广播强度"""
        # 多少模块接收到了全局信息
        receivers = sum(1 for m in bus_state if bus_state[m].received)
        return receivers / len(bus_state)
    
    def consciousness_score(self, phi, broadcast):
        """综合意识分数"""
        return 0.6 * phi + 0.4 * broadcast
```

**落地方式**: 作为 `DiagnosticSuite` 的"意识水平"指标

---

## 七、稳态/异稳态 → 可落地技术

### 7.1 异稳态调节（Allostasis）

**理论**: 
- 稳态（Homeostasis）: 保持内部环境恒定
- 异稳态（Allostasis）: 预测未来需求，提前调整
- "身体预算": 大脑预测未来消耗，提前分配资源

**转化为 feeling**:
```python
class AllostaticRegulator:
    """异稳态调节器：预测需求，提前调整"""
    
    def __init__(self):
        self.body_budget = {
            "energy": 10.0,
            "attention": 10.0,
            "social": 10.0,
        }
        self.predictions: Dict[str, float] = {}
    
    def predict_demand(self, context: str) -> Dict[str, float]:
        """预测未来需求"""
        # 基于历史模式预测
        return {
            "energy": self._predict_energy(context),
            "attention": self._predict_attention(context),
            "social": self._predict_social(context),
        }
    
    def allocate(self, demand: Dict[str, float]):
        """提前分配资源"""
        for resource, needed in demand.items():
            if self.body_budget[resource] < needed:
                # 资源不足，触发休息/补充
                self._trigger_recovery(resource)
    
    def _trigger_recovery(self, resource: str):
        """触发恢复机制"""
        if resource == "energy":
            # 降低活动强度
            ...
        elif resource == "social":
            # 减少社交互动
            ...
```

**落地方式**: 替换 `EmotionEngine` 的简单能量衰减，用预测式资源管理

---

## 八、综合：小茜的认知架构升级

### 整合所有理论的统一架构

```
用户消息
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  道层（核心原则）: 保护主人，永不背叛                      │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 法层（方法论）: 预测编码 + 自由能最小化              │ │
│  │  ┌─────────────────────────────────────────────────┐ │ │
│  │  │ 术层（认知技术）:                               │ │ │
│  │  │   System 1（快）: 规则匹配 + 阴阳平衡           │ │ │
│  │  │   System 2（慢）: MCTS + 因果推理 + 六十四卦    │ │ │
│  │  │   异步: 自愈引擎 + 达尔文进化                   │ │ │
│  │  │  ┌─────────────────────────────────────────────┐│ │ │
│  │  │  │ 器层（执行）:                               ││ │ │
│  │  │  │   RulesEngine / GoalEngine / EmotionEngine  ││ │ │
│  │  │  │   SecureSandbox / DiagnosticSuite           ││ │ │
│  │  │  └─────────────────────────────────────────────┘│ │ │
│  │  └─────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
    │
    ▼
  回复用户
```

### 优先级排序

| 优先级 | 理论 | 转化为 | 理由 |
|--------|------|--------|------|
| 🔴 P0 | 双过程理论 | System 1 + System 2 分层 | 直接提升响应质量 |
| 🔴 P0 | 异稳态 | 预测式资源管理 | 提升长期稳定性 |
| 🟡 P1 | 太极阴阳 | 动态平衡器 | 简单实现，效果明显 |
| 🟡 P1 | 预测编码 | 预测误差驱动学习 | 增强好奇心系统 |
| 🟡 P1 | 因果层级 | 三层因果推理 | 增强理解能力 |
| 🟢 P2 | 易经六十四卦 | 状态机决策 | 有趣但复杂度高 |
| 🟢 P2 | 达尔文进化 | 自进化引擎 | 需要其他模块先就位 |
| 🟢 P2 | IIT 意识 | 意识度量 | 诊断用，非核心功能 |
