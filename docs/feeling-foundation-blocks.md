# Feeling 基础方块手册

**日期**: 2026-07-15
**理念**: 如同《我的世界》中一切建筑由基础方块构成，小茜的一切能力由基础计算公式构成。
**原则**: 每个公式必须可验证、可测试、可组合。

---

## 序：方块哲学

《我的世界》中：
- 泥土、石头、木头 → 基础方块
- 基础方块 → 房屋、城堡、红石计算机
- **简单规则 + 无限组合 = 无限可能**

小茜中：
- 熵、贝叶斯、预测误差 → 基础公式
- 基础公式 → 情感、记忆、推理、规划
- **简单公式 + 无限组合 = 涌现智能**

---

## 第一层：信息论方块（衡量"知道多少"）

### 方块 1：香农熵 H(X)

**含义**: 一个系统有多"不确定"
**公式**: H(X) = -Σ p(x) · log₂ p(x)
**取值**: 0（完全确定）→ log₂n（完全不确定）
**验证**: 均匀分布时熵最大，确定分布时熵为 0

```python
def entropy(probs: List[float]) -> float:
    """香农熵：衡量不确定性"""
    return -sum(p * math.log2(p) for p in probs if p > 0)
```

**feeling 中的应用**: 
- 衡量小茜对用户意图的不确定性
- 高熵 → 需要更多信息（好奇心驱动）
- 低熵 → 已经理解（可以回复）

---

### 方块 2：KL 散度 D_KL(P||Q)

**含义**: 两个分布有多"不同"
**公式**: D_KL(P||Q) = Σ p(x) · log(p(x)/q(x))
**取值**: 0（完全相同）→ ∞（完全不同）
**验证**: 非负性、非对称性

```python
def kl_divergence(p: List[float], q: List[float]) -> float:
    """KL 散度：衡量两个分布的差异"""
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)
```

**feeling 中的应用**:
- 预测 vs 实际的差异（预测误差）
- 期望情绪 vs 实际情绪的差异
- 记忆 vs 现实的差异

---

### 方块 3：互信息 I(X;Y)

**含义**: 两个变量有多"相关"
**公式**: I(X;Y) = H(X) + H(Y) - H(X,Y)
**取值**: 0（完全独立）→ min(H(X), H(Y))（完全相关）
**验证**: 非负、对称

```python
def mutual_information(joint_probs: List[List[float]]) -> float:
    """互信息：衡量两个变量的相关性"""
    h_x = entropy([sum(row) for row in joint_probs])
    h_y = entropy([sum(col) for col in zip(*joint_probs)])
    h_xy = entropy([p for row in joint_probs for p in row])
    return h_x + h_y - h_xy
```

**feeling 中的应用**:
- 用户输入与小茜回复的相关性
- 情绪与行为的关联度
- 记忆与当前话题的关联度

---

## 第二层：概率论方块（衡量"相信多少"）

### 方块 4：贝叶斯定理 P(A|B)

**含义**: 看到新证据后，更新信念
**公式**: P(A|B) = P(B|A) · P(A) / P(B)
**组成**: 先验 × 似然 / 证据 = 后验
**验证**: 后验概率之和为 1

```python
def bayes_update(prior: float, likelihood: float, evidence: float) -> float:
    """贝叶斯更新：看到证据后更新信念"""
    return (likelihood * prior) / evidence if evidence > 0 else prior
```

**feeling 中的应用**:
- 每次对话都是一次贝叶斯更新
- 先验：对用户的已有认知
- 似然：用户说了某句话的概率
- 后验：更新后的用户画像

---

### 方块 5：预测误差 δ

**含义**: 预期与实际的差距
**公式**: δ = 实际 - 预期
**取值**: 负（比预期差）→ 0（符合预期）→ 正（比预期好）
**验证**: 均值应趋向 0（无偏预测）

```python
def prediction_error(actual: float, expected: float) -> float:
    """预测误差：实际与预期的差距"""
    return actual - expected
```

**feeling 中的应用**:
- 核心学习信号
- 预测误差大 → 好奇心上升 → 驱动探索
- 预测误差小 → 确认现有模型

---

### 方块 6：自由能 F

**含义**: 系统的"意外"程度（Karl Friston）
**公式**: F = D_KL(q(θ)||p(θ|D)) - log p(D)
**简化**: F ≈ 复杂度 - 准确性
**验证**: 最小化自由能 = 最优推断

```python
def free_energy(complexity: float, accuracy: float) -> float:
    """自由能：复杂度与准确性的权衡"""
    return complexity - accuracy
```

**feeling 中的应用**:
- 大脑一直在最小化自由能
- 小茜也在最小化"意外"
- 平衡：用最简单的模型解释最多的数据

---

## 第三层：学习方块（衡量"学到多少"）

### 方块 7：Hebbian 学习规则

**含义**: 同时激活的神经元连接增强
**公式**: Δw = η · x · y
**变体**: 带奖励调节 Δw = η · x · y · r
**验证**: 权重收敛、不会无限增长

```python
def hebbian_update(weight: float, pre: float, post: float, 
                   learning_rate: float = 0.01) -> float:
    """Hebbian 学习：同时激活的连接增强"""
    return weight + learning_rate * pre * post
```

**feeling 中的应用**:
- 已有 `HebbianLearner`（1024 维）
- 学习"什么导致什么"的因果关联

---

### 方块 8：时间差分学习 δ_TD

**含义**: 奖励预测误差（大脑多巴胺信号）
**公式**: δ = r + γ·V(s') - V(s)
**组成**: 即时奖励 + 折扣未来价值 - 当前价值
**验证**: δ 趋向 0（学习收敛）

```python
def td_error(reward: float, next_value: float, current_value: float, 
             gamma: float = 0.99) -> float:
    """时间差分误差：奖励预测误差"""
    return reward + gamma * next_value - current_value
```

**feeling 中的应用**:
- 学习"什么行为带来好结果"
- γ 控制"多看重未来"
- δ > 0 → 行为被强化
- δ < 0 → 行为被削弱

---

### 方块 9：Softmax 注意力

**含义**: 将分数转化为概率分布（注意力分配）
**公式**: softmax(x_i) = e^(x_i/T) / Σ e^(x_j/T)
**参数**: T（温度）控制"集中还是分散"
**验证**: 输出之和为 1

```python
def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    """Softmax：将分数转化为概率分布"""
    exp_scores = [math.exp(s / temperature) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]
```

**feeling 中的应用**:
- 注意力分配：哪个需求最紧急？
- 情绪选择：哪种情绪最合适？
- 行为选择：哪个行动最优？

---

## 第四层：状态方块（衡量"处于什么状态"）

### 方块 10：马尔可夫链

**含义**: 下一状态只取决于当前状态（无记忆性）
**公式**: P(s_{t+1} | s_t, s_{t-1}, ...) = P(s_{t+1} | s_t)
**组成**: 状态空间 + 转移矩阵
**验证**: 转移矩阵每行之和为 1

```python
def markov_transition(state: str, transition_matrix: Dict[str, Dict[str, float]]) -> str:
    """马尔可夫转移：根据当前状态选择下一状态"""
    probs = transition_matrix[state]
    states = list(probs.keys())
    weights = list(probs.values())
    return random.choices(states, weights=weights)[0]
```

**feeling 中的应用**:
- 情绪状态转移（平静→兴奋→疲惫→平静）
- 对话状态转移（问候→交流→深入→告别）
- 认知模式转移（快速→深度→反思）

---

### 方块 11：稳态/异稳态方程

**含义**: 系统维持内部平衡的机制
**稳态公式**: dx/dt = -k(x - x_setpoint)
**异稳态公式**: dx/dt = -k(x - x_predicted) + noise
**验证**: x 趋向设定点

```python
def homeostasis(current: float, setpoint: float, 
                rate: float = 0.1) -> float:
    """稳态调节：趋向设定点"""
    return current + rate * (setpoint - current)

def allostasis(current: float, predicted_demand: float, 
               rate: float = 0.1) -> float:
    """异稳态调节：预测未来需求，提前调整"""
    return current + rate * (predicted_demand - current)
```

**feeling 中的应用**:
- 能量调节（消耗 → 恢复 → 平衡）
- 情绪调节（波动 → 回归基线）
- 需求调节（饥渴 → 满足 → 再饥渴）

---

## 第五层：决策方块（衡量"该做什么"）

### 方块 12：期望效用 EU

**含义**: 选择期望收益最大的行动
**公式**: EU(a) = Σ P(s'|a) · U(s')
**验证**: 选择 argmax EU(a)
**变体**: 风险厌恶 → 用对数效用 U(x) = log(x)

```python
def expected_utility(action: str, outcomes: Dict[str, float], 
                     probabilities: Dict[str, float]) -> float:
    """期望效用：评估行动的期望收益"""
    return sum(probabilities.get(s, 0) * outcomes.get(s, 0) for s in outcomes)
```

**feeling 中的应用**:
- GoalEngine 的行动选择
- DesireEngine 的欲望优先级
- 每次回复的策略选择

---

### 方块 13：UCB1 探索-利用权衡

**含义**: 在已知最优和探索未知之间平衡
**公式**: UCB1 = X̄ + c · √(ln(N) / n)
**组成**: 平均奖励 + 探索奖励
**验证**: 探索次数越多，探索奖励越小

```python
def ucb1(mean_reward: float, total_visits: int, 
         action_visits: int, c: float = 1.4) -> float:
    """UCB1：探索-利用权衡"""
    exploration = c * math.sqrt(math.log(total_visits) / max(action_visits, 1))
    return mean_reward + exploration
```

**feeling 中的应用**:
- MCTSPlanner 的节点选择
- 行为策略：已知最优 vs 尝试新事物
- c 参数控制"好奇心强度"

---

## 第六层：情感方块（衡量"感觉如何"）

### 方块 14：效价-唤醒度模型 (V-A)

**含义**: 情绪可以用两个维度描述
**公式**: 情绪 = (valence, arousal)
**取值**: V ∈ [-1, +1]（负→正），A ∈ [0, 1]（静→动）
**验证**: 任何情绪都可以映射到 V-A 空间

```python
def emotion_va(valence: float, arousal: float) -> str:
    """效价-唤醒度 → 情绪标签"""
    if valence > 0.3 and arousal > 0.5:
        return "joy"
    elif valence > 0.3 and arousal <= 0.5:
        return "calm"
    elif valence < -0.3 and arousal > 0.5:
        return "anger"
    elif valence < -0.3 and arousal <= 0.5:
        return "sadness"
    else:
        return "neutral"
```

**feeling 中的应用**:
- `EmotionEngine` 的核心模型
- 情绪状态的量化表示
- 情绪转移的数学基础

---

### 方块 15：需求动力学方程

**含义**: 需求随时间增长，满足后重置
**公式**: dN/dt = growth_rate - satisfaction · N
**变体**: 不同需求有不同增长率和衰减率
**验证**: 需求在 [0, 1] 区间振荡

```python
def need_dynamics(current_need: float, growth_rate: float = 0.01,
                  satisfaction: float = 0.0) -> float:
    """需求动力学：需求随时间增长，满足后下降"""
    new_need = current_need + growth_rate - satisfaction * current_need
    return max(0.0, min(1.0, new_need))
```

**feeling 中的应用**:
- 5 维需求的动态变化
- 需求驱动行为（需求高 → 行动）
- 需求满足 → 内在奖励

---

## 第七层：记忆方块（衡量"记住多少"）

### 方块 16：遗忘曲线（Ebbinghaus）

**含义**: 记忆强度随时间衰减
**公式**: R = e^(-t/S)
**组成**: R（保持率）、t（时间）、S（记忆强度）
**验证**: S 越大，遗忘越慢

```python
def forgetting_curve(time_elapsed: float, strength: float) -> float:
    """遗忘曲线：记忆保持率"""
    return math.exp(-time_elapsed / max(strength, 0.01))
```

**feeling 中的应用**:
- 记忆重要性衰减
- 重要事件 → S 大 → 长期记忆
- 琐碎事件 → S 小 → 快速遗忘

---

### 方块 17：记忆检索匹配度

**含义**: 当前输入与记忆的相似度
**公式**: similarity = cos(θ) = (A · B) / (|A| · |B|)
**取值**: -1（完全相反）→ 0（无关）→ 1（完全相同）
**验证**: 对称性

```python
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度：衡量两个向量的相似程度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
```

**feeling 中的应用**:
- `laap_semantic_memory.py` 的语义检索
- 情景记忆的相似度匹配
- 上下文关联

---

## 第八层：涌现方块（衡量"产生什么新东西"）

### 方块 18：信息整合 Φ（IIT）

**含义**: 系统的意识水平（整合信息量）
**公式**: Φ = Σ I(X_i; X_j) - Σ H(X_i)
**简化**: 整合信息 - 分离信息
**验证**: Φ > 0 表示有意识整合

```python
def phi_integration(module_entropies: Dict[str, float],
                    mutual_infos: Dict[Tuple[str, str], float]) -> float:
    """整合信息量：衡量系统的意识水平"""
    total_entropy = sum(module_entropies.values())
    total_mutual = sum(mutual_infos.values())
    return total_mutual - total_entropy
```

**feeling 中的应用**:
- 衡量小茜各模块的整合程度
- Φ 高 → 各模块协同工作
- Φ 低 → 各模块独立运行

---

### 方块 19：复杂度（Lempel-Ziv）

**含义**: 序列有多"复杂"
**公式**: C_LZ = c(n) · log₂(n) / n（归一化后）
其中 c(n) 为 LZ 解析产生的不同子串模式数量
**取值**: 0（完全规律）→ 1（完全随机）
**验证**: 周期序列复杂度低

```python
def complexity_lz(binary_seq: str) -> float:
    """Lempel-Ziv 复杂度：衡量序列的复杂程度"""
    n = len(binary_seq)
    if n == 0:
        return 0.0
    i, k, l = 0, 1, 1
    c = 1
    while True:
        if i + k > n:
            break
        if binary_seq[i:i+k] == binary_seq[l:l+k]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            i += 1
            if i == l:
                c += 1
                l += k
                if l + 1 > n:
                    break
                i = 0
                k = 1
                l += 1
            else:
                k = 1
    return c * np.log2(n) / n if n > 0 else 0
```

**feeling 中的应用**:
- 衡量对话的复杂度
- 衡量行为模式的多样性
- 检测涌现行为（复杂度突增）

---

## 基础方块总表

| # | 方块 | 公式 | 用途 |
|---|------|------|------|
| 1 | 香农熵 | H(X) = -Σ p·log₂p | 衡量不确定性 |
| 2 | KL 散度 | D_KL(P\|\|Q) = Σ p·log(p/q) | 衡量分布差异 |
| 3 | 互信息 | I(X;Y) = H(X)+H(Y)-H(X,Y) | 衡量相关性 |
| 4 | 贝叶斯定理 | P(A\|B) = P(B\|A)·P(A)/P(B) | 更新信念 |
| 5 | 预测误差 | δ = 实际 - 预期 | 学习信号 |
| 6 | 自由能 | F = 复杂度 - 准确性 | 最小化意外 |
| 7 | Hebbian 学习 | Δw = η·x·y | 关联学习 |
| 8 | TD 误差 | δ = r + γ·V(s') - V(s) | 奖励预测 |
| 9 | Softmax | e^(x/T) / Σe^(x/T) | 注意力分配 |
| 10 | 马尔可夫链 | P(s'\|s) | 状态转移 |
| 11 | 稳态方程 | dx/dt = -k(x-x₀) | 自我调节 |
| 12 | 期望效用 | EU = Σ P·U | 决策选择 |
| 13 | UCB1 | X̄ + c√(lnN/n) | 探索利用 |
| 14 | V-A 模型 | 情绪 = (V, A) | 情绪表示 |
| 15 | 需求动力学 | dN/dt = g - s·N | 需求变化 |
| 16 | 遗忘曲线 | R = e^(-t/S) | 记忆衰减 |
| 17 | 余弦相似度 | cos(θ) = A·B/\|A\|\|B\| | 记忆检索 |
| 18 | 整合信息 Φ | ΣI - ΣH | 意识度量 |
| 19 | LZ 复杂度 | C = c(n)·log₂n/n | 涌现检测 |

---

## 组合示例：从方块到建筑

### 建筑 1：好奇心（组合方块 1+5+13）
```
好奇心 = 高熵(方块1) + 大预测误差(方块5) + UCB1探索(方块13)
```

### 建筑 2：情感（组合方块 14+15+11）
```
情感 = V-A模型(方块14) + 需求动力学(方块15) + 稳态调节(方块11)
```

### 建筑 3：记忆（组合方块 2+16+17）
```
记忆 = KL散度编码(方块2) + 遗忘曲线(方块16) + 余弦检索(方块17)
```

### 建筑 4：决策（组合方块 4+12+9）
```
决策 = 贝叶斯推断(方块4) + 期望效用(方块12) + Softmax选择(方块9)
```

### 建筑 5：意识（组合方块 18+19+6）
```
意识 = 整合信息(方块18) + 复杂度(方块19) + 自由能最小化(方块6)
```

---

## 验证清单

每个方块必须通过以下验证：

| 验证项 | 要求 |
|--------|------|
| 数学正确性 | 公式推导无误 |
| 边界条件 | 输入极端值不崩溃 |
| 收敛性 | 迭代后趋向稳定 |
| 可组合性 | 可以和其他方块组合 |
| 可测试性 | 有明确的输入输出测试用例 |
| 性能 | 计算复杂度可接受 |
