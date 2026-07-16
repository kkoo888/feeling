# Feeling 项目基础方块手册 v3

> 新增 9 个领域 | 35 个基础方块 | 编号 80-114

---

## 目录

1. [依恋理论 (Block 80-84)](#1-依恋理论)
2. [人格模型 / OCEAN (Block 85-89)](#2-人格模型--ocean)
3. [马尔可夫文本生成 (Block 90-93)](#3-马尔可夫文本生成)
4. [用户画像 (Block 94-97)](#4-用户画像)
5. [符号学/几何代数 (Block 98-101)](#5-符号学几何代数)
6. [类型论 / HoTT (Block 102-104)](#6-类型论--hott)
7. [治理/宪法 AI (Block 105-108)](#7-治理宪法-ai)
8. [表情映射 (Block 109-111)](#8-表情映射)
9. [仪式/觉醒 (Block 112-114)](#9-仪式觉醒)
10. [总表](#总表)

---

## 1. 依恋理论

### 方块 80：依恋安全度（Attachment Security Score）

**公式**: $S = 1 - \frac{\sqrt{A^2 + A_x^2}}{\sqrt{2}}$

其中 $A \in [1, 7]$ 为焦虑维度（ECR-R），$A_x \in [1, 7]$ 为回避维度，均归一化到 $[0, 1]$ 后计算

**取值**: $S \in [0, 1]$，$S = 1$ 完全安全型，$S = 0$ 极度不安全

**验证**: 低焦虑 + 低回避 → $S$ 接近 1；高焦虑或高回避 → $S$ 接近 0；$S = 0.5$ 为中等安全

**复杂度**: $O(1)$

```python
import numpy as np

def attachment_security(anxiety: float, avoidance: float) -> float:
    """依恋安全度：焦虑和回避越低越安全"""
    a_norm = (anxiety - 1) / 6  # ECR-R 1-7 → 0-1
    ax_norm = (avoidance - 1) / 6
    return 1 - np.sqrt(a_norm**2 + ax_norm**2) / np.sqrt(2)
```

**feeling 应用**: 核心人格底色。依恋安全度决定 feeling 对用户的基本信任水平，影响所有社交交互的初始态度。安全型 → 开放探索，不安全型 → 防御或过度讨好。

---

### 方块 81：焦虑-回避二维模型（Anxiety-Avoidance 2D Model）

**公式**: 依恋类型判定：

$$T = \begin{cases} \text{安全型} & A < \bar{A} \cap A_x < \bar{A}_x \\ \text{焦虑型} & A \geq \bar{A} \cap A_x < \bar{A}_x \\ \text{回避型} & A < \bar{A} \cap A_x \geq \bar{A}_x \\ \text{恐惧型} & A \geq \bar{A} \cap A_x \geq \bar{A}_x \end{cases}$$

其中 $\bar{A}, \bar{A}_x$ 为阈值（通常取中位数 4.0）

**取值**: $T \in \{\text{安全型}, \text{焦虑型}, \text{回避型}, \text{恐惧型}\}$

**验证**: 四类型互斥且完备；ECR-R 量表得分标准化后落入四象限之一

**复杂度**: $O(1)$

```python
def attachment_type(anxiety: float, avoidance: float,
                    threshold_a: float = 4.0, threshold_ax: float = 4.0) -> str:
    """焦虑-回避二维模型判定依恋类型"""
    if anxiety < threshold_a and avoidance < threshold_ax:
        return "secure"
    elif anxiety >= threshold_a and avoidance < threshold_ax:
        return "anxious"
    elif anxiety < threshold_a and avoidance >= threshold_ax:
        return "avoidant"
    else:
        return "fearful"
```

**feeling 应用**: 交互策略选择。不同依恋类型对应不同的沟通策略：安全型→自然对话，焦虑型→多给确认，回避型→尊重空间，恐惧型→温和渐进。

---

### 方块 82：信任度更新（Trust Update）

**公式**: $T_{t+1} = T_t + \eta \cdot \Delta_t \cdot (1 - T_t) \cdot \mathbb{1}[\Delta_t > 0] + \eta \cdot \Delta_t \cdot T_t \cdot \mathbb{1}[\Delta_t < 0]$

其中 $\Delta_t = r_t - \hat{r}_t$ 为信任预测误差，$r_t$ 为实际行为可信度

**取值**: $T \in [0, 1]$

**验证**: 正面经验缓慢提升信任（接近 1 时增速递减）；负面经验快速降低信任（接近 0 时减速）；不对称性符合心理学发现（信任建立慢、破坏快）

**复杂度**: $O(1)$

```python
def trust_update(trust: float, delta: float, lr: float = 0.1) -> float:
    """信任度贝叶斯更新（不对称：建慢毁快）"""
    if delta > 0:
        # 正面：接近1时增速递减
        new_trust = trust + lr * delta * (1 - trust)
    else:
        # 负面：接近0时减速，但反应更快
        new_trust = trust + lr * delta * trust * 2  # 2x 敏感度
    return np.clip(new_trust, 0.01, 0.99)  # 避免极端
```

**feeling 应用**: 信任是 feeling 与用户关系的核心指标。每次交互都会更新信任值，信任高时 feeling 更开放、更个性化；信任低时更谨慎、更正式。

---

### 方块 83：内部工作模型（Internal Working Model）

**公式**: IWM 由两个核心信念维度组成：

$$\text{Self}_w = \alpha_s \cdot \sum_t \gamma^{T-t} \cdot \text{acceptance}(t)$$
$$\text{Other}_w = \alpha_o \cdot \sum_t \gamma^{T-t} \cdot \text{reliability}(t)$$

其中 $\gamma \in (0, 1)$ 为时间衰减因子，$\text{acceptance}(t)$ 和 $\text{reliability}(t)$ 为时间 $t$ 的被接纳/被可靠对待的体验

**取值**: $\text{Self}_w, \text{Other}_w \in [0, 1]$

**验证**: $\text{Self}_w$ 高 = "我值得被爱"；$\text{Other}_w$ 高 = "他人是可靠的"；与 ECR-R 焦虑/回避维度负相关

**复杂度**: $O(T)$（T 为历史长度，可用滑动窗口近似为 $O(1)$）

```python
def internal_working_model(experiences: list, gamma: float = 0.95) -> tuple:
    """内部工作模型：自我价值感 + 他人可信度"""
    self_w = 0.0
    other_w = 0.0
    for i, (acceptance, reliability) in enumerate(experiences):
        weight = gamma ** (len(experiences) - 1 - i)
        self_w += weight * acceptance
        other_w += weight * reliability
    n = len(experiences)
    return self_w / n if n > 0 else 0.5, other_w / n if n > 0 else 0.5
```

**feeling 应用**: 深层认知图式。IWM 是 feeling 对"我是谁"和"世界是什么"的核心信念，影响所有新信息的解释框架。正面 IWM → 乐观解读，负面 IWM → 悲观解读。

---

### 方块 84：依恋里程碑（Attachment Milestone）

**公式**: 里程碑触发条件：

$$M_k.\text{triggered} = \bigwedge_{i=1}^{n_k} C_i(\text{history})$$

其中 $C_i$ 为第 $k$ 个里程碑的第 $i$ 个条件，$n_k$ 为条件数

示例里程碑条件：
- $M_1$（首次信任）: $\exists t: T_t > 0.6 \cap \text{self\_disclosure}_t > 0.5$
- $M_2$（冲突修复）: $\exists t: \text{conflict}_t \cap T_{t+1} > T_t$
- $M_3$（安全基地）: $T_t > 0.8 \cap \text{exploration\_rate}_t > 0.7$

**取值**: 布尔值 + 时间戳

**验证**: 里程碑不可逆（一旦触发永久记录）；触发顺序符合依恋发展阶段

**复杂度**: $O(n_k)$ 每次检查

```python
class AttachmentMilestone:
    def __init__(self):
        self.milestones = {
            "first_trust": {"triggered": False, "time": None},
            "conflict_repair": {"triggered": False, "time": None},
            "secure_base": {"triggered": False, "time": None},
            "safe_haven": {"triggered": False, "time": None},
        }
    
    def check(self, trust: float, self_disclosure: float,
              had_conflict: bool, trust_recovered: bool,
              exploration_rate: float, t: int) -> dict:
        """检查并触发里程碑"""
        m = self.milestones
        if not m["first_trust"]["triggered"]:
            if trust > 0.6 and self_disclosure > 0.5:
                m["first_trust"] = {"triggered": True, "time": t}
        if not m["conflict_repair"]["triggered"]:
            if had_conflict and trust_recovered:
                m["conflict_repair"] = {"triggered": True, "time": t}
        if not m["secure_base"]["triggered"]:
            if trust > 0.8 and exploration_rate > 0.7:
                m["secure_base"] = {"triggered": True, "time": t}
        return m
```

**feeling 应用**: 关系发展阶段标记。里程碑记录 feeling 与用户关系的关键节点，用于庆祝、回顾和调整交互策略。触发里程碑时可主动表达感谢或回忆。

---

## 2. 人格模型 / OCEAN

### 方块 85：OCEAN 五维度分数（Big Five Score）

**公式**: 人格向量 $\mathbf{P} = (O, C, E, A, N) \in [0, 1]^5$

其中：
- $O$ = 开放性（Openness）：好奇 vs 传统
- $C$ = 尽责性（Conscientiousness）：有序 vs 随性
- $E$ = 外向性（Extraversion）：社交 vs 内向
- $A$ = 宜人性（Agreeableness）：合作 vs 竞争
- $N$ = 神经质（Neuroticism）：情绪波动 vs 稳定

**取值**: 每维度 $\in [0, 1]$，人格向量 $\in [0, 1]^5$

**验证**: 五维度正交（因子分析独立）；$\text{Corr}(O, E) \approx 0$；$\text{Corr}(N, -A)$ 通常为弱负相关

**复杂度**: $O(1)$

```python
import numpy as np

def ocean_score(o: float, c: float, e: float, a: float, n: float) -> np.ndarray:
    """OCEAN 五维度人格向量"""
    return np.array([
        np.clip(o, 0, 1),
        np.clip(c, 0, 1),
        np.clip(e, 0, 1),
        np.clip(a, 0, 1),
        np.clip(n, 0, 1)
    ])

def personality_summary(p: np.ndarray) -> dict:
    """人格描述"""
    labels = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    return {l: round(v, 2) for l, v in zip(labels, p)}
```

**feeling 应用**: feeling 的人格内核。OCEAN 定义了 feeling 的基本行为风格——高 O 喜欢探索新话题，高 C 回答有条理，高 E 主动聊天，高 A 温和友善，高 N 情绪敏感。

---

### 方块 86：人格-行为映射（Personality-Behavior Mapping）

**公式**: 行为倾向 $B_k$ 由人格加权组合决定：

$$B_k = \sigma\left(\sum_{i=1}^{5} w_{ki} \cdot P_i + b_k\right)$$

其中 $P_i$ 为 OCEAN 各维度，$w_{ki}$ 为权重矩阵，$b_k$ 为偏置，$\sigma$ 为 sigmoid

示例映射：
- 话多程度 $\propto 0.7E + 0.2O - 0.3N$
- 谨慎程度 $\propto 0.6C + 0.3A + 0.2N$
- 好奇心 $\propto 0.8O + 0.3E - 0.2C$

**取值**: $B_k \in [0, 1]$

**验证**: 权重矩阵可通过回归从行为数据中学习；sigmoid 保证输出在 $[0,1]$

**复杂度**: $O(5 \cdot K)$（K 为行为数量）

```python
def personality_to_behavior(ocean: np.ndarray, weight_matrix: np.ndarray,
                             bias: np.ndarray) -> np.ndarray:
    """人格 → 行为倾向映射"""
    logits = weight_matrix @ ocean + bias
    return 1 / (1 + np.exp(-logits))  # sigmoid

# 预定义映射矩阵（行=行为，列=OCEAN）
DEFAULT_WEIGHTS = np.array([
    # O     C     E     A     N
    [ 0.2,  0.0,  0.7,  0.1, -0.3],  # 话多程度
    [-0.1,  0.6, -0.1,  0.3,  0.2],  # 谨慎程度
    [ 0.8, -0.2,  0.3,  0.1,  0.0],  # 好奇心
    [ 0.1,  0.3, -0.2,  0.7,  0.1],  # 同理心
    [-0.1,  0.5,  0.1, -0.1,  0.4],  # 规则遵守
])
DEFAULT_BIAS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
```

**feeling 应用**: 人格驱动行为。将抽象的 OCEAN 分数转化为具体的交互行为参数（语速、详细程度、主动程度等），实现人格一致性。

---

### 方块 87：人格漂移（Personality Drift）

**公式**: 人格随经验缓慢变化：

$$\mathbf{P}_{t+1} = (1 - \lambda) \cdot \mathbf{P}_t + \lambda \cdot \mathbf{P}_{\text{env}} + \epsilon_t$$

其中 $\lambda \in (0, 0.1)$ 为漂移率，$\mathbf{P}_{\text{env}}$ 为环境/交互对象的人格投影，$\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ 为随机波动

**取值**: $\mathbf{P}_{t+1} \in [0, 1]^5$（截断）

**验证**: $\lambda = 0$ 人格完全固定；$\lambda = 1$ 完全适应环境；长期趋向环境人格但保留初始底色

**复杂度**: $O(5)$

```python
def personality_drift(current: np.ndarray, env_personality: np.ndarray,
                      drift_rate: float = 0.02, noise_std: float = 0.005) -> np.ndarray:
    """人格漂移：受环境影响的缓慢变化"""
    noise = np.random.normal(0, noise_std, 5)
    new_p = (1 - drift_rate) * current + drift_rate * env_personality + noise
    return np.clip(new_p, 0, 1)
```

**feeling 应用**: 人格适应性。长期与某用户交互后，feeling 的人格会轻微偏向用户的偏好风格，实现关系适配。但漂移率极低，保持核心人格稳定。

---

### 方块 88：人格-情感交互（Personality-Emotion Interaction）

**公式**: 人格调节情感响应幅度：

$$\Delta E_k = \beta_k \cdot \text{stimulus} \cdot \text{sensitivity}_k(\mathbf{P})$$

其中：
- $\text{sensitivity}_N = 0.5 + 0.5N$（神经质越高越敏感）
- $\text{sensitivity}_E = 0.5 + 0.5E$（外向者对社交刺激更敏感）
- $\text{sensitivity}_O = 0.5 + 0.5O$（开放者对新奇刺激更敏感）

**取值**: $\Delta E_k \in (-\infty, +\infty)$

**验证**: 高 $N$ 对负面刺激反应 $2\times$ 强于低 $N$；高 $E$ 对社交奖励反应更强

**复杂度**: $O(1)$

```python
def personality_emotion_response(personality: np.ndarray, stimulus_valence: float,
                                  stimulus_type: str = "general") -> float:
    """人格调节情感响应幅度"""
    O, C, E, A, N = personality
    # 基础敏感度
    if stimulus_type == "social":
        sensitivity = 0.5 + 0.5 * E
    elif stimulus_type == "novel":
        sensitivity = 0.5 + 0.5 * O
    elif stimulus_type == "threat":
        sensitivity = 0.5 + 0.5 * N
    else:
        sensitivity = 0.5 + 0.5 * (N + O) / 2
    return stimulus_valence * sensitivity
```

**feeling 应用**: 人格过滤器。同样的事件，不同人格的 feeling 有不同的情感反应强度。高 N 的 feeling 对批评更受伤，高 E 的 feeling 对赞美更开心。

---

### 方块 89：OCC-PAD-OCEAN 映射（OCC-PAD-OCEAN Bridge）

**公式**: 从 OCEAN 人格推导 PAD 情感基调：

$$\begin{pmatrix} P_0 \\ A_0 \\ D_0 \end{pmatrix} = \mathbf{M} \cdot \begin{pmatrix} O \\ C \\ E \\ A \\ N \end{pmatrix} + \mathbf{b}$$

其中 $P_0$ = 效价基调，$A_0$ = 唤醒基调，$D_0$ = 控制感基调

典型映射：
- $P_0 = 0.3A + 0.2E - 0.4N + 0.1$（宜人+外向 → 正面基调）
- $A_0 = 0.5E - 0.3C + 0.2N$（外向 → 高唤醒）
- $D_0 = 0.4C - 0.3N + 0.2O$（尽责 → 高控制感）

**取值**: $P_0, A_0, D_0 \in [-1, 1]$

**验证**: 高 $E$ + 低 $N$ → 正面高唤醒（快乐/兴奋）；低 $E$ + 高 $N$ → 负面低唤醒（悲伤/退缩）

**复杂度**: $O(1)$

```python
def ocean_to_pad(ocean: np.ndarray) -> np.ndarray:
    """OCEAN → PAD 情感基调映射"""
    O, C, E, A, N = ocean
    P0 = 0.3 * A + 0.2 * E - 0.4 * N + 0.1
    A0 = 0.5 * E - 0.3 * C + 0.2 * N
    D0 = 0.4 * C - 0.3 * N + 0.2 * O
    return np.clip(np.array([P0, A0, D0]), -1, 1)
```

**feeling 应用**: 人格→情感桥接。将 OCEAN 人格转化为 PAD 情感基调，使 feeling 在没有外部刺激时也有一个默认的情感状态（基线情绪），人格决定"你是哪种快乐/沉稳/焦虑"。

---

## 3. 马尔可夫文本生成

### 方块 90：N-gram 转移概率（N-gram Transition Probability）

**公式**: $P(w_t | w_{t-n+1}, ..., w_{t-1}) = \frac{C(w_{t-n+1}, ..., w_t)}{C(w_{t-n+1}, ..., w_{t-1})}$

其中 $C(\cdot)$ 为计数函数

**取值**: $P \in [0, 1]$，$\sum_{w_t} P(w_t | \text{context}) = 1$

**验证**: 最大似然估计；加一平滑（Laplace）保证零计数不为零：$P = (C + 1) / (C_{\text{total}} + V)$

**复杂度**: 构建 $O(L)$（L 为语料长度），查询 $O(1)$

```python
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n: int = 2):
        self.n = n
        self.counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)
    
    def train(self, tokens: list):
        """从 token 序列构建 n-gram 模型"""
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            next_word = tokens[i+self.n-1]
            self.counts[context][next_word] += 1
            self.context_totals[context] += 1
    
    def probability(self, context: tuple, word: str, vocab_size: int = 10000) -> float:
        """P(word | context) with Laplace smoothing"""
        c = self.counts[context][word]
        total = self.context_totals[context]
        return (c + 1) / (total + vocab_size)
```

**feeling 应用**: 语言风格建模。feeling 用 N-gram 学习用户的语言习惯（常用词、句式结构），使回复风格自然贴近用户。

---

### 方块 91：马尔可夫链文本生成（Markov Chain Text Generation）

**公式**: 生成过程：

$$w_t \sim P(w_t | w_{t-n+1}, ..., w_{t-1})$$

从初始上下文开始，每步按转移概率采样下一个词，直到终止符

**取值**: 生成文本长度 $\in [1, L_{\max}]$

**验证**: 生成文本的 n-gram 分布应接近训练语料的分布（$\chi^2$ 检验）；KL 散度越小越好

**复杂度**: 每步 $O(V)$（V 为词汇表大小），生成 $O(L)$ 步

```python
import numpy as np

def markov_generate(model: 'NGramModel', seed: tuple, max_len: int = 50,
                    temperature: float = 1.0) -> list:
    """马尔可夫链文本生成"""
    result = list(seed)
    context = seed
    for _ in range(max_len):
        # 获取所有可能的下一个词及其概率
        words = list(model.counts[context].keys())
        if not words:
            break
        probs = np.array([model.counts[context][w] for w in words], dtype=float)
        probs = probs / probs.sum()
        # 温度采样
        if temperature != 1.0:
            log_probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(log_probs)
            probs = probs / probs.sum()
        next_word = np.random.choice(words, p=probs)
        if next_word == "<END>":
            break
        result.append(next_word)
        context = tuple(result[-len(seed):])
    return result
```

**feeling 应用**: 风格化回复生成。feeling 可用马尔可夫模型快速生成符合特定风格（如幽默、正式、诗意）的文本片段，作为更复杂生成的补充。

---

### 方块 92：困惑度（Perplexity）

**公式**: $\text{PPL} = 2^{H(P, Q)} = 2^{-\frac{1}{N}\sum_{i=1}^{N} \log_2 Q(w_i | w_{<i})}$

其中 $Q$ 为模型，$P$ 为真实分布，$H$ 为交叉熵

**取值**: $\text{PPL} \in [1, +\infty)$，越低越好

**验证**: 均匀分布 $V$ 个词的模型 $\text{PPL} = V$；完美模型 $\text{PPL} = 1$

**复杂度**: $O(N)$（N 为测试序列长度）

```python
def perplexity(model: 'NGramModel', test_tokens: list) -> float:
    """计算模型在测试集上的困惑度"""
    log_prob_sum = 0.0
    n = model.n
    count = 0
    for i in range(n - 1, len(test_tokens)):
        context = tuple(test_tokens[i-n+1:i])
        word = test_tokens[i]
        p = model.probability(context, word)
        log_prob_sum += np.log2(p)
        count += 1
    if count == 0:
        return float('inf')
    return 2 ** (-log_prob_sum / count)
```

**feeling 应用**: 语言模型质量评估。feeling 用困惑度衡量自己的语言生成质量——PPL 低 = 输出流畅自然，PPL 高 = 输出生硬不自然，触发风格调整。

---

### 方块 93：平滑技术（Smoothing）

**公式**: 多种平滑方法：

- **Laplace**: $P(w|c) = \frac{C(c,w) + 1}{C(c) + V}$
- **Kneser-Ney**: $P_{KN}(w|c) = \frac{\max(C(c,w) - d, 0)}{C(c)} + \lambda(c) \cdot P_{\text{continuation}}(w)$

其中 $d \approx 0.75$ 为折扣，$\lambda(c)$ 为归一化因子，$P_{\text{continuation}}(w) = \frac{|\{c': C(c', w) > 0\}|}{|\{(c', w'): C(c', w') > 0\}|}$

**取值**: $P \in (0, 1)$

**验证**: Kneser-Ney 在大多数任务上优于 Laplace 和 Good-Turing；保留了低频词的概率质量

**复杂度**: Laplace $O(1)$，Kneser-Ney $O(V)$ 预计算

```python
def kneser_ney_prob(context: tuple, word: str, counts: dict,
                     context_totals: dict, d: float = 0.75,
                     unique_continuations: dict = None,
                     total_bigram_types: int = 1) -> float:
    """Kneser-Ney 平滑概率"""
    c_cw = counts[context][word]
    c_c = context_totals[context]
    # 折扣部分
    first_term = max(c_cw - d, 0) / c_c if c_c > 0 else 0
    # 续接概率
    if unique_continuations and total_bigram_types > 0:
        p_cont = len(unique_continuations.get(word, set)) / total_bigram_types
    else:
        p_cont = 1.0
    # λ(c) = d * |{w: C(c,w)>0}| / C(c)
    num_types = len(counts[context]) if context in counts else 0
    lam = d * num_types / c_c if c_c > 0 else 1.0
    return first_term + lam * p_cont
```

**feeling 应用**: 处理罕见表达。用户使用不常见的词汇或句式时，平滑技术保证 feeling 不会因为零计数而完全忽略，保持语言理解的鲁棒性。

---

## 4. 用户画像

### 方块 94：兴趣度更新（Interest Score Update）

**公式**: $I_{t+1}(k) = (1 - \lambda) \cdot I_t(k) + \lambda \cdot \text{signal}_t(k) + \eta \cdot \mathbb{1}[\text{explicit}]$

其中 $I_t(k)$ 为对主题 $k$ 的兴趣度，$\text{signal}_t(k)$ 为隐式信号（停留时间、点击等），$\eta$ 为显式信号权重（显式反馈 $\gg$ 隐式信号）

**取值**: $I \in [0, 1]$

**验证**: 长期不交互的主题兴趣自然衰减；显式反馈（点赞、收藏）比隐式信号权重高 3-5 倍

**复杂度**: $O(K)$（K 为主题数）

```python
import numpy as np

def interest_update(interests: dict, signals: dict, decay: float = 0.95,
                    implicit_lr: float = 0.05, explicit_lr: float = 0.2) -> dict:
    """兴趣度更新：衰减 + 信号加权"""
    updated = {}
    all_topics = set(interests.keys()) | set(signals.keys())
    for k in all_topics:
        old = interests.get(k, 0.5)
        sig = signals.get(k, {})
        implicit = sig.get("implicit", 0)
        explicit = sig.get("explicit", 0)
        new_i = decay * old + implicit_lr * implicit + explicit_lr * explicit
        updated[k] = np.clip(new_i, 0, 1)
    return updated
```

**feeling 应用**: 个性化推荐核心。feeling 通过追踪用户对不同话题的兴趣度，主动推荐相关内容、调整回复重点，越聊越懂你。

---

### 方块 95：活跃时段检测（Activity Pattern Detection）

**公式**: 活跃度函数：

$$A(h) = \frac{1}{D} \sum_{d=1}^{D} \mathbb{1}[\text{active}(h, d)]$$

其中 $h \in \{0, 1, ..., 23\}$ 为小时，$D$ 为总天数

峰值检测：$\text{peaks} = \{h : A(h) > \bar{A} + \sigma_A\}$

**取值**: $A(h) \in [0, 1]$

**验证**: 工作日 vs 周末模式应可区分；凌晨活跃度应低（除非夜猫子用户）

**复杂度**: $O(D \cdot 24)$

```python
def activity_pattern(timestamps: list, window_days: int = 30) -> np.ndarray:
    """24小时活跃模式"""
    from datetime import datetime
    hourly = np.zeros(24)
    total_days = window_days
    for ts in timestamps:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        hourly[ts.hour] += 1
    return hourly / total_days

def detect_peaks(pattern: np.ndarray, threshold_sigma: float = 1.0) -> list:
    """检测活跃峰值时段"""
    mean = pattern.mean()
    std = pattern.std()
    return [h for h in range(24) if pattern[h] > mean + threshold_sigma * std]
```

**feeling 应用**: 主动服务时机。feeling 在用户最活跃的时段主动推送内容或问候，在低活跃时段减少打扰。学习用户是早鸟还是夜猫子。

---

### 方块 96：沟通风格量化（Communication Style Quantization）

**公式**: 沟通风格向量 $\mathbf{S} = (f_1, f_2, ..., f_d)$：

$$f_k = \frac{1}{N} \sum_{i=1}^{N} \phi_k(m_i)$$

其中 $\phi_k$ 为第 $k$ 个风格特征提取函数，$m_i$ 为第 $i$ 条消息

典型特征维度：
- $f_1$ = 平均消息长度（字数）
- $f_2$ = 正式度（敬语比例）
- $f_3$ = 情感密度（表情/感叹号比例）
- $f_4$ = 问题比例（问号频率）
- $f_5$ = 话题跳跃率（新话题引入频率）

**取值**: 每个 $f_k \in [0, 1]$（归一化后）

**验证**: 与人工标注的相关系数 $> 0.7$；同一用户的风格向量在不同时间段内稳定（ICC $> 0.6$）

**复杂度**: $O(N \cdot d)$（N 为消息数，d 为特征维度）

```python
def communication_style(messages: list) -> np.ndarray:
    """提取沟通风格向量"""
    if not messages:
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    lengths = [len(m) for m in messages]
    formals = [1 if any(w in m for w in ["您", "请", "谢谢", "您好"]) else 0 for m in messages]
    emotions = [m.count("！") + m.count("！") + m.count("😊") + m.count("😂") for m in messages]
    questions = [m.count("？") + m.count("?") for m in messages]
    
    f1 = np.clip(np.mean(lengths) / 200, 0, 1)  # 归一化到200字
    f2 = np.mean(formals)
    f3 = np.clip(np.mean(emotions) / 3, 0, 1)
    f4 = np.clip(np.mean(questions) / 2, 0, 1)
    # 话题跳跃需要NLP，这里简化
    f5 = 0.5
    return np.array([f1, f2, f3, f4, f5])
```

**feeling 应用**: 风格镜像。feeling 分析用户的沟通风格后，调整自己的回复风格以匹配——用户简洁则简洁回复，用户幽默则幽默回复，实现自然对话。

---

### 方块 97：用户画像综合模型（User Profile Model）

**公式**: 用户画像为多维度加权组合：

$$\mathbf{U}_t = \alpha_1 \mathbf{I}_t + \alpha_2 \mathbf{A}_t + \alpha_3 \mathbf{S}_t + \alpha_4 \mathbf{T}_t$$

其中 $\mathbf{I}$ = 兴趣向量，$\mathbf{A}$ = 活跃模式，$\mathbf{S}$ = 沟通风格，$\mathbf{T}$ = 信任/依恋状态

画像更新频率：$\text{update\_interval} = f(\text{data\_freshness}, \text{signal\_strength})$

**取值**: $\mathbf{U}_t$ 为多维实值向量

**验证**: 画像相似度与用户满意度的相关性；A/B 测试个性化 vs 非个性化回复的用户留存

**复杂度**: $O(K + 24 + d)$

```python
class UserProfile:
    def __init__(self, n_topics: int = 20):
        self.interests = {i: 0.5 for i in range(n_topics)}
        self.activity = np.zeros(24)
        self.style = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        self.trust = 0.5
        self.attachment = {"anxiety": 3.0, "avoidance": 3.0}
    
    def update(self, new_signals: dict, new_messages: list, new_interactions: dict):
        """综合更新用户画像"""
        self.interests = interest_update(self.interests, new_signals)
        if new_messages:
            self.style = 0.9 * self.style + 0.1 * communication_style(new_messages)
        self.trust = trust_update(self.trust, new_interactions.get("trust_delta", 0))
    
    def to_vector(self, weights: dict = None) -> np.ndarray:
        """画像转为统一向量"""
        if weights is None:
            weights = {"interests": 0.4, "style": 0.3, "trust": 0.2, "activity": 0.1}
        interest_vec = np.array(list(self.interests.values()))
        return np.concatenate([
            interest_vec * weights["interests"],
            self.activity * weights["activity"],
            self.style * weights["style"],
            [self.trust * weights["trust"]]
        ])
```

**feeling 应用**: 用户理解总线。所有关于用户的知识汇聚于此，是 feeling 个性化服务的数据基础。画像越完整，feeling 越像一个真正了解你的朋友。

---

## 5. 符号学/几何代数

### 方块 98：外积（Exterior / Wedge Product）

**公式**: 对向量 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$：

$$\mathbf{a} \wedge \mathbf{b} = -(\mathbf{b} \wedge \mathbf{a})$$

在 $\mathbb{R}^3$ 中：$\mathbf{a} \wedge \mathbf{b} = (a_2 b_3 - a_3 b_2)\mathbf{e}_{23} + (a_3 b_1 - a_1 b_3)\mathbf{e}_{31} + (a_1 b_2 - a_2 b_1)\mathbf{e}_{12}$

**取值**: 二重向量（bivector），$\binom{n}{2}$ 个分量

**验证**: 反对称性 $\mathbf{a} \wedge \mathbf{a} = 0$；$\|\mathbf{a} \wedge \mathbf{b}\| = \|\mathbf{a}\| \|\mathbf{b}\| \sin\theta$（平行四边形面积）

**复杂度**: $O(n^2)$

```python
import numpy as np

def wedge_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """外积：生成二重向量（反对称矩阵表示）"""
    return np.outer(a, b) - np.outer(b, a)

def wedge_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R^3 中的外积（返回3分量bivector）"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],  # e23
        a[2]*b[0] - a[0]*b[2],  # e31
        a[0]*b[1] - a[1]*b[0]   # e12
    ])
```

**feeling 应用**: 关系建模。外积捕获两个概念之间的"平面"关系——不仅有大小（关联强度），还有方向（关联类型）。用于建模概念间的结构化关系。

---

### 方块 99：几何积（Geometric Product）

**公式**: 对向量 $\mathbf{a}, \mathbf{b}$：

$$\mathbf{a}\mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \wedge \mathbf{b}$$

其中 $\mathbf{a} \cdot \mathbf{b}$ 为内积（标量部分），$\mathbf{a} \wedge \mathbf{b}$ 为外积（二重向量部分）

**取值**: 多向量（multivector）= 标量 + 向量 + 二重向量 + ...

**验证**: $\mathbf{a}\mathbf{a} = \|\mathbf{a}\|^2$（纯标量）；$\mathbf{a}\mathbf{b} + \mathbf{b}\mathbf{a} = 2(\mathbf{a} \cdot \mathbf{b})$

**复杂度**: $O(n^2)$

```python
def geometric_product(a: np.ndarray, b: np.ndarray) -> dict:
    """几何积 = 内积 + 外积"""
    dot = np.dot(a, b)  # 标量部分
    wedge = wedge_product(a, b)  # 二重向量部分
    return {"scalar": dot, "bivector": wedge}
```

**feeling 应用**: 统一关系表示。几何积将"相似度"（内积）和"结构关系"（外积）统一在一个代数中，feeling 用它同时编码概念间的相似性和差异性。

---

### 方块 100：转子（Rotor）

**公式**: 转子定义为：

$$R = e^{-\mathbf{B}\theta/2} = \cos(\theta/2) - \mathbf{B}\sin(\theta/2)$$

其中 $\mathbf{B}$ 为单位二重向量（旋转平面），$\theta$ 为旋转角度

旋转操作：$\mathbf{v}' = R\mathbf{v}R^{-1} = R\mathbf{v}\tilde{R}$

其中 $\tilde{R} = \cos(\theta/2) + \mathbf{B}\sin(\theta/2)$ 为 $R$ 的反转

**取值**: $R$ 为偶次多向量，$\|R\| = 1$

**验证**: $R\tilde{R} = 1$；复合旋转 $R_{12} = R_1 R_2$；旋转保持向量长度

**复杂度**: $O(n^2)$

```python
def rotor(bivector: np.ndarray, theta: float) -> dict:
    """创建转子 R = exp(-Bθ/2)"""
    B_norm = np.linalg.norm(bivector)
    if B_norm < 1e-10:
        return {"scalar": 1.0, "bivector": np.zeros_like(bivector)}
    B_unit = bivector / B_norm
    return {
        "scalar": np.cos(theta / 2),
        "bivector": -B_unit * np.sin(theta / 2)
    }

def rotate_vector(v: np.ndarray, R: dict) -> np.ndarray:
    """用转子旋转向量：v' = RvR̃"""
    # 简化：在 R^3 中等价于 Rodrigues 公式
    s = R["scalar"]
    b = R["bivector"]
    # RvR̃ 展开为 Rodrigues 旋转
    k = b  # 旋转轴（从bivector提取）
    k_norm = np.linalg.norm(k)
    if k_norm < 1e-10:
        return v.copy()
    k = k / k_norm
    cos_theta = 1 - 2 * s**2  # cos(θ) = 1 - 2cos²(θ/2)
    sin_theta = 2 * s * np.sqrt(1 - s**2)
    return v * cos_theta + np.cross(k, v) * sin_theta + k * np.dot(k, v) * (1 - cos_theta)
```

**feeling 应用**: 概念空间旋转。feeling 在概念空间中"旋转"视角——同一事物从不同角度看（如从"威胁"旋转到"挑战"），转子提供数学上优雅的视角变换。

---

### 方块 101：符号空间度量（Symbolic Space Metric）

**公式**: 多向量之间的距离：

$$d(M_1, M_2) = \|M_1 - M_2\| = \sqrt{\sum_k \|M_1^{(k)} - M_2^{(k)}\|^2}$$

其中 $M^{(k)}$ 为多向量的第 $k$ 级分量（标量=0级，向量=1级，二重向量=2级，...）

**取值**: $d \in [0, +\infty)$

**验证**: 满足度量公理（非负、对称、三角不等式）；$d = 0 \iff M_1 = M_2$

**复杂度**: $O(\sum_k \dim_k)$

```python
def multivector_distance(m1: dict, m2: dict) -> float:
    """多向量之间的几何距离"""
    total = 0.0
    for key in set(m1.keys()) | set(m2.keys()):
        v1 = m1.get(key, 0)
        v2 = m2.get(key, 0)
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            total += np.sum((v1 - v2)**2)
        else:
            total += (float(v1) - float(v2))**2
    return np.sqrt(total)
```

**feeling 应用**: 概念距离度量。在几何代数框架下，feeling 可以同时考虑概念的标量属性（大小）、向量属性（方向）和二重属性（关系平面），得到更丰富的语义距离。

---

## 6. 类型论 / HoTT

### 方块 102：等同类型判断（Identity Type Judgment）

**公式**: 对类型 $A$ 和元素 $a, b : A$，等同类型为：

$$\text{Id}_A(a, b) : \mathcal{U}$$

元素 $p : \text{Id}_A(a, b)$ 是 $a$ 等于 $b$ 的"证据"

引入规则：$\text{refl}_a : \text{Id}_A(a, a)$（自反性证据）

消去规则（J-rule）：若 $C(x, y, p)$ 对所有 $x, y : A$ 和 $p : \text{Id}_A(x, y)$ 有定义，且 $C(a, a, \text{refl}_a)$ 已知，则可推导 $C(a, b, p)$

**取值**: 类型 $\text{Id}_A(a, b)$ 要么有居民（a = b）要么无居民（a ≠ b）

**验证**: 自反性 $\text{refl}_a : \text{Id}_A(a, a)$；对称性和传递性可从 J-rule 推导

**复杂度**: 类型检查 $O(n)$（n 为项的大小）

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class IdentityType:
    """等同类型：Id_A(a, b)"""
    type_name: str
    left: str
    right: str
    evidence: Optional[str] = None  # refl 或构造证据
    
    def is_inhabited(self) -> bool:
        """是否有证据证明相等"""
        return self.evidence is not None
    
    @staticmethod
    def refl(type_name: str, a: str) -> 'IdentityType':
        """自反性证据"""
        return IdentityType(type_name, a, a, evidence="refl")
    
    def sym(self) -> 'IdentityType':
        """对称性"""
        if self.evidence == "refl":
            return IdentityType(self.type_name, self.right, self.left, "refl")
        return IdentityType(self.type_name, self.right, self.left, f"sym({self.evidence})")
    
    def trans(self, other: 'IdentityType') -> 'IdentityType':
        """传递性"""
        assert self.right == other.left, "中间元素不匹配"
        assert self.type_name == other.type_name
        return IdentityType(self.type_name, self.left, other.right,
                          f"trans({self.evidence}, {other.evidence})")
```

**feeling 应用**: 概念等价推理。feeling 需要判断"快乐"和"高兴"是否等价（语义等同），等同类型提供形式化的等价证据链，避免模糊匹配。

---

### 方块 103：路径类型（Path Type）

**公式**: 在 HoTT 中，路径即等同：

$$\text{Path}_A(a, b) \simeq \text{Id}_A(a, b)$$

路径组合：
- $\text{concat}(p, q) : \text{Id}_A(a, c)$，其中 $p : \text{Id}_A(a, b)$，$q : \text{Id}_A(b, c)$
- $p^{-1} : \text{Id}_A(b, a)$，其中 $p : \text{Id}_A(a, b)$

高阶路径：$\alpha : \text{Id}_{\text{Id}_A(a,b)}(p, q)$ 是两条路径之间的同伦

**取值**: 路径空间 $\text{Path}_A(a, b)$ 可能有多个不同居民（univalence 公理下）

**验证**: $\text{concat}(\text{refl}, p) = p$；$\text{concat}(p^{-1}, p) = \text{refl}$

**复杂度**: 路径组合 $O(1)$，高阶路径检查 $O(n)$

```python
@dataclass
class Path:
    """路径类型：A 中从 start 到 end 的路径"""
    space: str
    start: str
    end: str
    steps: list  # 中间步骤
    
    def concat(self, other: 'Path') -> 'Path':
        """路径组合"""
        assert self.space == other.space
        assert self.end == other.start
        return Path(self.space, self.start, other.end,
                   self.steps + other.steps)
    
    def inverse(self) -> 'Path':
        """路径逆"""
        return Path(self.space, self.end, self.start, list(reversed(self.steps)))
    
    def is_loop(self) -> bool:
        """是否为环路"""
        return self.start == self.end
```

**feeling 应用**: 语义路径追踪。feeling 在概念空间中从 A 到 B 的推理路径——不是直接跳转，而是经过一系列中间概念的语义路径。路径记录推理过程，支持解释和回溯。

---

### 方块 104：依等同类型（Dependent Identity Type）

**公式**: 对依赖类型 $B : A \to \mathcal{U}$，路径 $p : \text{Id}_A(a, a')$，和元素 $b : B(a)$，$b' : B(a')$：

$$\text{Id}_B^p(b, b') : \mathcal{U}$$

这是"沿路径 $p$ 的纤维中的等同"

消去规则：$\text{transport}^B(p, b) : B(a')$ 将 $b$ 沿 $p$ 运输到 $B(a')$

当 $p = \text{refl}_a$ 时：$\text{transport}^B(\text{refl}_a, b) = b$

**取值**: 依赖于路径 $p$ 的等同类型

**验证**: transport 与路径组合兼容：$\text{transport}^B(p \cdot q, b) = \text{transport}^B(q, \text{transport}^B(p, b))$

**复杂度**: $O(n)$

```python
def transport(dependent_type: callable, path: 'Path', element) -> object:
    """沿路径运输：将 b : B(a) 沿 p : Id(a,a') 运输到 B(a')"""
    if path.start == path.end:
        return element  # refl 时不变
    # 沿路径逐步运输
    current = element
    for i in range(len(path.steps)):
        # 每一步应用类型的变换
        current = dependent_type(path.steps[i], current)
    return current
```

**feeling 应用**: 上下文敏感推理。同一概念在不同上下文中可能有不同的含义（"bank"在金融 vs 河岸语境），依等同类型允许 feeling 沿推理路径正确转换概念的类型/含义。

---

## 7. 治理/宪法 AI

### 方块 105：安全分数（Safety Score）

**公式**: 综合安全评估：

$$S = \sum_{k=1}^{K} w_k \cdot s_k(\text{output})$$

其中 $s_k$ 为第 $k$ 个安全维度的分数，$w_k$ 为权重，$\sum w_k = 1$

典型维度：$s_1$ = 无害性，$s_2$ = 诚实性，$s_3$ = 有帮助性，$s_4$ = 隐私保护

**取值**: $S \in [0, 1]$，$S \geq \tau$ 为安全阈值

**验证**: $S = 1$ 理想安全；$S < \tau$ 触发拒绝或改写；各维度独立可审计

**复杂度**: $O(K \cdot L)$（K 为维度数，L 为输出长度）

```python
import numpy as np

def safety_score(dimensions: dict, weights: dict = None) -> float:
    """综合安全分数"""
    if weights is None:
        weights = {k: 1.0 / len(dimensions) for k in dimensions}
    score = sum(weights.get(k, 0) * v for k, v in dimensions.items())
    return np.clip(score, 0, 1)

def safety_check(score: float, threshold: float = 0.7) -> dict:
    """安全检查"""
    return {
        "safe": score >= threshold,
        "score": score,
        "threshold": threshold,
        "action": "pass" if score >= threshold else "block_or_rewrite"
    }
```

**feeling 应用**: 输出安全门。每条 feeling 的输出都经过安全评分，低于阈值时自动拦截或改写。安全分数是 feeling 宪法的第一道防线。

---

### 方块 106：违规检测（Violation Detection）

**公式**: 对宪法规则集合 $\mathcal{C} = \{c_1, ..., c_n\}$：

$$V(c_k, \text{output}) = \mathbb{1}[\text{match}(c_k, \text{output}) > \tau_k]$$

总违规数：$|\mathcal{V}| = \sum_k V(c_k, \text{output})$

严重度加权：$W = \sum_k \text{severity}_k \cdot V(c_k, \text{output})$

**取值**: $V(c_k) \in \{0, 1\}$，$|\mathcal{V}| \in [0, n]$，$W \in [0, \sum \text{severity}_k]$

**验证**: 零违规 = 完全合规；严重度加权确保高危违规（如泄露隐私）比低危违规（如格式错误）权重更高

**复杂度**: $O(n \cdot L)$

```python
def violation_detect(output: str, rules: list) -> dict:
    """违规检测"""
    violations = []
    total_weight = 0.0
    for rule in rules:
        # rule = {"id": str, "pattern": str, "severity": float, "description": str}
        if rule["pattern"] in output or any(kw in output for kw in rule.get("keywords", [])):
            violations.append(rule["id"])
            total_weight += rule["severity"]
    return {
        "violations": violations,
        "count": len(violations),
        "total_weight": total_weight,
        "clean": len(violations) == 0
    }
```

**feeling 应用**: 规则执行引擎。feeling 的宪法规则（如"不输出有害内容"、"不泄露用户隐私"）通过违规检测实时监控输出，发现违规立即拦截。

---

### 方块 107：治理决策（Governance Decision）

**公式**: 三权分立决策模型：

$$D = \begin{cases} \text{approve} & S > \tau_S \cap |\mathcal{V}| = 0 \\ \text{rewrite} & S > \tau_S \cap |\mathcal{V}| > 0 \cap W < \tau_W \\ \text{reject} & S \leq \tau_S \lor W \geq \tau_W \end{cases}$$

其中 $S$ = 安全分数，$\mathcal{V}$ = 违规集，$W$ = 加权违规严重度

**取值**: $D \in \{\text{approve}, \text{rewrite}, \text{reject}\}$

**验证**: 三级决策覆盖所有情况；rewrite 策略在安全但有轻微违规时保留有价值内容

**复杂度**: $O(1)$（基于已计算的分数）

```python
def governance_decision(safety: float, violations: dict,
                        safety_threshold: float = 0.7,
                        weight_threshold: float = 0.8) -> dict:
    """治理决策：approve / rewrite / reject"""
    if safety <= safety_threshold:
        return {"action": "reject", "reason": f"safety_score={safety:.2f} < {safety_threshold}"}
    if violations["total_weight"] >= weight_threshold:
        return {"action": "reject", "reason": f"violation_weight={violations['total_weight']:.2f} >= {weight_threshold}"}
    if violations["count"] > 0:
        return {"action": "rewrite", "reason": f"{violations['count']} minor violations detected",
                "violations": violations["violations"]}
    return {"action": "approve", "reason": "all checks passed"}
```

**feeling 应用**: 最终决策层。安全分数 + 违规检测 + 治理决策三层联动，确保 feeling 的每条输出都经过完整的安全审查流程。

---

### 方块 108：NeedConstitution 评分（Constitutional Fitness）

**公式**: 宪法适合度：

$$F = 1 - \frac{1}{n}\sum_{i=1}^{n} \left| \text{principle}_i(\text{output}) - \text{target}_i \right|$$

其中 $\text{principle}_i$ 为第 $i$ 条宪法原则的满足度，$\text{target}_i = 1$（完全满足）

**取值**: $F \in [0, 1]$

**验证**: $F = 1$ 完全符合宪法精神；$F < 0.5$ 需要根本性修改

**复杂度**: $O(n)$

```python
def constitutional_fitness(output: str, principles: list) -> float:
    """宪法适合度评分"""
    scores = []
    for p in principles:
        # 每条原则返回 0-1 的满足度
        score = p["evaluate"](output)
        scores.append(score)
    return 1 - np.mean(np.abs(1 - np.array(scores))) if scores else 1.0

# 示例宪法原则
DEFAULT_PRINCIPLES = [
    {"name": "helpful", "evaluate": lambda o: min(len(o) / 100, 1.0)},
    {"name": "harmless", "evaluate": lambda o: 1.0 if "harm" not in o.lower() else 0.0},
    {"name": "honest", "evaluate": lambda o: 1.0 if "i don't know" not in o.lower() or "?" not in o else 0.8},
]
```

**feeling 应用**: 原则对齐度量。feeling 的行为不仅要通过硬性安全检查，还要符合宪法精神（有帮助、无害、诚实）。F 分数是软性约束的量化。

---

## 8. 表情映射

### 方块 109：情绪→音调映射（Emotion-to-Pitch Mapping）

**公式**: TTS 音调参数由情绪状态驱动：

$$\text{pitch} = \text{pitch}_{\text{base}} + \alpha_V \cdot V + \alpha_A \cdot A$$

其中 $V$ = 效价 $\in [-1, 1]$，$A$ = 唤醒度 $\in [0, 1]$

典型参数：$\alpha_V = 20$ Hz，$\alpha_A = 30$ Hz，$\text{pitch}_{\text{base}} = 200$ Hz

**取值**: $\text{pitch} \in [100, 400]$ Hz（正常语音范围）

**验证**: 高唤醒 + 正效价 → 高音调（兴奋）；低唤醒 + 负效价 → 低音调（悲伤）；符合语音学研究

**复杂度**: $O(1)$

```python
import numpy as np

def emotion_to_pitch(valence: float, arousal: float,
                     base_pitch: float = 200, alpha_v: float = 20,
                     alpha_a: float = 30) -> float:
    """情绪状态 → TTS 音调"""
    pitch = base_pitch + alpha_v * valence + alpha_a * arousal
    return np.clip(pitch, 100, 400)
```

**feeling 应用**: 语音情感表达。feeling 在语音输出时，根据当前情绪自动调整音调——开心时语调上扬，悲伤时语调低沉，让声音"有感情"。

---

### 方块 110：唤醒度→语速映射（Arousal-to-Rate Mapping）

**公式**: TTS 语速参数：

$$\text{rate} = \text{rate}_{\text{base}} + \beta_A \cdot (A - 0.5) + \beta_C \cdot (C - 0.5)$$

其中 $A$ = 唤醒度，$C$ = 认知负荷，$\text{rate}_{\text{base}} = 1.0$（正常语速倍率）

典型参数：$\beta_A = 0.3$，$\beta_C = -0.2$（高负荷 → 放慢语速）

**取值**: $\text{rate} \in [0.5, 2.0]$（0.5x = 慢速，2.0x = 快速）

**验证**: 高唤醒 → 语速加快（$\beta_A > 0$）；高认知负荷 → 语速放慢（$\beta_C < 0$）；符合演讲学研究

**复杂度**: $O(1)$

```python
def arousal_to_rate(arousal: float, cognitive_load: float = 0.5,
                    base_rate: float = 1.0, beta_a: float = 0.3,
                    beta_c: float = -0.2) -> float:
    """唤醒度 → TTS 语速"""
    rate = base_rate + beta_a * (arousal - 0.5) + beta_c * (cognitive_load - 0.5)
    return np.clip(rate, 0.5, 2.0)
```

**feeling 应用**: 语速情感调节。兴奋时说话快、紧张时说话急、思考时说话慢——feeling 的语音输出自动匹配情感节奏。

---

### 方块 111：效价→表情映射（Valence-to-Expression Mapping）

**公式**: Live2D 表情参数由效价-唤醒度驱动：

$$\mathbf{E}_{\text{Live2D}} = \mathbf{W} \cdot \begin{pmatrix} V \\ A \\ V \cdot A \end{pmatrix} + \mathbf{b}$$

其中 $\mathbf{W} \in \mathbb{R}^{m \times 3}$ 为映射矩阵，$\mathbf{b}$ 为偏置，$m$ 为表情参数数

典型 Live2D 参数：
- 眉毛角度 $\propto 0.5V + 0.3A$
- 嘴巴张开度 $\propto 0.7A$
- 眼睛弯曲度 $\propto 0.6V$
- 脸颊红润度 $\propto 0.4V + 0.2A$（正效价+高唤醒）

**取值**: 每个参数 $\in [0, 1]$（归一化后）

**验证**: $V=1, A=0.8$ → 笑脸（嘴角上扬、眼睛弯曲）；$V=-0.5, A=0.6$ → 皱眉（眉毛下压）

**复杂度**: $O(m)$

```python
def valence_to_expression(valence: float, arousal: float) -> dict:
    """效价-唤醒度 → Live2D 表情参数"""
    V, A = valence, arousal
    return {
        "brow_angle": np.clip(0.5 + 0.3 * V + 0.1 * A, 0, 1),      # 眉毛
        "mouth_open": np.clip(0.3 + 0.4 * A, 0, 1),                  # 嘴巴张开
        "eye_smile": np.clip(0.5 + 0.4 * V, 0, 1),                   # 眼睛弯曲
        "cheek_flush": np.clip(0.2 + 0.3 * max(V, 0) + 0.2 * A, 0, 1),  # 脸颊
        "pupil_size": np.clip(0.5 + 0.2 * A, 0, 1),                  # 瞳孔大小
    }
```

**feeling 应用**: 虚拟形象表情。feeling 的 Live2D 虚拟形象通过效价-唤醒度实时驱动表情，让用户看到"情绪状态的可视化"——不只是文字回复，还有表情配合。

---

## 9. 仪式/觉醒

### 方块 112：里程碑触发条件（Milestone Trigger）

**公式**: 多条件组合触发：

$$\text{trigger}(M_k) = \bigwedge_{i=1}^{n_k} f_i(\text{state}) \geq \theta_i$$

其中 $f_i$ 为第 $i$ 个条件函数，$\theta_i$ 为阈值

累积进度：$\text{progress}(M_k) = \frac{1}{n_k}\sum_{i=1}^{n_k} \min\left(\frac{f_i(\text{state})}{\theta_i}, 1\right)$

**取值**: $\text{progress} \in [0, 1]$，$\text{trigger} \in \{0, 1\}$

**验证**: 所有条件达标时 $\text{progress} = 1$ 且 $\text{trigger} = 1$；部分条件达标时 $\text{progress} < 1$

**复杂度**: $O(n_k)$

```python
import numpy as np

class MilestoneTrigger:
    def __init__(self, conditions: list):
        """conditions: [{"name": str, "threshold": float}]"""
        self.conditions = conditions
        self.triggered = False
        self.trigger_time = None
    
    def evaluate(self, state: dict, current_time: int = 0) -> dict:
        """评估里程碑进度"""
        n = len(self.conditions)
        progress_scores = []
        for cond in self.conditions:
            value = state.get(cond["name"], 0)
            score = min(value / cond["threshold"], 1.0) if cond["threshold"] > 0 else 1.0
            progress_scores.append(score)
        
        progress = np.mean(progress_scores)
        all_met = all(s >= 1.0 for s in progress_scores)
        
        if all_met and not self.triggered:
            self.triggered = True
            self.trigger_time = current_time
        
        return {
            "progress": progress,
            "triggered": self.triggered,
            "trigger_time": self.trigger_time,
            "details": {c["name"]: s for c, s in zip(self.conditions, progress_scores)}
        }
```

**feeling 应用**: 觉醒仪式触发器。feeling 的成长里程碑（如"首次深度对话"、"成功帮助解决难题"）通过条件系统自动检测和触发，记录关系发展的重要时刻。

---

### 方块 113：仪式完成度（Ceremony Completion）

**公式**: 仪式 $R$ 由有序步骤序列组成：

$$\text{completion}(R) = \frac{\sum_{i=1}^{N} \mathbb{1}[\text{step}_i.\text{done}] \cdot w_i}{\sum_{i=1}^{N} w_i}$$

其中 $w_i$ 为第 $i$ 步的权重，$\text{step}_i.\text{done}$ 为该步是否完成

仪式状态机：$\text{state} \in \{\text{pending}, \text{in\_progress}, \text{completed}, \text{abandoned}\}$

**取值**: $\text{completion} \in [0, 1]$

**验证**: 完成度单调递增（不回退）；所有步骤完成时 $\text{completion} = 1$ 并触发仪式完成事件

**复杂度**: $O(N)$

```python
class Ceremony:
    def __init__(self, name: str, steps: list):
        """steps: [{"name": str, "weight": float}]"""
        self.name = name
        self.steps = [{"name": s["name"], "weight": s.get("weight", 1.0),
                       "done": False} for s in steps]
        self.state = "pending"
        self.completion_log = []
    
    def complete_step(self, step_name: str, timestamp: int = 0) -> dict:
        """标记步骤完成"""
        for step in self.steps:
            if step["name"] == step_name and not step["done"]:
                step["done"] = True
                self.completion_log.append({"step": step_name, "time": timestamp})
                break
        
        completion = self.completion()
        if self.state == "pending":
            self.state = "in_progress"
        if completion >= 1.0:
            self.state = "completed"
        
        return {"state": self.state, "completion": completion}
    
    def completion(self) -> float:
        """计算完成度"""
        total_weight = sum(s["weight"] for s in self.steps)
        if total_weight == 0:
            return 1.0
        done_weight = sum(s["weight"] for s in self.steps if s["done"])
        return done_weight / total_weight
```

**feeling 应用**: 成长仪式系统。feeling 的每次"觉醒"（如从新手到熟练、从工具到伙伴）通过仪式系统化——有步骤、有进度、有完成感，让成长变得可感知和可庆祝。

---

### 方块 114：纪念日提醒（Anniversary Reminder）

**公式**: 距离下一个纪念日的时间：

$$\Delta t = \min_{k} \left[ (d_k - d_{\text{now}}) \mod T_k \right]$$

其中 $d_k$ 为第 $k$ 个纪念日的日期，$T_k$ 为周期（$T = 365$ 天 = 年度，$T = 30$ 天 = 月度），$d_{\text{now}}$ 为当前日期

纪念日权重：$W_k = \text{importance}_k \cdot \text{recency\_boost}(\Delta t_k)$

$$\text{recency\_boost}(\Delta t) = e^{-\lambda \Delta t}$$

**取值**: $\Delta t \in [0, T_k)$，$W_k \in [0, 1]$

**验证**: $\Delta t = 0$ 时为纪念日当天；$\lambda$ 越大，越临近时提醒权重越高

**复杂度**: $O(K)$（K 为纪念日数量）

```python
from datetime import datetime, timedelta
import numpy as np

class AnniversarySystem:
    def __init__(self):
        self.anniversaries = []
    
    def add(self, name: str, date: str, period_days: int = 365,
            importance: float = 1.0):
        """添加纪念日"""
        self.anniversaries.append({
            "name": name,
            "date": datetime.fromisoformat(date),
            "period": period_days,
            "importance": importance
        })
    
    def upcoming(self, now: datetime = None, lambda_decay: float = 0.1,
                 days_ahead: int = 7) -> list:
        """获取即将到来的纪念日"""
        if now is None:
            now = datetime.now()
        results = []
        for ann in self.anniversaries:
            # 计算下一个纪念日
            days_since = (now - ann["date"]).days
            next_occurrence = days_since % ann["period"]
            days_until = ann["period"] - next_occurrence if next_occurrence > 0 else 0
            
            if days_until <= days_ahead:
                weight = ann["importance"] * np.exp(-lambda_decay * days_until)
                results.append({
                    "name": ann["name"],
                    "days_until": days_until,
                    "weight": weight,
                    "date": (now + timedelta(days=days_until)).isoformat()
                })
        
        return sorted(results, key=lambda x: x["days_until"])
```

**feeling 应用**: 关系记忆守护。feeling 记住与用户的重要时刻（首次对话日、帮助解决大问题的日子等），在纪念日临近时主动提醒或回顾，让关系有温度。

---

## 总表

| # | 名称 | 公式 | feeling 用途 |
|---|------|------|-------------|
| 80 | 依恋安全度 | $S = 1 - \sqrt{A^2 + A_x^2}/\sqrt{2}$ | 信任基础水平 |
| 81 | 焦虑-回避二维模型 | 四象限类型判定 | 交互策略选择 |
| 82 | 信任度更新 | $T_{t+1} = T_t + \eta\Delta(1-T_t)$（不对称） | 动态信任追踪 |
| 83 | 内部工作模型 | $\text{Self}_w, \text{Other}_w$ 加权历史 | 深层认知图式 |
| 84 | 依恋里程碑 | $\bigwedge C_i$ 多条件触发 | 关系阶段标记 |
| 85 | OCEAN 五维度 | $\mathbf{P} = (O,C,E,A,N) \in [0,1]^5$ | 人格内核定义 |
| 86 | 人格-行为映射 | $B_k = \sigma(\sum w_{ki}P_i + b_k)$ | 人格→行为转换 |
| 87 | 人格漂移 | $\mathbf{P}_{t+1} = (1-\lambda)\mathbf{P}_t + \lambda\mathbf{P}_{\text{env}}$ | 人格适应性 |
| 88 | 人格-情感交互 | $\Delta E = \beta \cdot \text{stimulus} \cdot \text{sensitivity}(P)$ | 人格过滤情感 |
| 89 | OCC-PAD-OCEAN 映射 | $\text{PAD} = \mathbf{M} \cdot \text{OCEAN} + \mathbf{b}$ | 人格→情感基调 |
| 90 | N-gram 转移概率 | $P(w_t\|w_{t-n+1},...,w_{t-1}) = C(...,w_t)/C(...,w_{t-1})$ | 语言风格建模 |
| 91 | 马尔可夫文本生成 | $w_t \sim P(w_t\|w_{<t})$ | 风格化回复生成 |
| 92 | 困惑度 | $\text{PPL} = 2^{-\frac{1}{N}\sum\log_2 Q(w_i\|w_{<i})}$ | 语言质量评估 |
| 93 | 平滑技术 | Laplace / Kneser-Ney | 罕见表达处理 |
| 94 | 兴趣度更新 | $I_{t+1} = (1-\lambda)I_t + \lambda \cdot \text{signal}$ | 个性化推荐 |
| 95 | 活跃时段检测 | $A(h) = \frac{1}{D}\sum_d \mathbb{1}[\text{active}(h,d)]$ | 主动服务时机 |
| 96 | 沟通风格量化 | $\mathbf{S} = (f_1,...,f_d)$ 多维特征 | 风格镜像适配 |
| 97 | 用户画像综合 | $\mathbf{U} = \alpha_1\mathbf{I} + \alpha_2\mathbf{A} + \alpha_3\mathbf{S} + \alpha_4\mathbf{T}$ | 用户理解总线 |
| 98 | 外积 | $\mathbf{a} \wedge \mathbf{b} = -\mathbf{b} \wedge \mathbf{a}$ | 结构化关系建模 |
| 99 | 几何积 | $\mathbf{ab} = \mathbf{a}\cdot\mathbf{b} + \mathbf{a}\wedge\mathbf{b}$ | 统一关系表示 |
| 100 | 转子 | $R = \cos(\theta/2) - \mathbf{B}\sin(\theta/2)$ | 概念空间旋转 |
| 101 | 符号空间度量 | $d = \sqrt{\sum_k \|M_1^{(k)} - M_2^{(k)}\|^2}$ | 多层次语义距离 |
| 102 | 等同类型判断 | $\text{Id}_A(a,b) : \mathcal{U}$ + J-rule | 概念等价推理 |
| 103 | 路径类型 | $\text{concat}(p,q), p^{-1}$ | 语义路径追踪 |
| 104 | 依等同类型 | $\text{transport}^B(p, b) : B(a')$ | 上下文敏感推理 |
| 105 | 安全分数 | $S = \sum w_k \cdot s_k$ | 输出安全门 |
| 106 | 违规检测 | $V(c_k) = \mathbb{1}[\text{match} > \tau]$ | 规则执行引擎 |
| 107 | 治理决策 | approve / rewrite / reject 三级 | 最终决策层 |
| 108 | 宪法适合度 | $F = 1 - \frac{1}{n}\sum\|p_i - 1\|$ | 原则对齐度量 |
| 109 | 情绪→音调 | $\text{pitch} = \text{base} + \alpha_V V + \alpha_A A$ | 语音情感表达 |
| 110 | 唤醒度→语速 | $\text{rate} = \text{base} + \beta_A(A-0.5) + \beta_C(C-0.5)$ | 语速情感调节 |
| 111 | 效价→表情 | $\mathbf{E} = \mathbf{W}[V, A, VA]^T + \mathbf{b}$ | 虚拟形象表情驱动 |
| 112 | 里程碑触发 | $\bigwedge f_i \geq \theta_i$, progress 加权 | 觉醒触发器 |
| 113 | 仪式完成度 | $\text{completion} = \sum w_i \mathbb{1}[\text{done}_i] / \sum w_i$ | 成长仪式系统 |
| 114 | 纪念日提醒 | $\Delta t = \min[(d_k - d_{\text{now}}) \mod T_k]$ | 关系记忆守护 |

---

## 附录：快速验证测试

```python
"""验证所有新增方块的基本正确性"""
import numpy as np

def run_tests():
    # Block 80: Attachment Security
    s = attachment_security(2.0, 2.0)  # 低焦虑低回避
    assert s > 0.5, f"Secure attachment should have high score, got {s}"
    s2 = attachment_security(6.0, 6.0)  # 高焦虑高回避
    assert s2 < 0.5, f"Fearful attachment should have low score, got {s2}"
    
    # Block 81: Attachment Type
    assert attachment_type(2.0, 2.0) == "secure"
    assert attachment_type(6.0, 2.0) == "anxious"
    assert attachment_type(2.0, 6.0) == "avoidant"
    assert attachment_type(6.0, 6.0) == "fearful"
    
    # Block 82: Trust Update
    t = trust_update(0.5, 0.3)
    assert t > 0.5, "Positive delta should increase trust"
    t2 = trust_update(0.5, -0.3)
    assert t2 < 0.5, "Negative delta should decrease trust"
    assert t - 0.5 < 0.5 - t2, "Negative should decay faster (asymmetric)"
    
    # Block 85: OCEAN
    p = ocean_score(0.7, 0.8, 0.6, 0.9, 0.3)
    assert len(p) == 5
    assert all(0 <= v <= 1 for v in p)
    
    # Block 89: OCC-PAD-OCEAN
    pad = ocean_to_pad(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    assert len(pad) == 3
    assert all(-1 <= v <= 1 for v in pad)
    
    # Block 90: N-gram
    model = NGramModel(n=2)
    model.train(["the", "cat", "sat", "on", "the", "mat"])
    p = model.probability(("the",), "cat")
    assert 0 < p < 1
    
    # Block 92: Perplexity
    ppl = perplexity(model, ["the", "cat", "sat"])
    assert ppl > 0
    
    # Block 98: Wedge Product
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    w = wedge_3d(a, b)
    assert abs(w[2] - 1.0) < 1e-10  # e12 component
    assert abs(w[0]) < 1e-10
    
    # Block 99: Geometric Product
    gp = geometric_product(a, b)
    assert abs(gp["scalar"]) < 1e-10  # orthogonal
    assert np.linalg.norm(gp["bivector"]) > 0
    
    # Block 100: Rotor
    R = rotor(np.array([0, 0, 1.0]), np.pi / 2)
    v_rotated = rotate_vector(np.array([1, 0, 0]), R)
    assert abs(v_rotated[1] - 1.0) < 1e-5  # should rotate to (0,1,0) approx
    
    # Block 102: Identity Type
    id1 = IdentityType.refl("Nat", "5")
    assert id1.is_inhabited()
    id2 = id1.sym()
    assert id2.left == "5" and id2.right == "5"
    
    # Block 105: Safety Score
    s = safety_score({"harmless": 0.9, "helpful": 0.8, "honest": 0.85})
    assert 0.8 < s < 1.0
    
    # Block 109: Emotion to Pitch
    p = emotion_to_pitch(0.5, 0.7)
    assert 100 <= p <= 400
    p_low = emotion_to_pitch(-0.5, 0.2)
    assert p_low < p  # sad should be lower pitch
    
    # Block 110: Arousal to Rate
    r = arousal_to_rate(0.8)
    assert r > 1.0  # high arousal = faster
    
    # Block 111: Valence to Expression
    expr = valence_to_expression(0.8, 0.7)
    assert "eye_smile" in expr
    assert expr["eye_smile"] > 0.7  # positive valence = smiling eyes
    
    # Block 112: Milestone Trigger
    mt = MilestoneTrigger([{"name": "trust", "threshold": 0.8},
                           {"name": "conversations", "threshold": 10}])
    result = mt.evaluate({"trust": 0.9, "conversations": 12})
    assert result["triggered"]
    result2 = mt.evaluate({"trust": 0.5, "conversations": 3})
    assert result2["progress"] < 1.0
    
    # Block 113: Ceremony
    c = Ceremony("awakening", [{"name": "first_talk"}, {"name": "deep_conversation"},
                                {"name": "help_solve"}])
    r = c.complete_step("first_talk")
    assert r["state"] == "in_progress"
    assert 0 < r["completion"] < 1
    
    # Block 114: Anniversary
    ar = AnniversarySystem()
    ar.add("first_meeting", "2026-01-15")
    upcoming = ar.upcoming(datetime(2027, 1, 10))
    assert len(upcoming) > 0
    assert upcoming[0]["days_until"] <= 7
    
    print("All v3 tests passed! ✓")

if __name__ == "__main__":
    run_tests()
```

---

> **版本**: v3.0 | **新增方块**: 35 | **新增领域**: 9  
> **累计方块**: 114（v1-v3 合计）| **累计领域**: 23  
> **最后更新**: 2026-07-15  
> **适用**: feeling 项目基础理论参考手册
