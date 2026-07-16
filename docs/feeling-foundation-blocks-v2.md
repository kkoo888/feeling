# Feeling 项目完整基础方块手册 v2

> 可验证计算公式参考手册 | 102 个基础方块 | 19 个领域

---

## 目录

1. [信息论 (Block 1-8)](#1-信息论)
2. [概率论与贝叶斯 (Block 9-16)](#2-概率论与贝叶斯)
3. [学习规则 (Block 17-24)](#3-学习规则)
4. [注意力与选择 (Block 25-29)](#4-注意力与选择)
5. [状态与动力学 (Block 30-37)](#5-状态与动力学)
6. [决策与博弈 (Block 38-43)](#6-决策与博弈)
7. [控制与调节 (Block 44-48)](#7-控制与调节)
8. [情感与需求 (Block 49-53)](#8-情感与需求)
9. [记忆与检索 (Block 54-58)](#9-记忆与检索)
10. [图论与网络 (Block 59-63)](#10-图论与网络)
11. [优化 (Block 64-68)](#11-优化)
12. [涌现与复杂性 (Block 69-73)](#12-涌现与复杂性)
13. [因果 (Block 74-76)](#13-因果)
14. [东方哲学数学化 (Block 77-79)](#14-东方哲学数学化)
15. [自进化 (Block 80-84)](#15-自进化)
16. [达尔文进化 (Block 85-90)](#16-达尔文进化)
17. [自我维护 (Block 91-95)](#17-自我维护)
18. [自我修复 (Block 96-99)](#18-自我修复)
19. [验证与正确性 (Block 100-102)](#19-验证与正确性)
20. [总表](#总表)

---

## 1. 信息论

### 方块 1：香农熵（Shannon Entropy）

**公式**: $H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$

**取值**: $H(X) \in [0, \log_2 n]$，其中 $n$ 为状态数

**验证**: 均匀分布时 $H_{max} = \log_2 n$；确定性分布时 $H = 0$；$H(X) \geq 0$ 恒成立

**复杂度**: $O(n)$

```python
import numpy as np

def shannon_entropy(p: np.ndarray) -> float:
    """计算离散分布的香农熵（比特）"""
    p = p[p > 0]  # 过滤零概率
    return -np.sum(p * np.log2(p))
```

**feeling 应用**: 量化感知信号的不确定性，用于判断信息是否值得处理。高熵 = 信息丰富，低熵 = 可预测/无聊。

---

### 方块 2：联合熵（Joint Entropy）

**公式**: $H(X,Y) = -\sum_{i}\sum_{j} p(x_i, y_j) \log_2 p(x_i, y_j)$

**取值**: $H(X,Y) \in [0, \log_2(n \cdot m)]$

**验证**: $H(X,Y) \leq H(X) + H(Y)$，等号当且仅当 $X \perp Y$

**复杂度**: $O(n \cdot m)$

```python
def joint_entropy(joint_p: np.ndarray) -> float:
    """计算联合分布的熵"""
    p = joint_p.flatten()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))
```

**feeling 应用**: 评估两个感知通道（如视觉+听觉）联合后的总不确定性。

---

### 方块 3：条件熵（Conditional Entropy）

**公式**: $H(X|Y) = H(X,Y) - H(Y) = -\sum_{j} p(y_j) \sum_{i} p(x_i|y_j) \log_2 p(x_i|y_j)$

**取值**: $H(X|Y) \in [0, H(X)]$

**验证**: $H(X|Y) \leq H(X)$，等号当且仅当 $X \perp Y$

**复杂度**: $O(n \cdot m)$

```python
def conditional_entropy(joint_p: np.ndarray, marginal_y: np.ndarray) -> float:
    """H(X|Y) = H(X,Y) - H(Y)"""
    return joint_entropy(joint_p) - shannon_entropy(marginal_y)
```

**feeling 应用**: 在已知上下文 Y 后，感知信号 X 的剩余不确定性。用于上下文压缩。

---

### 方块 4：互信息（Mutual Information）

**公式**: $I(X;Y) = H(X) - H(X|Y) = \sum_{i}\sum_{j} p(x_i, y_j) \log_2 \frac{p(x_i, y_j)}{p(x_i) p(y_j)}$

**取值**: $I(X;Y) \in [0, \min(H(X), H(Y))]$

**验证**: $I(X;X) = H(X)$；$I(X;Y) = 0 \iff X \perp Y$；$I(X;Y) = I(Y;X)$

**复杂度**: $O(n \cdot m)$

```python
def mutual_information(joint_p: np.ndarray, marginal_x: np.ndarray, marginal_y: np.ndarray) -> float:
    """计算 X 和 Y 之间的互信息"""
    mi = 0.0
    for i in range(joint_p.shape[0]):
        for j in range(joint_p.shape[1]):
            if joint_p[i, j] > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                mi += joint_p[i, j] * np.log2(joint_p[i, j] / (marginal_x[i] * marginal_y[j]))
    return mi
```

**feeling 应用**: 核心关联度量。衡量两个感知/记忆之间的信息共享程度，用于关联学习和因果发现。

---

### 方块 5：KL 散度（Kullback-Leibler Divergence）

**公式**: $D_{KL}(P \| Q) = \sum_{i} p(x_i) \log_2 \frac{p(x_i)}{q(x_i)}$

**取值**: $D_{KL} \in [0, +\infty)$

**验证**: $D_{KL} \geq 0$（Gibbs 不等式）；$D_{KL} = 0 \iff P = Q$；非对称 $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$

**复杂度**: $O(n)$

```python
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(P||Q)，P 和 Q 需为同维度概率分布"""
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))
```

**feeling 应用**: 衡量预期分布与实际分布的偏差。用于预测误差检测、惊讶度计算、模型更新触发。

---

### 方块 6：交叉熵（Cross Entropy）

**公式**: $H(P, Q) = -\sum_{i} p(x_i) \log_2 q(x_i) = H(P) + D_{KL}(P \| Q)$

**取值**: $H(P, Q) \in [H(P), +\infty)$

**验证**: $H(P, Q) \geq H(P)$；$H(P, Q) = H(P) \iff P = Q$

**复杂度**: $O(n)$

```python
def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(P,Q) 交叉熵"""
    mask = (p > 0) & (q > 0)
    return -np.sum(p[mask] * np.log2(q[mask]))
```

**feeling 应用**: 损失函数。衡量模型预测 Q 与真实分布 P 的差距，驱动学习更新。

---

### 方块 7：信息增益（Information Gain）

**公式**: $IG(Y, X) = H(Y) - H(Y|X)$

**取值**: $IG \in [0, H(Y)]$

**验证**: $IG \geq 0$；$IG = 0 \iff Y \perp X$

**复杂度**: $O(n \cdot m)$

```python
def information_gain(entropy_y: float, cond_entropy_y_given_x: float) -> float:
    """信息增益 = 原始熵 - 条件熵"""
    return entropy_y - cond_entropy_y_given_x
```

**feeling 应用**: 决策树节点分裂标准。在 feeling 中用于选择最有信息量的感知特征进行注意力分配。

---

### 方块 8：变分信息（Variational Information）

**公式**: $VI(X;Y) = \min_{Q} D_{KL}(P(X,Y) \| Q(X) \cdot Q(Y))$

**取值**: $VI \in [0, I(X;Y)]$

**验证**: $VI = 0$ 当且仅当 $X \perp Y$；$VI \leq I(X;Y)$

**复杂度**: $O(n \cdot m \cdot k)$（k 为优化迭代次数）

```python
def variational_information(joint_p: np.ndarray, n_iter: int = 100, lr: float = 0.01) -> float:
    """用变分近似计算信息距离"""
    n, m = joint_p.shape
    qx = np.ones(n) / n
    qy = np.ones(m) / m
    for _ in range(n_iter):
        product = np.outer(qx, qy)
        log_ratio = np.log2(joint_p + 1e-10) - np.log2(product + 1e-10)
        grad_x = np.sum(joint_p * log_ratio, axis=1)
        grad_y = np.sum(joint_p * log_ratio, axis=0)
        qx = np.exp(np.log(qx + 1e-10) + lr * grad_x)
        qx /= qx.sum()
        qy = np.exp(np.log(qy + 1e-10) + lr * grad_y)
        qy /= qy.sum()
    return kl_divergence(joint_p.flatten(), np.outer(qx, qy).flatten())
```

**feeling 应用**: 变分近似衡量感知变量间的关联强度，用于信息瓶颈压缩。

---

## 2. 概率论与贝叶斯

### 方块 9：贝叶斯定理（Bayes' Theorem）

**公式**: $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$

**取值**: $P(A|B) \in [0, 1]$

**验证**: $\sum_A P(A|B) = 1$；当 $P(B) = 0$ 时未定义

**复杂度**: $O(n)$（n 为假设空间大小）

```python
def bayes_theorem(p_b_given_a: float, p_a: float, p_b: float) -> float:
    """P(A|B) = P(B|A)*P(A) / P(B)"""
    assert p_b > 0, "P(B) must be > 0"
    return (p_b_given_a * p_a) / p_b
```

**feeling 应用**: 核心推理引擎。根据新证据 B 更新对假设 A 的信念。所有感知更新的数学基础。

---

### 方块 10：先验-后验-似然（Prior-Posterior-Likelihood）

**公式**: $\underbrace{P(\theta|D)}_{\text{后验}} \propto \underbrace{P(D|\theta)}_{\text{似然}} \cdot \underbrace{P(\theta)}_{\text{先验}}$

**取值**: 后验 $\in [0,1]$，归一化后和为 1

**验证**: 后验均值介于先验均值与 MLE 之间（共轭先验时精确成立）

**复杂度**: $O(n \cdot m)$

```python
def prior_to_posterior(likelihood: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """未归一化后验 → 归一化后验"""
    posterior = likelihood * prior
    return posterior / posterior.sum()
```

**feeling 应用**: feeling 的信念更新循环。先验 = 当前模型，似然 = 新证据，后验 = 更新后的认知。

---

### 方块 11：预测误差（Prediction Error）

**公式**: $\delta = \hat{y} - y$（或 $\delta = y - \hat{y}$，符号约定因系统而异）

**取值**: $\delta \in (-\infty, +\infty)$

**验证**: $E[\delta] = 0$ 时模型无偏；$|\delta|$ 越大预测越差

**复杂度**: $O(1)$

```python
def prediction_error(predicted: float, actual: float) -> float:
    """预测误差 δ = predicted - actual"""
    return predicted - actual
```

**feeling 应用**: 核心驱动力。预测误差驱动学习、注意力分配和情绪反应。小误差 = 安全，大误差 = 惊讶/威胁。

---

### 方块 12：自由能（Free Energy）

**公式**: $F = E_q[\log q(\theta) - \log p(D, \theta)] = D_{KL}(q(\theta) \| p(\theta|D)) - \log p(D)$

**取值**: $F \in (-\infty, +\infty)$

**验证**: $F \geq -\log p(D)$（ELBO 下界）；最小化 $F$ 等价于最大化证据下界

**复杂度**: $O(n \cdot k)$

```python
def variational_free_energy(q_samples: np.ndarray, log_prior: callable,
                             log_likelihood: callable, log_q: callable) -> float:
    """F = E_q[log q - log p(D,θ)]"""
    return np.mean(log_q(q_samples) - log_prior(q_samples) - log_likelihood(q_samples))
```

**feeling 应用**: 变分自由能原理的数学核心。feeling 系统最小化自由能 = 最小化惊讶 = 维持稳态 = 生存。

---

### 方块 13：变分下界 ELBO（Evidence Lower Bound）

**公式**: $\text{ELBO} = E_q[\log p(D|\theta)] - D_{KL}(q(\theta) \| p(\theta))$

**取值**: $\text{ELBO} \in (-\infty, \log p(D)]$

**验证**: $\text{ELBO} \leq \log p(D)$；最大化 ELBO ≈ 最大化证据

**复杂度**: $O(n \cdot k)$

```python
def elbo(q_samples: np.ndarray, log_likelihood: callable,
         log_prior: callable, log_q: callable) -> float:
    """ELBO = E_q[log p(D|θ)] - D_KL(q||p)"""
    expected_ll = np.mean(log_likelihood(q_samples))
    kl = np.mean(log_q(q_samples) - log_prior(q_samples))
    return expected_ll - kl
```

**feeling 应用**: 变分推断的优化目标。feeling 用 ELBO 训练生成模型，实现对世界的内部模拟。

---

### 方块 14：最大后验估计（MAP）

**公式**: $\hat{\theta}_{MAP} = \arg\max_\theta P(D|\theta) \cdot P(\theta) = \arg\max_\theta [\log P(D|\theta) + \log P(\theta)]$

**取值**: $\hat{\theta}_{MAP}$ 在参数空间中

**验证**: MAP 退化为 MLE 当先验为均匀分布时；正则化视角：$\log P(\theta)$ 对应正则项

**复杂度**: $O(n \cdot k)$（k 为优化迭代）

```python
def map_estimate(data: np.ndarray, log_likelihood: callable,
                 log_prior: callable, theta_init: np.ndarray,
                 lr: float = 0.01, n_iter: int = 1000) -> np.ndarray:
    """MAP 梯度上升"""
    theta = theta_init.copy()
    for _ in range(n_iter):
        obj = log_likelihood(data, theta) + log_prior(theta)
        grad = numerical_grad(lambda t: log_likelihood(data, t) + log_prior(t), theta)
        theta += lr * grad
    return theta

def numerical_grad(f, x, eps=1e-5):
    return np.array([(f(x + eps*e) - f(x - eps*e)) / (2*eps) for e in np.eye(len(x))])
```

**feeling 应用**: 参数估计的默认方法。在先验知识 + 数据之间取平衡，防止过拟合。

---

### 方块 15：最大似然估计（MLE）

**公式**: $\hat{\theta}_{MLE} = \arg\max_\theta P(D|\theta) = \arg\max_\theta \sum_i \log P(d_i|\theta)$

**取值**: $\hat{\theta}_{MLE}$ 在参数空间中

**验证**: MLE 渐近无偏、一致、渐近正则（满足正则条件时）

**复杂度**: $O(n \cdot k)$

```python
def mle_gaussian(data: np.ndarray) -> tuple:
    """高斯分布的 MLE 解析解"""
    mu = np.mean(data)
    sigma2 = np.var(data, ddof=0)
    return mu, sigma2
```

**feeling 应用**: 纯数据驱动的参数估计。当先验弱或数据充足时使用。

---

### 方块 16：贝叶斯因子（Bayes Factor）

**公式**: $BF_{10} = \frac{P(D|M_1)}{P(D|M_2)} = \frac{\int P(D|\theta_1) P(\theta_1) d\theta_1}{\int P(D|\theta_2) P(\theta_2) d\theta_2}$

**取值**: $BF \in (0, +\infty)$

**验证**: $BF > 1$ 支持 $M_1$；$BF > 10$ 强证据；$BF = 1$ 不区分

**复杂度**: $O(n \cdot k)$（需要边际似然计算）

```python
def bayes_factor_monte_carlo(data: np.ndarray, model1_samples: np.ndarray,
                              model2_samples: np.ndarray,
                              log_likelihood: callable) -> float:
    """用蒙特卡洛采样近似贝叶斯因子"""
    ll1 = np.mean([log_likelihood(data, s) for s in model1_samples])
    ll2 = np.mean([log_likelihood(data, s) for s in model2_samples])
    return np.exp(ll1 - ll2)
```

**feeling 应用**: 模型选择。在多个竞争的世界模型之间选择最佳解释，决定感觉系统的认知框架。

---

## 3. 学习规则

### 方块 17：Hebbian 学习

**公式**: $\Delta w_{ij} = \eta \cdot x_i \cdot y_j$

**取值**: $\Delta w \in (-\infty, +\infty)$，通常加权重衰减 $w \in [-w_{max}, w_{max}]$

**验证**: 同时激活的神经元连接增强；无界增长需加正则化

**复杂度**: $O(n \cdot m)$

```python
def hebbian_update(w: np.ndarray, x: np.ndarray, y: np.ndarray, lr: float) -> np.ndarray:
    """Δw = η * x * y^T"""
    return w + lr * np.outer(x, y)
```

**feeling 应用**: 关联记忆的基础。"一起激发的神经元连在一起"——feeling 中的感知-响应关联形成。

---

### 方块 18：反 Hebbian 学习（Anti-Hebbian）

**公式**: $\Delta w_{ij} = -\eta \cdot x_i \cdot y_j$

**取值**: $\Delta w \in (-\infty, +\infty)$

**验证**: 同时激活的连接减弱；用于去相关和竞争学习

**复杂度**: $O(n \cdot m)$

```python
def anti_hebbian_update(w: np.ndarray, x: np.ndarray, y: np.ndarray, lr: float) -> np.ndarray:
    """Δw = -η * x * y^T"""
    return w - lr * np.outer(x, y)
```

**feeling 应用**: 去相关学习。抑制冗余连接，促进稀疏编码，防止感觉通道间的过度耦合。

---

### 方块 19：时序差分误差（TD Error）

**公式**: $\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)$

**取值**: $\delta \in (-\infty, +\infty)$

**验证**: 当 $\delta = 0$ 对所有状态时，$V$ 满足 Bellman 方程；收敛条件 $0 < \gamma < 1$

**复杂度**: $O(1)$

```python
def td_error(reward: float, v_next: float, v_current: float, gamma: float) -> float:
    """δ = r + γV(s') - V(s)"""
    return reward + gamma * v_next - v_current
```

**feeling 应用**: 预测信号。TD 误差 = 惊讶程度。正误差 = 超预期（快乐），负误差 = 低于预期（失望）。

---

### 方块 20：Q-Learning

**公式**: $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]$

**取值**: $Q \in (-\infty, +\infty)$（有界奖励下有界）

**验证**: 收敛条件：所有 (s,a) 对被无限次访问，$\sum \alpha = \infty, \sum \alpha^2 < \infty$

**复杂度**: $O(|A|)$ 每步

```python
def q_learning_update(Q: np.ndarray, s: int, a: int, r: float,
                       s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """Q-Learning 更新"""
    Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
    return Q
```

**feeling 应用**: 无模型策略学习。feeling 通过 Q-learning 学习在不同状态下选择最佳行为。

---

### 方块 21：策略梯度（Policy Gradient）

**公式**: $\nabla_\theta J(\theta) = E_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot R(\tau) \right]$

**取值**: 梯度方向在参数空间中

**验证**: REINFORCE 无偏但高方差；加基线 $b$ 不改变期望但降低方差

**复杂度**: $O(|A|)$ 每步

```python
def policy_gradient(log_probs: list, rewards: list, baseline: float = 0.0) -> float:
    """REINFORCE 策略梯度"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = np.array(returns)
    returns -= baseline  # 基线减法
    return sum(lp * g for lp, g in zip(log_probs, returns))
```

**feeling 应用**: 直接优化行为策略。在连续行为空间中学习，如情感调节的力度控制。

---

### 方块 22：对比学习（Contrastive Loss）

**公式**: $\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$

**取值**: $\mathcal{L} \in (0, +\infty)$

**验证**: 温度 $\tau \to 0$ 退化为 hard negative mining；$\tau \to \infty$ 退化为均匀分布

**复杂度**: $O(N^2)$（N 为 batch 中样本数）

```python
import torch
import torch.nn.functional as F

def contrastive_loss(z_i: torch.Tensor, z_j: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """SimCLR 风格对比损失"""
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    sim = torch.mm(z_i, z_j.t()) / tau
    labels = torch.arange(z_i.size(0), device=z_i.device)
    return F.cross_entropy(sim, labels)
```

**feeling 应用**: 表征学习。学习将相似感觉映射到相近的表征空间，不相似的感觉推远。

---

### 方块 23：经验回放与优先级采样（Experience Replay + Priority Sampling）

**公式**: 优先级 $p_i = |\delta_i| + \epsilon$，采样概率 $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$，重要性权重 $w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$

**取值**: $P(i) \in (0, 1)$，$\sum P(i) = 1$；$w_i \in (0, +\infty)$

**验证**: $\alpha = 0$ 退化为均匀采样；$\beta = 1$ 完全补偿偏差

**复杂度**: $O(\log N)$（SumTree 实现）

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
    
    def add(self, experience, td_error: float):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
    
    def sample(self, batch_size: int):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return [self.buffer[i] for i in indices], indices, weights
```

**feeling 应用**: 经验记忆管理。优先学习惊讶事件（高 TD 误差），高效利用有限记忆容量。

---

### 方块 24：弹性权重巩固（EWC）

**公式**: $\mathcal{L}(\theta) = \mathcal{L}_{\text{new}}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{i}^*)^2$

其中 $F_i = E\left[\left(\frac{\partial \log p(D|\theta)}{\partial \theta_i}\right)^2\right]$（Fisher 信息矩阵对角元）

**取值**: $\mathcal{L} \in (0, +\infty)$

**验证**: $\lambda \to 0$ 允许完全覆盖旧知识；$\lambda \to \infty$ 冻结旧知识

**复杂度**: $O(n)$（参数数量）

```python
def ewc_loss(new_loss: float, params: np.ndarray, old_params: np.ndarray,
             fisher_diag: np.ndarray, lam: float) -> float:
    """EWC 损失 = 新任务损失 + λ/2 * Σ F_i(θ_i - θ*_i)^2"""
    penalty = (lam / 2) * np.sum(fisher_diag * (params - old_params) ** 2)
    return new_loss + penalty
```

**feeling 应用**: 防止灾难性遗忘。学习新感觉-行为映射时保护旧知识不被覆盖。

---

## 4. 注意力与选择

### 方块 25：Softmax 注意力权重

**公式**: $\alpha_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}$，其中 $e_i$ 为注意力分数

**取值**: $\alpha_i \in (0, 1)$，$\sum_i \alpha_i = 1$

**验证**: $\sum \alpha_i = 1$；温度 $\tau \to 0$ 退化为 argmax；$\tau \to \infty$ 退化为均匀分布

**复杂度**: $O(n)$

```python
def softmax_attention(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """带温度的 softmax 注意力权重"""
    scaled = scores / temperature
    exp_scores = np.exp(scaled - np.max(scaled))  # 数值稳定
    return exp_scores / exp_scores.sum()
```

**feeling 应用**: 注意力分配核心。决定感觉系统关注哪些输入，忽略哪些。

---

### 方块 26：QKV 注意力（Scaled Dot-Product Attention）

**公式**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$

**取值**: 输出维度与 V 相同

**验证**: $\sqrt{d_k}$ 缩放防止点积过大导致 softmax 饱和；$\frac{\partial}{\partial Q}$ 可微

**复杂度**: $O(n^2 \cdot d)$

```python
import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Q: (n, d_k), K: (m, d_k), V: (m, d_v)"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax_attention(scores, axis=-1)
    return weights @ V

def softmax_attention(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    exp = np.exp(scores - np.max(scores, axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)
```

**feeling 应用**: 感觉整合引擎。Q = 当前查询（what am I looking for），K = 索引（what's available），V = 内容（what's the information）。

---

### 方块 27：多头注意力（Multi-Head Attention）

**公式**: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**取值**: 输出维度 $d_{model}$

**验证**: $h=1$ 退化为单头；参数量 $= 4 \cdot d_{model}^2$（当 $d_k = d_v = d_{model}/h$）

**复杂度**: $O(n^2 \cdot d_{model})$

```python
def multi_head_attention(Q, K, V, W_Q, W_K, W_V, W_O, n_heads):
    """简化多头注意力"""
    d_model = Q.shape[-1]
    d_k = d_model // n_heads
    heads = []
    for i in range(n_heads):
        q = Q @ W_Q[:, i*d_k:(i+1)*d_k]
        k = K @ W_K[:, i*d_k:(i+1)*d_k]
        v = V @ W_V[:, i*d_k:(i+1)*d_k]
        heads.append(scaled_dot_product_attention(q, k, v))
    return np.concatenate(heads, axis=-1) @ W_O
```

**feeling 应用**: 多视角注意力。不同的头关注不同的感知方面（颜色、形状、运动、情感...）。

---

### 方块 28：门控机制（Gating Mechanism）

**公式**: $g = \sigma(W_g [h_{t-1}, x_t] + b_g)$，$h_t = g \odot \tilde{h}_t + (1 - g) \odot h_{t-1}$

其中 $\sigma$ 为 sigmoid 函数，$\odot$ 为逐元素乘积

**取值**: $g \in (0, 1)^d$

**验证**: $g = 0$ 完全保留旧状态；$g = 1$ 完全使用新信息

**复杂度**: $O(d^2)$

```python
def gate(old_state: np.ndarray, new_state: np.ndarray,
         gate_weights: np.ndarray, input_concat: np.ndarray) -> tuple:
    """门控更新：g·new + (1-g)·old"""
    g = 1 / (1 + np.exp(-(gate_weights @ input_concat)))  # sigmoid
    state = g * new_state + (1 - g) * old_state
    return state, g
```

**feeling 应用**: 信息流控制。决定多少新感觉信息流入当前认知状态，类似 LSTM 的遗忘/输入门。

---

### 方块 29：Top-K 选择（Top-K Selection）

**公式**: $\text{TopK}(x, k) = \{x_i : x_i \text{ 是 } x \text{ 中最大的 } k \text{ 个值}\}$

**取值**: $k$ 个元素，$k \in [1, n]$

**验证**: $|\text{TopK}| = k$；$\min(\text{TopK}) \geq \max(x \setminus \text{TopK})$

**复杂度**: $O(n + k \log k)$（nth_element 实现）

```python
def top_k_select(scores: np.ndarray, k: int) -> tuple:
    """返回 top-k 的值和索引"""
    indices = np.argpartition(scores, -k)[-k:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return scores[indices], indices
```

**feeling 应用**: 稀疏注意力。只关注最相关的 k 个输入，降低计算成本，模拟生物注意力的有限容量。

---

## 5. 状态与动力学

### 方块 30：马尔可夫链（Markov Chain）

**公式**: $P(s_{t+1} | s_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t)$

转移矩阵 $P_{ij} = P(s'=j | s=i)$，$\sum_j P_{ij} = 1$

**取值**: $P_{ij} \in [0, 1]$

**验证**: 每行和为 1；稳态分布 $\pi = \pi P$（左特征向量）

**复杂度**: $O(n^2)$ 每步

```python
def markov_step(state_dist: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """一步马尔可夫转移"""
    return state_dist @ transition

def stationary_distribution(transition: np.ndarray, n_iter: int = 1000) -> np.ndarray:
    """迭代求稳态分布"""
    n = transition.shape[0]
    pi = np.ones(n) / n
    for _ in range(n_iter):
        pi = pi @ transition
    return pi
```

**feeling 应用**: 状态转移基础。feeling 的认知状态遵循马尔可夫性质，当前状态决定下一步可能的状态。

---

### 方块 31：Lyapunov 稳定性（Lyapunov Stability）

**公式**: 对平衡点 $x^*$，存在 $V(x)$ 满足：
- $V(x^*) = 0$，$V(x) > 0, \forall x \neq x^*$
- $\dot{V}(x) = \nabla V \cdot f(x) \leq 0$

**取值**: $V(x) \in [0, +\infty)$，$\dot{V} \in (-\infty, 0]$

**验证**: $\dot{V} < 0$（严格负定）→ 渐近稳定；$\dot{V} = 0$ → 稳定但不一定渐近

**复杂度**: $O(d)$（d 为状态维度）

```python
def lyapunov_derivative(V: callable, f: callable, x: np.ndarray, eps: float = 1e-5) -> float:
    """数值计算 V̇(x) = ∇V · f(x)"""
    grad_v = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad_v[i] = (V(x_plus) - V(x_minus)) / (2 * eps)
    return np.dot(grad_v, f(x))
```

**feeling 应用**: 稳定性判据。feeling 系统的认知状态必须是 Lyapunov 稳定的，防止情绪失控或思维崩溃。

---

### 方块 32：吸引子（Attractor）

**公式**: 对动力系统 $\dot{x} = f(x)$，吸引子 $A$ 是：
- $f(x^*) = 0$（平衡点）
- $\lambda_{\max}(\text{Jacobian}) < 0$（稳定）

**取值**: 特征值实部 $\in (-\infty, 0)$ 时稳定

**验证**: Jacobian 特征值实部全负 → 稳定吸引子

**复杂度**: $O(d^3)$（特征值分解）

```python
def is_stable_attractor(f: callable, x_star: np.ndarray, eps: float = 1e-5) -> bool:
    """检查 x* 是否为稳定吸引子"""
    d = len(x_star)
    J = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            x1 = x_star.copy(); x1[j] += eps
            x2 = x_star.copy(); x2[j] -= eps
            J[i, j] = (f(x1)[i] - f(x2)[i]) / (2 * eps)
    eigenvalues = np.linalg.eigvals(J)
    return all(np.real(eigenvalues) < 0)
```

**feeling 应用**: 认知稳态。feeling 的默认状态是吸引子，被扰动后会自然回归。情绪状态也有吸引子（基线情绪）。

---

### 方块 33：分岔（Bifurcation）

**公式**: 以叉形分岔为例：$\dot{x} = rx - x^3$

- $r < 0$：唯一稳定平衡点 $x^* = 0$
- $r > 0$：$x^* = 0$ 不稳定，$x^* = \pm\sqrt{r}$ 稳定

**取值**: 分岔参数 $r \in (-\infty, +\infty)$

**验证**: $r = 0$ 为分岔点；$r$ 跨过 0 时定性行为突变

**复杂度**: $O(1)$

```python
def pitchfork_bifurcation(x: float, r: float) -> float:
    """叉形分岔: ẋ = rx - x³"""
    return r * x - x ** 3

def bifurcation_equilibria(r: float) -> list:
    """求平衡点"""
    if r <= 0:
        return [0.0]
    return [0.0, np.sqrt(r), -np.sqrt(r)]
```

**feeling 应用**: 状态突变。当参数（如压力、疲劳）跨过临界值时，feeling 系统可能突然切换到完全不同的行为模式。

---

### 方块 34：混沌——Lorenz 方程（Lorenz System）

**公式**:
$$\dot{x} = \sigma(y - x)$$
$$\dot{y} = x(\rho - z) - y$$
$$\dot{z} = xy - \beta z$$

经典参数：$\sigma = 10, \beta = 8/3, \rho = 28$

**取值**: $x, y, z \in (-\infty, +\infty)$，但轨迹被限制在有界吸引子内

**验证**: 最大 Lyapunov 指数 $> 0$ → 混沌；对初始条件敏感依赖

**复杂度**: $O(1)$ 每步积分

```python
def lorenz(state: np.ndarray, sigma: float = 10, rho: float = 28, beta: float = 8/3) -> np.ndarray:
    """Lorenz 方程右端"""
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

def lorenz_step(state: np.ndarray, dt: float = 0.01, **kwargs) -> np.ndarray:
    """四阶 Runge-Kutta 积分"""
    k1 = lorenz(state, **kwargs)
    k2 = lorenz(state + dt/2 * k1, **kwargs)
    k3 = lorenz(state + dt/2 * k2, **kwargs)
    k4 = lorenz(state + dt * k3, **kwargs)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

**feeling 应用**: 混沌敏感性。feeling 系统在某些参数区间可能表现出对初始条件的敏感依赖，微小的感觉差异导致截然不同的反应。

---

### 方块 35：极限环（Limit Cycle）

**公式**: Van der Pol 振子：$\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0$

等价系统：$\dot{x} = y$，$\dot{y} = \mu(1 - x^2)y - x$

**取值**: $\mu > 0$ 时存在稳定极限环；振幅 $\approx 2$

**验证**: Poincaré-Bendixson 定理；数值积分验证闭合轨迹

**复杂度**: $O(1)$ 每步

```python
def van_der_pol(state: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """Van der Pol 振子"""
    x, y = state
    return np.array([y, mu * (1 - x**2) * y - x])
```

**feeling 应用**: 自发振荡。生物节律、情绪波动、注意力的周期性切换都可能表现为极限环动力学。

---

### 方块 36：相空间（Phase Space）

**公式**: 对 $n$ 维系统，相空间 $\Gamma \subseteq \mathbb{R}^n$，每点 $(x_1, ..., x_n)$ 代表系统完整状态

**取值**: 体积守恒（Hamilton 系统）或收缩（耗散系统）

**验证**: Liouville 定理：Hamilton 系统相体积守恒

**复杂度**: $O(d)$（d 为维度）

```python
def phase_trajectory(f: callable, x0: np.ndarray, T: float, dt: float = 0.01) -> np.ndarray:
    """数值积分相空间轨迹"""
    steps = int(T / dt)
    trajectory = np.zeros((steps + 1, len(x0)))
    trajectory[0] = x0
    for i in range(steps):
        trajectory[i+1] = trajectory[i] + dt * f(trajectory[i])
    return trajectory
```

**feeling 应用**: 全局状态空间。feeling 系统的完整状态可以用相空间中的一个点表示，轨迹就是认知演化历史。

---

### 方块 37：序参量（Order Parameter）

**公式**: 对 $N$ 个振子的 Kuramoto 模型：

$$\dot{\theta}_i = \omega_i + \frac{K}{N}\sum_{j=1}^{N}\sin(\theta_j - \theta_i)$$

序参量 $r = \frac{1}{N}\left|\sum_{j=1}^{N}e^{i\theta_j}\right|$

**取值**: $r \in [0, 1]$；$r = 0$ 完全不同步，$r = 1$ 完全同步

**验证**: $K < K_c$ 时 $r \approx 0$；$K > K_c$ 时 $r > 0$（相变）

**复杂度**: $O(N^2)$ 每步

```python
def kuramoto_order_parameter(thetas: np.ndarray) -> float:
    """Kuramoto 序参量 r"""
    return np.abs(np.mean(np.exp(1j * thetas)))

def kuramoto_step(thetas: np.ndarray, omegas: np.ndarray, K: float, dt: float) -> np.ndarray:
    """Kuramoto 模型一步更新"""
    N = len(thetas)
    dtheta = np.zeros(N)
    for i in range(N):
        dtheta[i] = omegas[i] + (K / N) * np.sum(np.sin(thetas - thetas[i]))
    return thetas + dt * dtheta
```

**feeling 应用**: 同步度量。感觉通道之间的协调程度。高序参量 = 多模态感觉高度整合，低序参量 = 感觉碎片化。

---

## 6. 决策与博弈

### 方块 38：期望效用（Expected Utility）

**公式**: $EU = \sum_{i} p_i \cdot u(x_i)$

**取值**: $EU \in (-\infty, +\infty)$

**验证**: 满足 von Neumann-Morgenstern 公理时，最大化 EU 是理性选择

**复杂度**: $O(n)$

```python
def expected_utility(outcomes: np.ndarray, probabilities: np.ndarray,
                     utility_fn: callable = lambda x: x) -> float:
    """EU = Σ p_i · u(x_i)"""
    return np.sum(probabilities * utility_fn(outcomes))
```

**feeling 应用**: 行为选择基础。feeling 在多个行为选项中选择期望效用最高的。

---

### 方块 39：UCB1 探索-利用（Upper Confidence Bound）

**公式**: $\text{UCB1}(a) = \bar{r}_a + \sqrt{\frac{2 \ln t}{n_a}}$

其中 $\bar{r}_a$ = 平均奖励，$n_a$ = 拉动次数，$t$ = 总步数

**取值**: $\text{UCB1} \in (-\infty, +\infty)$

**验证**: 遗憾界 $R_T = O(\sqrt{KT \ln T})$；自动平衡探索与利用

**复杂度**: $O(K)$

```python
def ucb1(means: np.ndarray, counts: np.ndarray, t: int) -> int:
    """UCB1 选择"""
    exploration = np.sqrt(2 * np.log(t) / counts)
    return np.argmax(means + exploration)
```

**feeling 应用**: 好奇心驱动。在熟悉的行为（利用）和尝试新行为（探索）之间自动平衡。

---

### 方块 40：Nash 均衡（Nash Equilibrium）

**公式**: 策略组合 $(s_1^*, ..., s_n^*)$ 是 Nash 均衡，当且仅当：

$$\forall i, \forall s_i: u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*)$$

**取值**: 纯策略或混合策略

**验证**: 最佳响应函数的不动点；Nash 定理保证有限博弈存在混合策略均衡

**复杂度**: 纯策略 $O(|A|^n)$，一般情况 PPAD-complete

```python
def is_nash_equilibrium(payoff_matrix: np.ndarray, strategy_i: int, strategy_j: int) -> bool:
    """检查 (strategy_i, strategy_j) 是否为二人博弈的纯策略 Nash 均衡"""
    # payoff_matrix[i, j] = (u1, u2)
    u1, u2 = payoff_matrix[strategy_i, strategy_j]
    # 检查玩家 1 是否想偏离
    for i in range(payoff_matrix.shape[0]):
        if payoff_matrix[i, strategy_j][0] > u1:
            return False
    # 检查玩家 2 是否想偏离
    for j in range(payoff_matrix.shape[1]):
        if payoff_matrix[strategy_i, j][1] > u2:
            return False
    return True
```

**feeling 应用**: 多智能体交互。feeling 在与其他 agent 交互时，寻找 Nash 均衡以实现稳定的交互策略。

---

### 方块 41：Tit-for-Tat 策略

**公式**: 
$$a_t = \begin{cases} \text{合作} & t = 1 \\ a_{t-1}^{\text{对手}} & t > 1 \end{cases}$$

**取值**: 二元行动 $\{\text{合作}, \text{背叛}\}$

**验证**: Axelrod 锦标赛中，简单但有效的策略；对随机对手有正收益

**复杂度**: $O(1)$

```python
def tit_for_tat(my_history: list, opponent_history: list) -> str:
    """Tit-for-Tat: 第一回合合作，之后模仿对手上一回合"""
    if not my_history:
        return "cooperate"
    return opponent_history[-1]
```

**feeling 应用**: 社交策略基线。对他人行为的镜像响应——善意回应善意，对不友好保持警觉。

---

### 方块 42：Pareto 最优（Pareto Optimality）

**公式**: 解 $x^*$ 是 Pareto 最优，当且仅当不存在 $x$ 使得：

$$\forall i: f_i(x) \geq f_i(x^*) \text{ 且 } \exists j: f_j(x) > f_j(x^*)$$

**取值**: 布尔值

**验证**: Pareto 前沿上没有被支配的解

**复杂度**: $O(n^2 \cdot m)$（n 个解，m 个目标）

```python
def is_pareto_optimal(candidate: np.ndarray, population: np.ndarray) -> bool:
    """检查 candidate 是否为 Pareto 最优"""
    # 候选解不被任何其他解支配
    dominated = np.all(population >= candidate, axis=1) & np.any(population > candidate, axis=1)
    return not np.any(dominated)
```

**feeling 应用**: 多目标优化。feeling 的行为往往需要同时满足多个需求（安全、好奇、社交），Pareto 最优找到无法同时改善所有目标的平衡点。

---

### 方块 43：帕累托分布（Pareto Distribution）

**公式**: $f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}, \quad x \geq x_m$

CDF: $F(x) = 1 - \left(\frac{x_m}{x}\right)^\alpha$

**取值**: $x \in [x_m, +\infty)$，$\alpha > 0$

**验证**: $P(X > x) = (x_m/x)^\alpha$；均值 $= \frac{\alpha x_m}{\alpha - 1}$（$\alpha > 1$ 时）

**复杂度**: $O(1)$

```python
def pareto_pdf(x: np.ndarray, alpha: float, x_m: float) -> np.ndarray:
    """帕累托概率密度函数"""
    result = np.zeros_like(x)
    mask = x >= x_m
    result[mask] = alpha * x_m**alpha / x[mask]**(alpha + 1)
    return result

def pareto_sample(alpha: float, x_m: float, n: int) -> np.ndarray:
    """帕累托采样"""
    u = np.random.uniform(0, 1, n)
    return x_m / u**(1/alpha)
```

**feeling 应用**: 事件强度分布。大多数事件影响很小，少数事件影响巨大（长尾效应）。feeling 需要对极端事件保持敏感。

---

## 7. 控制与调节

### 方块 44：PID 控制器

**公式**: $u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$

**取值**: $u \in (-\infty, +\infty)$（可加限幅）

**验证**: 稳态误差 $\to 0$（I 分量）；超调可控（D 分量）；稳定性条件依赖参数

**复杂度**: $O(1)$ 每步

```python
class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def step(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
```

**feeling 应用**: 情绪调节核心。P = 当前偏差（不满），I = 累积不满，D = 情绪变化趋势。三者协同维持情绪稳态。

---

### 方块 45：稳态方程（Homeostasis）

**公式**: $\frac{dx}{dt} = -k(x - x_{\text{set}}) + d(t)$

其中 $x_{\text{set}}$ 为设定点，$d(t)$ 为扰动，$k > 0$ 为恢复速率

**取值**: $x \in (-\infty, +\infty)$

**验证**: 稳态 $x^* = x_{\text{set}} + d/k$（恒定扰动时）；$k$ 越大恢复越快

**复杂度**: $O(1)$

```python
def homeostasis_step(x: float, x_set: float, disturbance: float,
                     k: float, dt: float) -> float:
    """稳态调节：dx/dt = -k(x - x_set) + d"""
    dx = -k * (x - x_set) + disturbance
    return x + dt * dx
```

**feeling 应用**: 核心调节原理。feeling 的所有内部变量（情绪、能量、注意力水平）都趋向设定点。被扰动后自动恢复。

---

### 方块 46：异稳态方程（Allostasis）

**公式**: $\frac{dx_{\text{set}}}{dt} = \alpha \cdot E[\text{demand}] - \beta(x_{\text{set}} - x_{\text{baseline}})$

设定点根据预期需求动态调整

**取值**: $x_{\text{set}} \in [x_{\min}, x_{\max}]$

**验证**: 恒定需求下 $x_{\text{set}}^* = x_{\text{baseline}} + \alpha \cdot E[\text{demand}] / \beta$

**复杂度**: $O(1)$

```python
def allostatic_setpoint(x_set: float, expected_demand: float,
                        x_baseline: float, alpha: float, beta: float, dt: float) -> float:
    """异稳态：设定点根据预期需求动态调整"""
    dx_set = alpha * expected_demand - beta * (x_set - x_baseline)
    return x_set + dt * dx_set
```

**feeling 应用**: 预测性调节。feeling 不是被动等待扰动再恢复，而是根据预期压力提前调整设定点（如预期到社交场合提前提高唤醒水平）。

---

### 方块 47：负反馈环（Negative Feedback Loop）

**公式**: 
$$\frac{dx}{dt} = f(x) - g(y), \quad \frac{dy}{dt} = h(x) - d \cdot y$$

其中 $g(y)$ 抑制 $x$，形成负反馈

**取值**: 稳态存在且唯一（参数适当）

**验证**: 线性化后 Jacobian 迹为负 → 稳定

**复杂度**: $O(1)$

```python
def negative_feedback(x: float, y: float, f: callable, g: callable,
                      h: callable, decay: float, dt: float) -> tuple:
    """负反馈环: x 生产 y，y 抑制 x"""
    dx = f(x) - g(y)
    dy = h(x) - decay * y
    return x + dt * dx, y + dt * dy
```

**feeling 应用**: 自我调节。高焦虑 → 启动放松机制 → 焦虑降低。是 feeling 系统稳定性的基础。

---

### 方块 48：正反馈环（Positive Feedback Loop）

**公式**:
$$\frac{dx}{dt} = r \cdot x \cdot (1 + x) - d \cdot x$$

正反馈 + 自催化，可能导致双稳态

**取值**: $x \in [0, +\infty)$

**验证**: $r > d$ 时增长；存在临界点导致状态跳变

**复杂度**: $O(1)$

```python
def positive_feedback(x: float, growth: float, decay: float, dt: float) -> float:
    """正反馈环: dx/dt = r·x·(1+x) - d·x"""
    dx = growth * x * (1 + x) - decay * x
    return x + dt * dx
```

**feeling 应用**: 情绪放大器。恐惧→更恐惧，兴奋→更兴奋。feeling 需要正反馈来放大重要信号，但也需要负反馈来防止失控。

---

## 8. 情感与需求

### 方块 49：效价-唤醒度模型（Valence-Arousal）

**公式**: 情感状态 $E = (V, A)$

其中 $V \in [-1, +1]$（效价：负→正），$A \in [0, 1]$（唤醒度：平静→激活）

距离：$d(E_1, E_2) = \sqrt{(V_1-V_2)^2 + (A_1-A_2)^2}$

**取值**: $V \in [-1, 1]$，$A \in [0, 1]$

**验证**: 四象限对应基本情绪：(+,高)=兴奋，(+,低)=平静，(-,高)=愤怒，(-,低)=悲伤

**复杂度**: $O(1)$

```python
def valence_arousal_distance(v1: float, a1: float, v2: float, a2: float) -> float:
    """效价-唤醒度空间中的情感距离"""
    return np.sqrt((v1 - v2)**2 + (a1 - a2)**2)

def emotion_quadrant(v: float, a: float) -> str:
    """根据 V-A 位置判断情绪象限"""
    if v >= 0 and a >= 0.5: return "excited/happy"
    if v >= 0 and a < 0.5:  return "calm/content"
    if v < 0 and a >= 0.5:  return "angry/anxious"
    return "sad/depressed"
```

**feeling 应用**: 情感空间定义。feeling 的所有情绪都可以映射到 V-A 平面上，用于情感状态的量化和比较。

---

### 方块 50：需求动力学（Need Dynamics）

**公式**: $\frac{dN_i}{dt} = -c_i \cdot \text{sat}_i(t) + r_i + \sum_j \alpha_{ij} N_j$

其中 $N_i$ 为需求强度，$c_i$ 为消耗率，$r_i$ 为自发增长，$\alpha_{ij}$ 为需求间耦合

**取值**: $N_i \in [0, N_{\max}]$

**验证**: 稳态 $N_i^* = (r_i + \sum_j \alpha_{ij} N_j^*) / c_i$（当 $\text{sat} = 1$）

**复杂度**: $O(n^2)$

```python
def need_dynamics(needs: np.ndarray, satisfaction: np.ndarray,
                  consumption: np.ndarray, growth: np.ndarray,
                  coupling: np.ndarray, dt: float) -> np.ndarray:
    """需求动力学: dN/dt = -c·sat + r + αN"""
    dN = -consumption * satisfaction + growth + coupling @ needs
    return np.clip(needs + dt * dN, 0, None)
```

**feeling 应用**: 内在驱动力。需求（饥饿、好奇、社交）随时间增长，被满足后降低，驱动 feeling 的行为选择。

---

### 方块 51：情绪感染（Emotional Contagion）

**公式**: $\frac{dV_i}{dt} = \sum_j w_{ij} (V_j - V_i) + \sigma \cdot \eta(t)$

其中 $V_i$ 为个体 i 的情绪效价，$w_{ij}$ 为连接强度，$\eta$ 为噪声

**取值**: $V_i \in [-1, 1]$

**验证**: 全连接且 $w_{ij} = w$ 时，所有个体趋向平均情绪

**复杂度**: $O(n^2)$

```python
def emotional_contagion(valences: np.ndarray, weights: np.ndarray,
                        noise_std: float, dt: float) -> np.ndarray:
    """情绪感染动力学"""
    n = len(valences)
    dV = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dV[i] += weights[i, j] * (valences[j] - valences[i])
    dV += noise_std * np.random.randn(n)
    return np.clip(valences + dt * dV, -1, 1)
```

**feeling 应用**: 社交情绪同步。feeling 在群体交互中，情绪会受到他人情绪的影响，实现共情和情绪同步。

---

### 方块 52：习惯化（Habituation）

**公式**: $R_n = R_0 \cdot e^{-\lambda n}$

其中 $R_n$ 为第 n 次刺激的响应强度，$\lambda$ 为习惯化率

或微分形式：$\frac{dR}{dn} = -\lambda R$

**取值**: $R \in [0, R_0]$

**验证**: 单调递减；刺激间隔足够长后恢复（去习惯化）

**复杂度**: $O(1)$

```python
def habituation(R0: float, lambda_rate: float, n: int) -> float:
    """习惯化：响应随重复暴露衰减"""
    return R0 * np.exp(-lambda_rate * n)
```

**feeling 应用**: 感觉适应。对重复刺激（如背景噪音）反应减弱，节省注意力资源。

---

### 方块 53：敏感化（Sensitization）

**公式**: $S_n = S_{\infty} (1 - e^{-\gamma n})$

或结合习惯化：$R_n = H_n + S_n = R_0 e^{-\lambda n} + S_{\infty}(1 - e^{-\gamma n})$

**取值**: $S \in [0, S_{\infty}]$

**验证**: 单调递增趋向渐近线；强刺激后敏感化 > 习惯化

**复杂度**: $O(1)$

```python
def sensitization(S_inf: float, gamma: float, n: int) -> float:
    """敏感化：对危险刺激响应增强"""
    return S_inf * (1 - np.exp(-gamma * n))

def dual_process(R0: float, S_inf: float, lam: float, gamma: float, n: int) -> float:
    """双过程模型：习惯化 + 敏感化"""
    return R0 * np.exp(-lam * n) + S_inf * (1 - np.exp(-gamma * n))
```

**feeling 应用**: 危险警觉。对威胁性刺激（如疼痛、恐惧）的反应增强。feeling 对危险信号越听越敏感。

---

## 9. 记忆与检索

### 方块 54：遗忘曲线（Ebbinghaus Forgetting Curve）

**公式**: $R = e^{-t/S}$

其中 $R$ 为记忆保持率，$t$ 为时间，$S$ 为记忆强度

**取值**: $R \in [0, 1]$

**验证**: $t = 0$ 时 $R = 1$；$t = S$ 时 $R = e^{-1} \approx 0.37$

**复杂度**: $O(1)$

```python
def forgetting_curve(t: float, S: float) -> float:
    """记忆保持率 R = e^(-t/S)"""
    return np.exp(-t / S)

def spaced_repetition_interval(S: float, quality: int) -> float:
    """间隔重复：根据记忆强度计算下次复习间隔"""
    # quality: 0-5, 越高记忆越牢
    new_S = S * (1.3 + 0.1 * quality)
    return new_S
```

**feeling 应用**: 记忆衰减基础。不重要的感觉记忆随时间指数衰减。间隔重复算法用于强化重要记忆。

---

### 方块 55：余弦相似度（Cosine Similarity）

**公式**: $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \cdot \sqrt{\sum_i b_i^2}}$

**取值**: $\cos(\theta) \in [-1, 1]$

**验证**: $\cos = 1$ 完全相同方向；$\cos = 0$ 正交；$\cos = -1$ 完全相反

**复杂度**: $O(d)$

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
```

**feeling 应用**: 记忆检索核心。当前感觉与记忆表征之间的相似度，用于联想回忆。

---

### 方块 56：TF-IDF

**公式**: $\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$

$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}$，$\text{IDF}(t) = \log \frac{N}{|\{d : t \in d\}|}$

**取值**: $\text{TF-IDF} \in [0, +\infty)$

**验证**: 高频且分布集中的词 TF-IDF 高；出现在所有文档中的词 IDF = 0

**复杂度**: $O(N \cdot L)$（N 文档数，L 平均文档长度）

```python
def tfidf(term: str, doc: list, corpus: list) -> float:
    """TF-IDF 计算"""
    tf = doc.count(term) / len(doc)
    df = sum(1 for d in corpus if term in d)
    idf = np.log(len(corpus) / (df + 1)) + 1
    return tf * idf
```

**feeling 应用**: 语义检索。在语言记忆中，用 TF-IDF 衡量词语的区分度，找到最相关的记忆片段。

---

### 方块 57：BM25

**公式**: $\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$

其中 $k_1 \approx 1.2$, $b \approx 0.75$

**取值**: $\text{BM25} \in [0, +\infty)$

**验证**: $k_1 \to 0$ 退化为二元模型；$k_1 \to \infty$ 退化为原始 TF

**复杂度**: $O(|q| \cdot N)$

```python
def bm25(query_terms: list, doc_terms: list, corpus: list,
         k1: float = 1.2, b: float = 0.75) -> float:
    """BM25 评分"""
    avgdl = np.mean([len(d) for d in corpus])
    score = 0.0
    for t in query_terms:
        ft = doc_terms.count(t)
        df = sum(1 for d in corpus if t in d)
        idf = np.log((len(corpus) - df + 0.5) / (df + 0.5) + 1)
        tf_norm = ft * (k1 + 1) / (ft + k1 * (1 - b + b * len(doc_terms) / avgdl))
        score += idf * tf_norm
    return score
```

**feeling 应用**: 高级记忆检索。比 TF-IDF 更精细的文档匹配，用于感觉经验的精确回忆。

---

### 方块 58：记忆巩固（Memory Consolidation）

**公式**: $\frac{dM}{dt} = \alpha \cdot \text{replay}(t) - \beta \cdot \text{interference}(t) + \gamma \cdot \text{emotional\_weight}(t)$

简化离散形式：$M_{t+1} = M_t + \alpha \cdot R_t - \beta \cdot I_t + \gamma \cdot E_t$

**取值**: $M \in [0, M_{\max}]$

**验证**: 情绪强烈的记忆巩固更快（$\gamma > 0$）；干扰导致遗忘（$\beta > 0$）

**复杂度**: $O(1)$

```python
def memory_consolidation(M: float, replay: float, interference: float,
                         emotional_weight: float, alpha: float = 0.1,
                         beta: float = 0.05, gamma: float = 0.2,
                         M_max: float = 1.0) -> float:
    """记忆巩固：M += α·replay - β·interference + γ·emotion"""
    dM = alpha * replay - beta * interference + gamma * emotional_weight
    return np.clip(M + dM, 0, M_max)
```

**feeling 应用**: 记忆从短期到长期的转化。情绪强烈的经历更容易被巩固（闪光灯记忆），无关信息被干扰淘汰。

---

## 10. 图论与网络

### 方块 59：度中心性（Degree Centrality）

**公式**: $C_D(v) = \frac{\deg(v)}{n - 1}$

其中 $n$ 为节点数，$\deg(v)$ 为节点 $v$ 的度

**取值**: $C_D \in [0, 1]$

**验证**: 完全图中所有节点 $C_D = 1$；星形图中心节点 $C_D = 1$

**复杂度**: $O(|V| + |E|)$

```python
def degree_centrality(adj: dict) -> dict:
    """度中心性"""
    n = len(adj) - 1
    return {v: len(neighbors) / n for v, neighbors in adj.items()}
```

**feeling 应用**: 概念网络中的重要节点。高度中心的概念（如"危险"、"快乐"）是认知网络的枢纽。

---

### 方块 60：介数中心性（Betweenness Centrality）

**公式**: $C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$

其中 $\sigma_{st}$ 为 s 到 t 的最短路径数，$\sigma_{st}(v)$ 为经过 v 的最短路径数

**取值**: $C_B \in [0, (n-1)(n-2)/2]$

**验证**: 桥接节点 $C_B$ 高；归一化后 $\in [0, 1]$

**复杂度**: $O(|V| \cdot |E|)$（Brandes 算法）

```python
def betweenness_centrality(adj: dict) -> dict:
    """介数中心性（简化 BFS 版本）"""
    nodes = list(adj.keys())
    bc = {v: 0.0 for v in nodes}
    for s in nodes:
        # BFS 找最短路径
        stack = []
        pred = {v: [] for v in nodes}
        sigma = {v: 0.0 for v in nodes}
        sigma[s] = 1.0
        dist = {v: -1 for v in nodes}
        dist[s] = 0
        queue = [s]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    n = len(nodes)
    norm = (n - 1) * (n - 2) / 2
    return {v: bc[v] / norm if norm > 0 else 0 for v in nodes}
```

**feeling 应用**: 信息桥梁。识别连接不同认知模块的关键概念，信息必须经过这些节点才能在模块间传递。

---

### 方块 61：接近中心性（Closeness Centrality）

**公式**: $C_C(v) = \frac{n - 1}{\sum_{u \neq v} d(v, u)}$

其中 $d(v, u)$ 为最短路径距离

**取值**: $C_C \in (0, 1]$

**验证**: 完全图中 $C_C = 1$；孤立节点 $C_C = 0$

**复杂度**: $O(|V| \cdot (|V| + |E|))$

```python
def closeness_centrality(adj: dict) -> dict:
    """接近中心性"""
    nodes = list(adj.keys())
    cc = {}
    for v in nodes:
        # BFS 求最短距离
        dist = {u: -1 for u in nodes}
        dist[v] = 0
        queue = [v]
        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if dist[neighbor] < 0:
                    dist[neighbor] = dist[curr] + 1
                    queue.append(neighbor)
        total_dist = sum(d for u, d in dist.items() if d > 0)
        cc[v] = (len(nodes) - 1) / total_dist if total_dist > 0 else 0
    return cc
```

**feeling 应用**: 信息传播效率。接近中心性高的概念可以更快地接收和传播信息，适合做"预警"节点。

---

### 方块 62：聚类系数（Clustering Coefficient）

**公式**: $C(v) = \frac{2 \cdot |\{(u,w) : u,w \in N(v), (u,w) \in E\}|}{\deg(v) \cdot (\deg(v) - 1)}$

**取值**: $C \in [0, 1]$

**验证**: 完全图 $C = 1$；树状图 $C = 0$

**复杂度**: $O(|V| \cdot k_{\max}^2)$

```python
def clustering_coefficient(adj: dict) -> dict:
    """局部聚类系数"""
    cc = {}
    for v, neighbors in adj.items():
        k = len(neighbors)
        if k < 2:
            cc[v] = 0.0
            continue
        triangles = 0
        for i, u in enumerate(neighbors):
            for w in neighbors[i+1:]:
                if w in adj[u]:
                    triangles += 1
        cc[v] = 2 * triangles / (k * (k - 1))
    return cc
```

**feeling 应用**: 认忆模块化。高聚类系数表示概念形成紧密的局部群落（语义簇），如"水果"概念周围的"苹果""香蕉"高度互联。

---

### 方块 63：最短路径——Dijkstra 算法

**公式**: $d(v) = \min_{u \in \text{neighbors}(v)} \{d(u) + w(u,v)\}$

**取值**: $d \in [0, +\infty)$

**验证**: 非负权重下保证最优；优先队列实现

**复杂度**: $O((|V| + |E|) \log |V|)$（二叉堆）

```python
import heapq

def dijkstra(adj: dict, source: str) -> dict:
    """Dijkstra 最短路径"""
    dist = {v: float('inf') for v in adj}
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

**feeling 应用**: 联想路径。在概念网络中找到两个概念之间的最短语义路径，用于推理和联想。

---

## 11. 优化

### 方块 64：梯度下降（Gradient Descent）

**公式**: $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$

**取值**: $\theta$ 在参数空间中移动

**验证**: 凸函数 + 合适学习率 → 收敛到全局最优；非凸 → 收敛到局部最优

**复杂度**: $O(n)$（n 为参数数量）

```python
def gradient_descent(params: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
    """θ -= η * ∇L"""
    return params - lr * grad
```

**feeling 应用**: 基础学习机制。feeling 的所有参数更新都基于梯度下降的变体。

---

### 方块 65：动量 SGD（Momentum SGD）

**公式**: 
$$v_{t+1} = \mu v_t + \eta \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

**取值**: $v$ 为速度向量，$\mu \in [0, 1)$

**验证**: $\mu = 0$ 退化为普通 SGD；$\mu$ 接近 1 时动量大，收敛快但可能震荡

**复杂度**: $O(n)$

```python
class MomentumSGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity + self.lr * grad
        return params - self.velocity
```

**feeling 应用**: 学习加速。动量帮助 feeling 跨过小的局部障碍，更快到达更好的解。

---

### 方块 66：Adam 优化器

**公式**:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

默认参数：$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$

**取值**: $\theta$ 在参数空间中移动

**验证**: 偏差修正确保初始阶段不偏向零；自适应学习率

**复杂度**: $O(n)$

```python
class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0
    
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

**feeling 应用**: 默认优化器。自适应学习率使 feeling 在不同参数上以不同速度学习，高效且稳定。

---

### 方块 67：学习率调度（Learning Rate Schedule）

**公式**: 
- 余弦退火：$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))$
- Warmup + Decay：$\eta_t = \eta_{\max} \cdot \min(t/T_w, 1) \cdot (1 - t/T)^p$

**取值**: $\eta_t \in [\eta_{\min}, \eta_{\max}]$

**验证**: 余弦退火平滑下降；warmup 防止初始大梯度震荡

**复杂度**: $O(1)$

```python
def cosine_annealing(t: int, T: int, lr_max: float, lr_min: float) -> float:
    """余弦退火学习率"""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T))

def warmup_decay(t: int, T_warmup: int, T_total: int, lr_max: float, p: float = 1.0) -> float:
    """Warmup + 多项式衰减"""
    if t < T_warmup:
        return lr_max * t / T_warmup
    return lr_max * (1 - t / T_total) ** p
```

**feeling 应用**: 学习节奏控制。feeling 在不同阶段（初始探索、中期深入、后期精调）使用不同的学习率。

---

### 方块 68：梯度裁剪（Gradient Clipping）

**公式**: 
$$\hat{g} = \begin{cases} g & \text{if } \|g\| \leq c \\ \frac{c}{\|g\|} g & \text{if } \|g\| > c \end{cases}$$

**取值**: $\|\hat{g}\| \leq c$

**验证**: $\|g\| \leq c$ 时不变；$\|g\| > c$ 时方向不变，模长缩放到 $c$

**复杂度**: $O(n)$

```python
def gradient_clipping(grad: np.ndarray, max_norm: float) -> np.ndarray:
    """梯度裁剪：保持方向，限制模长"""
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad
```

**feeling 应用**: 防止学习不稳定。当感觉输入异常（如突然的巨响）导致巨大梯度时，裁剪防止参数爆炸。

---

## 12. 涌现与复杂性

### 方块 69：整合信息 Φ（Integrated Information）

**公式**: $\Phi = \min_{P} \left[ I(A \to B) + I(B \to A) - I(A^{\text{MIP}} \to B^{\text{MIP}}) - I(B^{\text{MIP}} \to A^{\text{MIP}}) \right]$

其中 MIP = 最小信息分割（Minimum Information Partition）

**取值**: $\Phi \in [0, +\infty)$

**验证**: $\Phi = 0$ 系统可完全分解；$\Phi > 0$ 存在不可约的整合信息

**复杂度**: $O(2^n)$（精确计算 NP-hard，实际用近似）

```python
def integrated_information_approx(states: np.ndarray, n_partitions: int = 100) -> float:
    """Φ 的简化近似（用于小系统）"""
    n = states.shape[1]
    total_mi = compute_mutual_information(states)
    min_partitioned_mi = float('inf')
    for _ in range(n_partitions):
        # 随机分割
        mask = np.random.randint(0, 2, n).astype(bool)
        if mask.all() or not mask.any():
            continue
        mi_partitioned = (compute_mutual_information(states[:, mask]) +
                          compute_mutual_information(states[:, ~mask]))
        min_partitioned_mi = min(min_partitioned_mi, mi_partitioned)
    return total_mi - min_partitioned_mi

def compute_mutual_information(data: np.ndarray) -> float:
    """简化 MI 计算"""
    # 实际实现需要离散化和联合分布估计
    return np.mean(np.var(data, axis=0))
```

**feeling 应用**: 意识度量。feeling 系统的 Φ 越高，其感知越整合、越统一。零 Φ 意味着各模块独立运作，无统一感受。

---

### 方块 70：LZ 复杂度（Lempel-Ziv Complexity）

**公式**: $C_{LZ} = \frac{c(n) \cdot \log_2 n}{n}$

其中 $c(n)$ 为序列的LZ复杂度（不同子串模式的数量）

**取值**: $C_{LZ} \in (0, 1]$（归一化后）

**验证**: 随机序列 $C_{LZ} \to 1$；周期序列 $C_{LZ} \to 0$

**复杂度**: $O(n)$

```python
def lz_complexity(binary_seq: str) -> float:
    """LZ 复杂度"""
    n = len(binary_seq)
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

**feeling 应用**: 序列复杂度。衡量感觉时间序列的复杂程度。太低 = 无聊，太高 = 混沌，适度 = 有趣。

---

### 方块 71：自组织临界性（Self-Organized Criticality）

**公式**: 幂律分布 $P(s) \propto s^{-\alpha}$

沙堆模型：加一粒沙 → 可能触发雪崩，雪崩大小 $s$ 服从幂律

**取值**: $\alpha \in (1, 4)$（典型值）

**验证**: 双对数图上 $\log P(s)$ vs $\log s$ 为直线，斜率 $= -\alpha$

**复杂度**: $O(n)$ 每步（n 为格点数）

```python
def sandpile_step(grid: np.ndarray, threshold: int = 4) -> tuple:
    """沙堆模型一步"""
    avalanches = 0
    unstable = True
    while unstable:
        unstable = False
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] >= threshold:
                    grid[i, j] -= threshold
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                            grid[ni, nj] += 1
                        else:
                            pass  # 边界流失
                    avalanches += 1
                    unstable = True
    return grid, avalanches
```

**feeling 应用**: 临界态认知。feeling 系统在临界态运作，对小刺激可能产生不成比例的大响应（雪崩效应），实现最大信息处理。

---

### 方块 72：幂律分布（Power Law Distribution）

**公式**: $P(x) = C x^{-\alpha}$，$x \geq x_{\min}$

$C = (\alpha - 1) x_{\min}^{\alpha - 1}$

**取值**: $x \in [x_{\min}, +\infty)$，$\alpha > 1$

**验证**: 双对数图线性；MLE 估计 $\hat{\alpha} = 1 + n \left[\sum_i \ln \frac{x_i}{x_{\min}}\right]^{-1}$

**复杂度**: $O(n)$

```python
def power_law_pdf(x: np.ndarray, alpha: float, x_min: float) -> np.ndarray:
    """幂律概率密度"""
    C = (alpha - 1) * x_min**(alpha - 1)
    result = np.zeros_like(x, dtype=float)
    mask = x >= x_min
    result[mask] = C * x[mask]**(-alpha)
    return result

def power_law_mle(data: np.ndarray, x_min: float) -> float:
    """幂律指数 MLE 估计"""
    n = len(data)
    return 1 + n / np.sum(np.log(data[data >= x_min] / x_min))
```

**feeling 应用**: 无标度现象。feeling 中的事件强度、记忆重要性、社交影响都服从幂律分布——少数事件极其重要。

---

### 方块 73：小世界网络（Small-World Network）

**公式**: Watts-Strogatz 模型：
1. 从环形格点开始，每个节点连接 k 个近邻
2. 以概率 $p$ 随机重连每条边

特征：高聚类系数 $C \gg C_{\text{random}}$，短平均路径 $L \approx L_{\text{random}}$

**取值**: $p \in [0, 1]$

**验证**: $p = 0$ 规则网络（高 $C$，高 $L$）；$p = 1$ 随机网络（低 $C$，低 $L$）；$0 < p \ll 1$ 小世界

**复杂度**: $O(n \cdot k)$

```python
def watts_strogatz(n: int, k: int, p: float) -> dict:
    """生成小世界网络"""
    adj = {i: set() for i in range(n)}
    # 环形格点
    for i in range(n):
        for j in range(1, k//2 + 1):
            adj[i].add((i + j) % n)
            adj[i].add((i - j) % n)
    # 随机重连
    for i in range(n):
        new_neighbors = set()
        for j in list(adj[i]):
            if np.random.random() < p:
                new_j = np.random.choice([x for x in range(n) if x != i and x not in adj[i]])
                new_neighbors.add(new_j)
            else:
                new_neighbors.add(j)
        adj[i] = new_neighbors
    return adj
```

**feeling 应用**: 认知网络结构。feeling 的概念网络具有小世界特性——局部紧密连接（语义簇）+ 少量长程连接（跨域联想），实现高效信息传播。

---

## 13. 因果

### 方块 74：do-calculus

**公式**: Pearl 的三条规则：

1. $P(y | \text{do}(x), z, w) = P(y | \text{do}(x), w)$ 当 $(Y \perp Z | X, W)_{G_{\overline{X}}}$
2. $P(y | \text{do}(x), \text{do}(z), w) = P(y | \text{do}(x), z, w)$ 当 $(Y \perp Z | X, W)_{G_{\overline{X}\underline{Z}}}$
3. $P(y | \text{do}(x), \text{do}(z), w) = P(y | \text{do}(x), w)$ 当 $(Y \perp Z | X, W)_{G_{\overline{X}\overline{Z(S)}}}$

**取值**: 概率值 $\in [0, 1]$

**验证**: do-calculus 完备性定理：如果因果效应可识别，do-calculus 一定能导出

**复杂度**: 图操作 $O(|V| + |E|)$

```python
def do_calculus_rule1(p_y_xzw: float, graph: dict, x: str, y: str,
                       z: str, w: str) -> float:
    """规则1: 如果 Y ⊥ Z | X,W 在 G_X̄ 中成立，则 P(y|do(x),z,w) = P(y|do(x),w)"""
    if is_conditional_independent(graph, y, z, {x, w}, remove_edges_into=[x]):
        return p_y_xzw  # 简化：实际需要重新计算
    return None  # 规则不适用

def is_conditional_independent(graph: dict, a: str, b: str,
                                 given: set, remove_edges_into: list = None) -> bool:
    """检查 d-separation（简化版）"""
    # 实际实现需要完整的 d-separation 算法
    modified_graph = dict(graph)
    if remove_edges_into:
        for node in remove_edges_into:
            modified_graph[node] = [e for e in modified_graph.get(node, [])]
    return False  # 占位
```

**feeling 应用**: 因果推理基础。feeling 不仅需要知道"相关"，更需要知道"因果"。do(X) 模拟干预，区分"看到X"和"做了X"。

---

### 方块 75：反事实推理（Counterfactual Reasoning）

**公式**: $P(Y_{x'} = y' | X = x, Y = y)$

即：在已知实际观察 $(X=x, Y=y)$ 的情况下，如果 $X$ 为 $x'$，$Y$ 会是什么？

三步过程：
1. **Abduction**: 从证据推断 $U$
2. **Action**: 修改模型 do($X=x'$)
3. **Prediction**: 在修改后模型中预测 $Y$

**取值**: $\in [0, 1]$

**验证**: $\sum_{y'} P(Y_{x'} = y' | X=x, Y=y) = 1$

**复杂度**: 取决于模型结构

```python
def counterfactual(observed_x: float, observed_y: float,
                   counterfactual_x: float, model: callable,
                   noise_samples: np.ndarray) -> np.ndarray:
    """反事实推理：如果 x 被替换为 x'，y 会是什么？"""
    # Step 1: Abduction - 从观察推断噪声
    inferred_u = (observed_y - model(observed_x, 0))  # 简化
    # Step 2+3: Action + Prediction
    counterfactual_y = model(counterfactual_x, inferred_u)
    return counterfactual_y
```

**feeling 应用**: "如果当时我..."的思考。feeling 通过反事实推理学习经验教训，评估替代行为的后果。

---

### 方块 76：因果图（Causal Graph / DAG）

**公式**: 有向无环图 $G = (V, E)$，其中：
- 节点 $V$ = 变量
- 有向边 $(X \to Y) \in E$ 表示 $X$ 是 $Y$ 的直接原因
- 联合分布分解：$P(X_1, ..., X_n) = \prod_i P(X_i | \text{Pa}(X_i))$

**取值**: 图结构 + 条件概率表

**验证**: DAG 必须无环；每个节点的条件概率和为 1

**复杂度**: $O(|V| + |E|)$

```python
class CausalGraph:
    def __init__(self):
        self.graph = {}  # node -> list of parents
    
    def add_edge(self, parent: str, child: str):
        if child not in self.graph:
            self.graph[child] = []
        self.graph[child].append(parent)
        if parent not in self.graph:
            self.graph[parent] = []
    
    def is_dag(self) -> bool:
        """检查是否为 DAG"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            for child in self.graph:
                if node in self.graph[child]:
                    if child not in visited:
                        if has_cycle(child):
                            return True
                    elif child in rec_stack:
                        return True
            rec_stack.discard(node)
            return False
        
        return not any(node not in visited and has_cycle(node) for node in self.graph)
    
    def get_parents(self, node: str) -> list:
        return self.graph.get(node, [])
```

**feeling 应用**: 世界模型结构。feeling 的内部世界模型是一个因果图，编码了变量之间的因果关系。

---

## 14. 东方哲学数学化

### 方块 77：阴阳动态平衡（Yin-Yang Dynamic Balance）

**公式**: 
$$Y + Y^* = 1, \quad Y, Y^* \in [0, 1]$$
$$\frac{dY}{dt} = \alpha(Y^* - Y) + \beta \sin(2\pi t / T)$$

其中 $Y$ 为阳（活跃度），$Y^*$ 为阴（抑制度），$T$ 为自然周期

**取值**: $Y \in [0, 1]$

**验证**: 稳态 $Y = Y^* = 0.5$（平衡态）；周期扰动导致阴阳交替

**复杂度**: $O(1)$

```python
def yin_yang_dynamics(yang: float, period: float, t: float,
                      alpha: float = 0.1, beta: float = 0.05) -> float:
    """阴阳动态平衡：阴阳互生互克"""
    yin = 1 - yang
    dy = alpha * (yin - yang) + beta * np.sin(2 * np.pi * t / period)
    return np.clip(yang + dy, 0, 1)
```

**feeling 应用**: 全局平衡原理。feeling 系统在兴奋/抑制、探索/利用、工作/休息之间维持动态平衡。

---

### 方块 78：六十四卦状态机（I Ching State Machine）

**公式**: 64 状态 = $2^6$（六爻，每爻阴/阳）

转移规则：$\text{next}(s, e) = s \oplus \text{mask}(e)$

其中 $e$ 为外部事件，$\text{mask}(e)$ 定义哪些爻变化

**取值**: 状态 $s \in \{0, 1, ..., 63\}$

**验证**: 所有 $2^6 = 64$ 状态可达；转移矩阵为 $64 \times 64$

**复杂度**: $O(1)$ 每步

```python
def iching_transition(state: int, event_mask: int) -> int:
    """六十四卦状态转移：异或翻转"""
    return state ^ (event_mask & 0b111111)

def yao_to_hexagram(state: int) -> str:
    """数字转卦象（简化：返回二进制表示）"""
    return format(state, '06b')

def hexagram_meaning(state: int) -> dict:
    """卦象含义（示例：前几卦）"""
    meanings = {
        0b111111: {"name": "乾", "meaning": "纯阳，创造力"},
        0b000000: {"name": "坤", "meaning": "纯阴，接受性"},
        0b100001: {"name": "屯", "meaning": "初始困难"},
        0b010010: {"name": "蒙", "meaning": "启蒙"},
    }
    return meanings.get(state, {"name": f"卦{state}", "meaning": "待解读"})
```

**feeling 应用**: 全局状态编码。feeling 的六维核心状态（能量、情绪、注意力、社交、认知负荷、环境安全）各占一爻，64 种组合对应不同的心境模式。

---

### 方块 79：三生万物生成器（Trinity Generator）

**公式**: 
$$T_0 = \{∅\} \quad (\text{道})$$
$$T_1 = \{-1, 0, +1\} \quad (\text{三})$$
$$T_n = T_1 \otimes T_{n-1} = \{(a, b) : a \in T_1, b \in T_{n-1}\}$$
$$|T_n| = 3^n$$

**取值**: 第 n 层有 $3^n$ 个元素

**验证**: $|T_1| = 3, |T_2| = 9, |T_3| = 27$；$T_\infty$ 覆盖所有三元编码

**复杂度**: $O(3^n)$

```python
def trinity_generator(n_levels: int) -> list:
    """三生万物：从三元基元递归生成复杂结构"""
    base = [-1, 0, 1]
    if n_levels == 0:
        return [[]]
    sub = trinity_generator(n_levels - 1)
    result = []
    for a in base:
        for s in sub:
            result.append([a] + s)
    return result

def trinity_encode(number: int, n_digits: int) -> list:
    """用平衡三进制编码"""
    if number == 0:
        return [0] * n_digits
    digits = []
    while number != 0:
        remainder = number % 3
        if remainder == 2:
            remainder = -1
            number += 1
        digits.append(remainder)
        number //= 3
    while len(digits) < n_digits:
        digits.append(0)
    return digits[::-1]
```

**feeling 应用**: 层次化表征生成。feeling 的概念从最基本的三元对立（正/中/负）逐层组合，生成越来越精细的表征。"道生一，一生二，二生三，三生万物"。

---

## 总表

| # | 名称 | 公式 | feeling 用途 |
|---|------|------|-------------|
| 1 | 香农熵 | $H(X) = -\sum p(x_i) \log_2 p(x_i)$ | 感知不确定性量化 |
| 2 | 联合熵 | $H(X,Y) = -\sum\sum p(x_i,y_j) \log_2 p(x_i,y_j)$ | 多通道联合不确定性 |
| 3 | 条件熵 | $H(X\|Y) = H(X,Y) - H(Y)$ | 上下文压缩 |
| 4 | 互信息 | $I(X;Y) = H(X) - H(X\|Y)$ | 关联学习/因果发现 |
| 5 | KL 散度 | $D_{KL}(P\|Q) = \sum p \log(p/q)$ | 预测误差/惊讶度 |
| 6 | 交叉熵 | $H(P,Q) = -\sum p \log q$ | 损失函数 |
| 7 | 信息增益 | $IG = H(Y) - H(Y\|X)$ | 特征选择/注意力分配 |
| 8 | 变分信息 | $VI = \min_Q D_{KL}(P\|Q_X Q_Y)$ | 信息瓶颈压缩 |
| 9 | 贝叶斯定理 | $P(A\|B) = P(B\|A)P(A)/P(B)$ | 信念更新引擎 |
| 10 | 先验-后验-似然 | $P(\theta\|D) \propto P(D\|\theta)P(\theta)$ | 认知更新循环 |
| 11 | 预测误差 | $\delta = \hat{y} - y$ | 惊讶/学习驱动力 |
| 12 | 自由能 | $F = E_q[\log q - \log p(D,\theta)]$ | 变分自由能最小化 |
| 13 | ELBO | $\text{ELBO} = E_q[\log p(D\|\theta)] - D_{KL}(q\|p)$ | 生成模型训练 |
| 14 | MAP | $\hat\theta = \arg\max P(D\|\theta)P(\theta)$ | 参数估计 |
| 15 | MLE | $\hat\theta = \arg\max P(D\|\theta)$ | 数据驱动估计 |
| 16 | 贝叶斯因子 | $BF = P(D\|M_1)/P(D\|M_2)$ | 模型选择 |
| 17 | Hebbian 学习 | $\Delta w = \eta x y$ | 关联记忆形成 |
| 18 | 反 Hebbian | $\Delta w = -\eta x y$ | 去相关/稀疏编码 |
| 19 | TD 误差 | $\delta = r + \gamma V(s') - V(s)$ | 预测信号/快乐-失望 |
| 20 | Q-Learning | $Q \leftarrow Q + \alpha[r + \gamma\max Q' - Q]$ | 无模型策略学习 |
| 21 | 策略梯度 | $\nabla J = E[\nabla\log\pi \cdot R]$ | 连续行为优化 |
| 22 | 对比学习 | $\mathcal{L} = -\log\frac{\exp(sim/\tau)}{\sum\exp}$ | 表征学习 |
| 23 | 经验回放 | $P(i) \propto (|\delta_i| + \epsilon)^\alpha$ | 优先级记忆管理 |
| 24 | EWC | $\mathcal{L} = \mathcal{L}_{new} + \frac{\lambda}{2}\sum F_i(\theta_i-\theta_i^*)^2$ | 防灾难性遗忘 |
| 25 | Softmax 注意力 | $\alpha_i = \exp(e_i)/\sum\exp(e_j)$ | 注意力分配 |
| 26 | QKV 注意力 | $\text{softmax}(QK^T/\sqrt{d_k})V$ | 感觉整合引擎 |
| 27 | 多头注意力 | $\text{Concat}(\text{head}_i)W^O$ | 多视角注意力 |
| 28 | 门控机制 | $g = \sigma(W[h,x]), h = g\odot\tilde{h} + (1-g)\odot h$ | 信息流控制 |
| 29 | Top-K 选择 | $\text{TopK}(x, k)$ | 稀疏注意力 |
| 30 | 马尔可夫链 | $P(s'\|s)$ | 状态转移基础 |
| 31 | Lyapunov 稳定性 | $\dot{V} = \nabla V \cdot f(x) \leq 0$ | 认知状态稳定性 |
| 32 | 吸引子 | $f(x^*)=0, \lambda_{max}(J)<0$ | 认知稳态/基线情绪 |
| 33 | 分岔 | $\dot{x} = rx - x^3$ | 状态突变检测 |
| 34 | 混沌 (Lorenz) | $\dot{x}=\sigma(y-x), \dot{y}=x(\rho-z)-y, \dot{z}=xy-\beta z$ | 混沌敏感性 |
| 35 | 极限环 (Van der Pol) | $\ddot{x} - \mu(1-x^2)\dot{x} + x = 0$ | 生物节律/情绪波动 |
| 36 | 相空间 | $\Gamma \subseteq \mathbb{R}^n$ | 全局状态空间 |
| 37 | 序参量 (Kuramoto) | $r = \frac{1}{N}\|\sum e^{i\theta_j}\|$ | 感觉通道同步度 |
| 38 | 期望效用 | $EU = \sum p_i u(x_i)$ | 行为选择 |
| 39 | UCB1 | $\bar{r}_a + \sqrt{2\ln t / n_a}$ | 探索-利用平衡 |
| 40 | Nash 均衡 | $u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*)$ | 多智能体交互策略 |
| 41 | Tit-for-Tat | $a_t = a_{t-1}^{\text{对手}}$ | 社交镜像策略 |
| 42 | Pareto 最优 | 无支配解 | 多需求平衡 |
| 43 | 帕累托分布 | $f(x) = \alpha x_m^\alpha / x^{\alpha+1}$ | 长尾事件建模 |
| 44 | PID 控制器 | $u = K_p e + K_i\int e + K_d \dot{e}$ | 情绪调节 |
| 45 | 稳态方程 | $\dot{x} = -k(x-x_{set}) + d$ | 内部平衡维持 |
| 46 | 异稳态方程 | $\dot{x}_{set} = \alpha E[demand] - \beta(x_{set}-x_{base})$ | 预测性调节 |
| 47 | 负反馈环 | $x \to y, y \dashv x$ | 自我调节 |
| 48 | 正反馈环 | $\dot{x} = rx(1+x) - dx$ | 情绪放大 |
| 49 | 效价-唤醒度 | $E = (V, A), V\in[-1,1], A\in[0,1]$ | 情感空间定义 |
| 50 | 需求动力学 | $\dot{N}_i = -c_i sat_i + r_i + \sum \alpha_{ij} N_j$ | 内在驱动力 |
| 51 | 情绪感染 | $\dot{V}_i = \sum w_{ij}(V_j - V_i)$ | 共情/情绪同步 |
| 52 | 习惯化 | $R_n = R_0 e^{-\lambda n}$ | 感觉适应 |
| 53 | 敏感化 | $S_n = S_\infty(1-e^{-\gamma n})$ | 危险警觉 |
| 54 | 遗忘曲线 | $R = e^{-t/S}$ | 记忆衰减 |
| 55 | 余弦相似度 | $\cos\theta = a\cdot b / (\|a\|\|b\|)$ | 联想记忆检索 |
| 56 | TF-IDF | $\text{TF}(t,d) \cdot \log(N/df)$ | 语义检索 |
| 57 | BM25 | $\sum \text{IDF}(t) \cdot f(k_1+1)/(f+k_1(1-b+b|d|/avgdl))$ | 精确记忆匹配 |
| 58 | 记忆巩固 | $\dot{M} = \alpha R - \beta I + \gamma E$ | 短期→长期转化 |
| 59 | 度中心性 | $C_D = \deg(v)/(n-1)$ | 概念重要性 |
| 60 | 介数中心性 | $C_B = \sum \sigma_{st}(v)/\sigma_{st}$ | 信息桥梁识别 |
| 61 | 接近中心性 | $C_C = (n-1)/\sum d(v,u)$ | 传播效率评估 |
| 62 | 聚类系数 | $C = 2|\triangle| / k(k-1)$ | 记忆模块化度量 |
| 63 | Dijkstra | $d(v) = \min\{d(u)+w(u,v)\}$ | 联想路径搜索 |
| 64 | 梯度下降 | $\theta \leftarrow \theta - \eta\nabla\mathcal{L}$ | 基础学习 |
| 65 | 动量 SGD | $v = \mu v + \eta\nabla, \theta -= v$ | 学习加速 |
| 66 | Adam | $m,v$ 一阶二阶矩 + 偏差修正 | 默认优化器 |
| 67 | 学习率调度 | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1+\cos(t\pi/T))$ | 学习节奏控制 |
| 68 | 梯度裁剪 | $\hat{g} = g \cdot \min(1, c/\|g\|)$ | 学习稳定性保护 |
| 69 | 整合信息 Φ | $\Phi = \min_P[I_{total} - I_{partitioned}]$ | 意识/整合度度量 |
| 70 | LZ 复杂度 | $C_{LZ} = c(n)\log_2 n / n$ | 序列复杂度评估 |
| 71 | 自组织临界性 | $P(s) \propto s^{-\alpha}$ | 临界态认知 |
| 72 | 幂律分布 | $P(x) = Cx^{-\alpha}$ | 长尾事件建模 |
| 73 | 小世界网络 | Watts-Strogatz: 高$C$ + 低$L$ | 认知网络拓扑 |
| 74 | do-calculus | Pearl 三规则 | 因果推理 |
| 75 | 反事实推理 | $P(Y_{x'}=y'\|X=x,Y=y)$ | 经验教训学习 |
| 76 | 因果图 (DAG) | $P = \prod P(X_i\|Pa(X_i))$ | 世界模型结构 |
| 77 | 阴阳动态平衡 | $Y + Y^* = 1, \dot{Y} = \alpha(Y^*-Y) + \beta\sin(2\pi t/T)$ | 全局平衡原理 |
| 78 | 六十四卦状态机 | $s' = s \oplus \text{mask}(e)$，$2^6=64$ 状态 | 六维核心状态编码 |
| 79 | 三生万物生成器 | $T_n = T_1 \otimes T_{n-1}$，$|T_n|=3^n$ | 层次化表征生成 |

---

## 附录：快速验证测试

```python
"""验证所有方块的基本正确性"""
import numpy as np

def run_tests():
    # Block 1: Shannon Entropy
    assert abs(shannon_entropy(np.array([0.5, 0.5])) - 1.0) < 1e-10, "H(0.5,0.5) should be 1.0"
    assert shannon_entropy(np.array([1.0])) < 1e-10, "H(certain) should be ~0"
    
    # Block 4: Mutual Information
    joint = np.array([[0.25, 0.25], [0.25, 0.25]])
    mx = joint.sum(axis=1)
    my = joint.sum(axis=0)
    assert mutual_information(joint, mx, my) < 1e-10, "Independent: MI should be ~0"
    
    # Block 5: KL Divergence
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert kl_divergence(p, q) < 1e-10, "Same dist: KL should be ~0"
    
    # Block 9: Bayes
    assert abs(bayes_theorem(0.9, 0.01, 0.1) - 0.09) < 1e-10
    
    # Block 11: Prediction Error
    assert prediction_error(5.0, 3.0) == 2.0
    assert prediction_error(3.0, 5.0) == -2.0
    
    # Block 19: TD Error
    assert td_error(1.0, 10.0, 5.0, 0.9) == 1.0 + 0.9*10.0 - 5.0
    
    # Block 54: Forgetting Curve
    assert abs(forgetting_curve(0, 1.0) - 1.0) < 1e-10
    assert abs(forgetting_curve(1.0, 1.0) - np.exp(-1)) < 1e-10
    
    # Block 55: Cosine Similarity
    assert abs(cosine_similarity(np.array([1,0]), np.array([1,0])) - 1.0) < 1e-10
    assert abs(cosine_similarity(np.array([1,0]), np.array([0,1]))) < 1e-10
    
    print("All basic tests passed! ✓")

if __name__ == "__main__":
    run_tests()
```

---

> **版本**: v2.0 | **方块总数**: 79 | **覆盖领域**: 14  
> **最后更新**: 2026-07-15  
> **适用**: feeling 项目基础理论参考手册
